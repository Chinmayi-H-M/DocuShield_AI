import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("docushield.diagnostic")

# Setup environment paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # backend/
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Import codebase services
import sys
sys.path.append(BASE_DIR)
from app.services.dataset_loader import load_dataset_splits, get_data_loaders, FraudDetectionDataset
from app.services.train_model import get_model

def run_diagnostics():
    logger.info("==================================================")
    logger.info("STARTING COMPREHENSIVE ML PIPELINE DIAGNOSTICS")
    logger.info("==================================================")

    # ----------------------------------------------------
    # 1. Dataset Loading Diagnostic
    # ----------------------------------------------------
    logger.info("\n--- STEP 1: Dataset Loading Diagnostic ---")
    genuine_dir = os.path.join(DATASET_DIR, "genuine")
    tampered_dir = os.path.join(DATASET_DIR, "tampered")
    
    import glob
    supported_exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    
    genuine_files = []
    seen_gen = set()
    for ext in supported_exts:
        for f in glob.glob(os.path.join(genuine_dir, ext)):
            abs_p = os.path.abspath(f)
            norm_p = os.path.normcase(abs_p)
            if norm_p not in seen_gen:
                seen_gen.add(norm_p)
                genuine_files.append(abs_p)
        
    tampered_files = []
    seen_tamp = set()
    for ext in supported_exts:
        for f in glob.glob(os.path.join(tampered_dir, ext)):
            abs_p = os.path.abspath(f)
            norm_p = os.path.normcase(abs_p)
            if norm_p not in seen_tamp:
                seen_tamp.add(norm_p)
                tampered_files.append(abs_p)
        
    logger.info(f"Total Genuine image files found: {len(genuine_files)}")
    logger.info(f"Total Tampered image files found: {len(tampered_files)}")
    logger.info(f"Total dataset files physically found: {len(genuine_files) + len(tampered_files)}")

    corrupted_files = []
    for file_path in genuine_files + tampered_files:
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            logger.error(f"Corrupted or unreadable image: {file_path} - Error: {e}")
            corrupted_files.append((file_path, str(e)))
            
    logger.info(f"Total corrupted or unreadable images: {len(corrupted_files)}")
    if corrupted_files:
        logger.warning(f"Corrupted files details: {corrupted_files[:10]}")
    else:
        logger.info("No corrupted or unreadable images found in directories.")

    # ----------------------------------------------------
    # 2. Label Verification Diagnostic
    # ----------------------------------------------------
    logger.info("\n--- STEP 2: Label Verification Diagnostic ---")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = load_dataset_splits(DATASET_DIR)
    
    # Check labels alignment
    genuine_labeled_correctly = True
    tampered_labeled_correctly = True
    
    for path, label in zip(train_paths, train_labels):
        parent_dir = os.path.basename(os.path.dirname(path)).lower()
        if parent_dir == "genuine" and label != 0:
            genuine_labeled_correctly = False
            logger.error(f"LABEL INVERSION: Genuine document has label {label}! Path: {path}")
        elif parent_dir == "tampered" and label != 1:
            tampered_labeled_correctly = False
            logger.error(f"LABEL INVERSION: Tampered document has label {label}! Path: {path}")
            
    if genuine_labeled_correctly and tampered_labeled_correctly:
        logger.info("Label assignment verified successfully: Genuine = 0, Tampered = 1.")
    else:
        logger.error("WARNING: Label inversion or incorrect mapping detected!")

    # ----------------------------------------------------
    # 3. Train/Test Split Diagnostic
    # ----------------------------------------------------
    logger.info("\n--- STEP 3: Train/Val/Test Split Diagnostic ---")
    logger.info(f"Train set size: {len(train_paths)}")
    logger.info(f"Val set size: {len(val_paths)}")
    logger.info(f"Test set size: {len(test_paths)}")
    
    def print_distribution(name, labels):
        total = len(labels)
        if total == 0:
            logger.info(f"  {name} is empty.")
            return
        g_count = labels.count(0)
        t_count = labels.count(1)
        logger.info(f"  {name} set distribution: Genuine={g_count} ({g_count/total:.2%}), Tampered={t_count} ({t_count/total:.2%})")
        
    print_distribution("Train", train_labels)
    print_distribution("Val", val_labels)
    print_distribution("Test", test_labels)

    # ----------------------------------------------------
    # 4. Image Preprocessing Diagnostic
    # ----------------------------------------------------
    logger.info("\n--- STEP 4: Image Preprocessing Diagnostic ---")
    train_loader, val_loader, test_loader = get_data_loaders(DATASET_DIR, batch_size=4)
    if train_loader is not None:
        images, labels = next(iter(train_loader))
        logger.info(f"Batch image tensor shape: {images.shape}")
        logger.info(f"Batch labels: {labels}")
        logger.info(f"Pixel values - Min: {images.min().item():.4f}, Max: {images.max().item():.4f}")
        logger.info(f"Pixel values - Mean: {images.mean().item():.4f}, Std: {images.std().item():.4f}")
    else:
        logger.error("Dataloader initialization failed.")

    # ----------------------------------------------------
    # 5. Model Updates & Training Diagnostic
    # ----------------------------------------------------
    logger.info("\n--- STEP 5: Model Initialization and Weight Updates Diagnostic ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("resnet50").to(device)
    
    # Check parameter status
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    logger.info(f"Total model parameters: {total_params:,}")
    logger.info(f"Trainable model parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

    # Verify optimizer updates weight by running 1 gradient step
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0002)
    criterion = nn.BCEWithLogitsLoss()
    
    if train_loader is not None:
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        
        # Keep original weights
        fc_weights_before = model.fc[0].weight.clone().detach()
        
        model.train()
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        fc_weights_after = model.fc[0].weight.clone().detach()
        
        weights_updated = not torch.equal(fc_weights_before, fc_weights_after)
        if weights_updated:
            logger.info("SUCCESS: Model weights are updating successfully after one backward step.")
        else:
            logger.error("FAILURE: Model weights did NOT update after backward step!")
    else:
        logger.error("No training data loader available for training test step.")

    # ----------------------------------------------------
    # 6. Evaluation Diagnostic
    # ----------------------------------------------------
    logger.info("\n--- STEP 6: Evaluation Diagnostic ---")
    # Evaluate with current checkpoint
    model_path = os.path.join(MODELS_DIR, "model.pth")
    if os.path.exists(model_path):
        logger.info(f"Found saved model checkpoint at {model_path}. Loading weights...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            all_preds = []
            all_probs = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = model(images).squeeze(1)
                    probs = torch.sigmoid(outputs)
                    preds = (probs >= 0.5).int()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.numpy().astype(int))
                    
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, zero_division=0)
            rec = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            cm = confusion_matrix(all_labels, all_preds)
            
            logger.info(f"Evaluation on Test Set (size {len(all_labels)}):")
            logger.info(f"  Accuracy:  {acc:.4f}")
            logger.info(f"  Precision: {prec:.4f}")
            logger.info(f"  Recall:    {rec:.4f}")
            logger.info(f"  F1-Score:  {f1:.4f}")
            logger.info(f"  Confusion Matrix:\n{cm}")
            logger.info(f"  Predictions distribution: Genuine predictions={all_preds.count(0)}, Tampered predictions={all_preds.count(1)}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate saved model: {e}")
    else:
        logger.warning(f"No checkpoint found at {model_path}.")

    # ----------------------------------------------------
    # 7. Data Leakage & Logic Bugs Diagnostic
    # ----------------------------------------------------
    logger.info("\n--- STEP 7: Data Leakage & Logic Bugs Diagnostic ---")
    overlap = set(train_paths).intersection(set(test_paths))
    logger.info(f"Overlap between Train set and Test set (number of shared images): {len(overlap)}")
    if len(overlap) > 0:
        logger.error(f"DATA LEAKAGE WARNING: Train and Test set share {len(overlap)} files!")
        
    overlap_val = set(train_paths).intersection(set(val_paths))
    logger.info(f"Overlap between Train set and Val set: {len(overlap_val)}")
    
    # Verify dataset splits sizes match what's physically in genuine/tampered directories
    split_total = len(train_paths) + len(val_paths) + len(test_paths)
    dir_total = len(genuine_files) + len(tampered_files)
    logger.info(f"Total paths in splits: {split_total} vs Total files in folders: {dir_total}")
    
    if split_total != dir_total:
        logger.error(f"LOGIC BUG: The split dataset size ({split_total}) does not match physical directory size ({dir_total})!")
    else:
        logger.info("Split dataset size matches directory size.")

    # Check metrics files values
    for path in [os.path.join(BASE_DIR, "metrics.json"), os.path.join(MODELS_DIR, "metrics.json")]:
        if os.path.exists(path):
            with open(path, "r") as f:
                content = json.load(f)
                logger.info(f"Content of {path}: accuracy={content.get('accuracy')}, model_name={content.get('model_name')}, confusion_matrix={content.get('confusion_matrix')}")
        else:
            logger.warning(f"{path} does not exist.")

if __name__ == "__main__":
    run_diagnostics()
