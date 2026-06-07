import os
import json
import logging
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image, ImageDraw
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.services.dataset_loader import get_data_loaders

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docushield.ml.train")

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # backend/
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "model.pth")
METRICS_SAVE_PATH = os.path.join(BASE_DIR, "metrics.json")  # Saving in backend folder as per requirements
ALT_METRICS_SAVE_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Ensure models dir exists
os.makedirs(MODELS_DIR, exist_ok=True)


def get_model(model_name: str = "resnet50") -> nn.Module:
    """
    Initializes a transfer learning model with pretrained weights and modifies
    the classification head for binary fraud detection.
    """
    try:
        if model_name == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            )
            # Freeze base convolutional parameters except layer4 and fc for transfer learning
            for name, param in model.named_parameters():
                if "fc" in name or "layer4" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        elif model_name == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            )
            for name, param in model.named_parameters():
                if "classifier" in name or "features.8" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        elif model_name == "efficientnet_b3":
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            )
            for name, param in model.named_parameters():
                if "classifier" in name or "features.8" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        elif model_name == "mobilenet_v3":
            from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
            model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            num_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            )
            for name, param in model.named_parameters():
                if "classifier" in name or "features.16" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")
            
        logger.info(f"Initialized {model_name} with transfer learning head.")
        return model
    except Exception as e:
        logger.error(f"Error initializing model {model_name}: {e}. Fallback to resnet50 with pretrained=True.")
        try:
            model = models.resnet50(pretrained=True)
        except Exception:
            model = models.resnet50()
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        return model


def get_resnet50_model() -> nn.Module:
    """
    Backward-compatible wrapper.
    """
    return get_model("resnet50")


def generate_tiny_synthetic_dataset(dataset_dir: str):
    """
    Generates a tiny dummy dataset (15 genuine, 15 tampered) if empty,
    and updates the generator to process all images in backend/datasets/genuine
    and create a corresponding tampered version for each image.
    """
    import glob
    import random
    
    genuine_dir = os.path.join(dataset_dir, "genuine")
    tampered_dir = os.path.join(dataset_dir, "tampered")
    
    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(tampered_dir, exist_ok=True)
    
    # 1. Fallback: If genuine directory is completely empty, generate 15 fallback genuine images first
    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    genuine_files = []
    seen = set()
    for ext in supported_extensions:
        for f in glob.glob(os.path.join(genuine_dir, ext)):
            abs_p = os.path.abspath(f)
            norm_p = os.path.normcase(abs_p)
            if norm_p not in seen:
                seen.add(norm_p)
                genuine_files.append(abs_p)
        
    if len(genuine_files) == 0:
        logger.info("Genuine dataset folder is empty. Generating 15 fallback genuine images...")
        for i in range(15):
            img = Image.new("RGB", (224, 224), color=(240, 240, 240))
            d = ImageDraw.Draw(img)
            d.text((10, 10), f"Genuine Invoice #{1000+i}", fill=(0, 0, 0))
            d.text((10, 50), "Amount Due: $150.00", fill=(0, 0, 0))
            img.save(os.path.join(genuine_dir, f"genuine_{i}.png"))
        # Refresh the list
        seen = set()
        genuine_files = []
        for ext in supported_extensions:
            for f in glob.glob(os.path.join(genuine_dir, ext)):
                abs_p = os.path.abspath(f)
                norm_p = os.path.normcase(abs_p)
                if norm_p not in seen:
                    seen.add(norm_p)
                    genuine_files.append(abs_p)

    # 2. Process all images in backend/datasets/genuine and create a corresponding tampered version for each
    logger.info(f"Processing all {len(genuine_files)} images in genuine/ to generate tampered/ versions...")
    
    tampered_count = 0
    for file_path in genuine_files:
        try:
            basename = os.path.basename(file_path)
            tampered_file_path = os.path.join(tampered_dir, f"tampered_{basename}")
            
            # Do not overwrite existing tampered images if already created
            if os.path.exists(tampered_file_path):
                tampered_count += 1
                continue
                
            img = Image.open(file_path).convert("RGB")
            width, height = img.size
            
            tampered_img = img.copy()
            draw = ImageDraw.Draw(tampered_img)
            
            bg_color = (255, 255, 255)
            try:
                bg_color = tampered_img.getpixel((5, 5))
            except Exception:
                pass
                
            # Random combination of text overlay, number mod, date change, rect patch, and region shift
            techniques = ["text_overlay", "number_mod", "date_change", "rect_patch", "region_shift"]
            num_to_apply = random.randint(2, len(techniques))
            applied = random.sample(techniques, num_to_apply)
            
            for tech in applied:
                if tech == "text_overlay":
                    text = random.choice(["COPY", "PAST DUE", "VOID", "REVISED", "AUDIT", "UNAUTHORIZED", "FORGED"])
                    x = random.randint(int(width * 0.1), int(width * 0.7))
                    y = random.randint(int(height * 0.1), int(height * 0.7))
                    color = random.choice([(255, 0, 0), (0, 0, 255), (128, 128, 128), (0, 0, 0)])
                    draw.text((x, y), text, fill=color)
                    
                elif tech == "number_mod":
                    patch_w = random.randint(30, 80)
                    patch_h = random.randint(15, 30)
                    x = random.randint(int(width * 0.2), int(width * 0.7))
                    y = random.randint(int(height * 0.2), int(height * 0.8))
                    draw.rectangle([x, y, x + patch_w, y + patch_h], fill=bg_color)
                    new_num = f"${random.randint(1, 999)},{random.randint(100, 999)}.{random.randint(10, 99)}"
                    text_color = (0, 0, 0) if sum(bg_color)/3 > 128 else (255, 255, 255)
                    draw.text((x + 2, y + 2), new_num, fill=text_color)
                    
                elif tech == "date_change":
                    patch_w = random.randint(60, 100)
                    patch_h = random.randint(15, 30)
                    x = random.randint(int(width * 0.2), int(width * 0.7))
                    y = random.randint(int(height * 0.2), int(height * 0.8))
                    draw.rectangle([x, y, x + patch_w, y + patch_h], fill=bg_color)
                    new_date = f"{random.randint(1, 12):02d}/{random.randint(1, 28):02d}/{random.choice([2024, 2025, 2026])}"
                    text_color = (0, 0, 0) if sum(bg_color)/3 > 128 else (255, 255, 255)
                    draw.text((x + 2, y + 2), new_date, fill=text_color)
                    
                elif tech == "rect_patch":
                    patch_w = random.randint(40, 120)
                    patch_h = random.randint(20, 60)
                    x = random.randint(int(width * 0.15), int(width * 0.75))
                    y = random.randint(int(height * 0.15), int(height * 0.75))
                    patch_color = tuple(max(0, min(255, c + random.choice([-15, -10, 10, 15]))) for c in bg_color)
                    draw.rectangle([x, y, x + patch_w, y + patch_h], fill=patch_color)
                    
                elif tech == "region_shift":
                    patch_w = random.randint(40, 100)
                    patch_h = random.randint(15, 40)
                    x = random.randint(int(width * 0.2), int(width * 0.7))
                    y = random.randint(int(height * 0.2), int(height * 0.8))
                    crop = tampered_img.crop((x, y, x + patch_w, y + patch_h))
                    draw.rectangle([x, y, x + patch_w, y + patch_h], fill=bg_color)
                    dx = random.choice([-15, -10, 10, 15])
                    dy = random.choice([-15, -10, 10, 15])
                    tampered_img.paste(crop, (x + dx, y + dy))
                    
            # Save tampered version
            tampered_img.save(tampered_file_path)
            tampered_count += 1
        except Exception as e:
            logger.error(f"Failed to tamper image {file_path}: {e}")
            
    logger.info(f"Tampered dataset generated. Created/verified {tampered_count} tampered images.")


def train_model(epochs: int = 5, batch_size: int = 16, patience: int = 3, quick_test: bool = False) -> Dict[str, Any]:
    """
    Trains and compares ResNet50, EfficientNet-B0, EfficientNet-B3, and MobileNetV3 models.
    Automatically selects the best-performing model based on validation F1 score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for training: {device}")

    # Generate synthetic data if none exists
    generate_tiny_synthetic_dataset(DATASET_DIR)

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(DATASET_DIR, batch_size=batch_size)
    if train_loader is None or val_loader is None or test_loader is None:
        raise ValueError("Data loaders could not be initialized. Please check the dataset directories.")

    if quick_test:
        logger.info("Quick test mode enabled. Reducing dataset size for fast CPU verification.")
        train_loader.dataset.file_paths = train_loader.dataset.file_paths[:16]
        train_loader.dataset.labels = train_loader.dataset.labels[:16]
        val_loader.dataset.file_paths = val_loader.dataset.file_paths[:8]
        val_loader.dataset.labels = val_loader.dataset.labels[:8]
        test_loader.dataset.file_paths = test_loader.dataset.file_paths[:8]
        test_loader.dataset.labels = test_loader.dataset.labels[:8]

    # Calculate class weights for weighted loss
    train_labels = train_loader.dataset.labels
    num_genuine = sum(1 for label in train_labels if label == 0)
    num_tampered = sum(1 for label in train_labels if label == 1)
    
    pos_weight_val = num_genuine / max(num_tampered, 1)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    models_to_compare = ["resnet50"]
    best_overall_f1 = -1.0
    best_model_name = None
    best_model_metrics = {}
    best_model_state = None

    for model_name in models_to_compare:
        logger.info(f"==========================================")
        logger.info(f"Training and evaluating architecture: {model_name}")
        logger.info(f"==========================================")
        
        try:
            model = get_model(model_name).to(device)
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}. Skipping.")
            continue
            
        # Select all parameters that require training (including unfrozen conv blocks)
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_optimize, lr=0.0002)
            
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
        
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_state_local = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Validation epoch
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images).squeeze(1)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    
            val_loss /= len(val_loader.dataset)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            scheduler.step(val_loss)
            
            # Checkpoint and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state_local = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered for {model_name} after {epoch+1} epochs.")
                    break
                    
        # Load best local weights for evaluation on Test set
        if best_model_state_local is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state_local.items()})
            
        # Test evaluation
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
                
        # Calculate Evaluation Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            roc_auc = 0.5
            
        cm = confusion_matrix(all_labels, all_preds).tolist()
        report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
        
        logger.info(f"Results for {model_name}: Accuracy={accuracy:.4f}, F1-Score={f1:.4f}")
        
        # Track overall best model
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_model_name = model_name
            best_model_state = best_model_state_local
            best_model_metrics = {
                "model_name": model_name,
                "accuracy": round(float(accuracy), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
                "roc_auc": round(float(roc_auc), 4),
                "confusion_matrix": cm,
                "classification_report": report
            }

    # Save the overall best model and metadata
    if best_model_state is not None:
        if quick_test:
            logger.info("Quick test mode: skipping checkpoint saving to avoid overwriting production weights.")
        else:
            torch.save(best_model_state, MODEL_SAVE_PATH)
            logger.info(f"==========================================")
            logger.info(f"Best model selected: {best_model_name} with F1-Score: {best_overall_f1:.4f}")
            logger.info(f"Saved checkpoint to {MODEL_SAVE_PATH}")
            logger.info(f"==========================================")
            
            metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
            try:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump({"model_name": best_model_name}, f, indent=4)
                logger.info(f"Saved model metadata to {metadata_path}")
            except Exception as e:
                logger.error(f"Failed to save model metadata: {e}")
    else:
        raise RuntimeError("No model was successfully trained and evaluated.")

    # Save metrics to JSON file
    if quick_test:
        logger.info("Quick test mode: saving metrics to quick_test_metrics.json.")
        quick_test_metrics_path = os.path.join(BASE_DIR, "quick_test_metrics.json")
        try:
            with open(quick_test_metrics_path, "w", encoding="utf-8") as f:
                json.dump(best_model_metrics, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save quick test metrics: {e}")
    else:
        for path in [METRICS_SAVE_PATH, ALT_METRICS_SAVE_PATH]:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(best_model_metrics, f, indent=4)
                logger.info(f"Saved training metrics to {path}")
            except Exception as e:
                logger.error(f"Failed to save metrics to {path}: {e}")

    return best_model_metrics


if __name__ == "__main__":
    train_model(epochs=15, batch_size=16, patience=5)
