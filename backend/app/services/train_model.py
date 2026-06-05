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


def get_resnet50_model() -> nn.Module:
    """
    Initializes ResNet50 model with pretrained weights and modifies
    the classification head for binary fraud detection.
    """
    try:
        # Modern torchvision version
        from torchvision.models import resnet50, ResNet50_Weights
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        logger.info("Loaded ResNet50 with default ImageNet weights.")
    except ImportError:
        # Fallback for older torchvision versions
        resnet = models.resnet50(pretrained=True)
        logger.info("Loaded ResNet50 with pretrained=True fallback.")

    # Freeze base model parameters for transfer learning
    for param in resnet.parameters():
        param.requires_grad = False

    # Replace the classification head
    # ResNet50 final layer is fc with 2048 input features
    num_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)  # Logits output for BCEWithLogitsLoss
    )
    
    # Ensure classification head parameters have gradients enabled
    for param in resnet.fc.parameters():
        param.requires_grad = True

    return resnet


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
    for ext in supported_extensions:
        genuine_files.extend(glob.glob(os.path.join(genuine_dir, ext)))
        
    if len(genuine_files) == 0:
        logger.info("Genuine dataset folder is empty. Generating 15 fallback genuine images...")
        for i in range(15):
            img = Image.new("RGB", (224, 224), color=(240, 240, 240))
            d = ImageDraw.Draw(img)
            d.text((10, 10), f"Genuine Invoice #{1000+i}", fill=(0, 0, 0))
            d.text((10, 50), "Amount Due: $150.00", fill=(0, 0, 0))
            img.save(os.path.join(genuine_dir, f"genuine_{i}.png"))
        # Refresh the list
        for ext in supported_extensions:
            genuine_files.extend(glob.glob(os.path.join(genuine_dir, ext)))

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



def train_model(epochs: int = 10, batch_size: int = 4, patience: int = 3) -> Dict[str, Any]:
    """
    Trains the model with early stopping, saves weights to model.pth,
    and returns test evaluation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for training: {device}")

    # Generate synthetic data if none exists
    generate_tiny_synthetic_dataset(DATASET_DIR)

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(DATASET_DIR, batch_size=batch_size)
    if train_loader is None or val_loader is None or test_loader is None:
        raise ValueError("Data loaders could not be initialized. Please check the dataset directories.")

    # Initialize model, loss, and optimizer
    model = get_resnet50_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    logger.info("Starting training loop...")
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
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            # Save checkpoint
            torch.save(best_model_state, MODEL_SAVE_PATH)
            logger.info("Saved new best model checkpoint.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best checkpoint for final evaluation on Test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        logger.warning("No checkpoint was saved. Evaluating final model state.")

    # Test evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy().astype(int))

    # Calculate Evaluation Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist() # Convert matrix to nested list for json

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm
    }

    # Save metrics to JSON file
    for path in [METRICS_SAVE_PATH, ALT_METRICS_SAVE_PATH]:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Saved training metrics to {path}")
        except Exception as e:
            logger.error(f"Failed to save metrics to {path}: {e}")

    return metrics


if __name__ == "__main__":
    train_model(epochs=10, batch_size=4, patience=3)
