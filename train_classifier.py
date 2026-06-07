import os
import sys
import json
import time
import random
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# Import newer weights definitions if available, otherwise fallback
try:
    from torchvision.models import ResNet18_Weights
    USE_NEW_WEIGHTS_API = True
except ImportError:
    USE_NEW_WEIGHTS_API = False

# ----------------------------------------------------
# 1. DATASET VALIDATION
# ----------------------------------------------------
def validate_dataset(dataset_dir):
    """
    Validates that the Kaggle dataset directory exists and is populated with
    genuine and tampered document images.
    """
    genuine_dir = os.path.join(dataset_dir, "genuine")
    tampered_dir = os.path.join(dataset_dir, "tampered")

    if not os.path.exists(genuine_dir) or not os.path.exists(tampered_dir):
        raise FileNotFoundError(
            f"Kaggle dataset directories not found. Please ensure both "
            f"'{genuine_dir}' and '{tampered_dir}' directories exist."
        )

    existing_gen = len([f for f in os.listdir(genuine_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    existing_tamp = len([f for f in os.listdir(tampered_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if existing_gen == 0 or existing_tamp == 0:
        raise FileNotFoundError(
            f"Kaggle dataset is empty. Found genuine={existing_gen}, tampered={existing_tamp} images. "
            f"Please place the Kaggle dataset files in '{dataset_dir}'."
        )

    print(f"[Dataset Validator] Verified Kaggle dataset: genuine={existing_gen}, tampered={existing_tamp} files.")

# ----------------------------------------------------
# 2. MAIN TRAINING PROCESS
# ----------------------------------------------------
def main():
    print("=" * 60)
    print("  DOCUSHIELD AI - RESNET18 KAGGLE DATASET CLASSIFIER TRAINING  ")
    print("=" * 60)

    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    models_dir = os.path.join(base_dir, "backend", "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Ensure Kaggle dataset exists and is valid
    validate_dataset(dataset_dir)

    # 2. Image Preprocessing & Transforms
    # Resize 224x224, Normalize with ImageNet stats
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load Dataset
    print("\n[DataLoader] Loading dataset from directory...")
    full_dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    print(f"[DataLoader] Classes: {full_dataset.classes}")
    print(f"[DataLoader] Class to idx mapping: {full_dataset.class_to_idx}")

    # Set indices mapping so 'genuine' is 0 and 'tampered' is 1
    # Check alignment
    genuine_idx = full_dataset.class_to_idx.get("genuine", 0)
    tampered_idx = full_dataset.class_to_idx.get("tampered", 1)
    print(f"[DataLoader] Genuine idx: {genuine_idx}, Tampered idx: {tampered_idx}")

    # 4. Train/Val Split (80% / 20%)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.2)
    train_size = dataset_size - val_size

    # Fix seed for reproducibility
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(f"[DataLoader] Split details: Train size = {train_size}, Val size = {val_size}")

    # 5. Initialize Model (ResNet18)
    print("\n[Model] Instantiating ResNet18 model architecture...")
    if USE_NEW_WEIGHTS_API:
        try:
            print("[Model] Loading pretrained weights using ResNet18_Weights.DEFAULT...")
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as e:
            print(f"[Model] Pretrained weight download failed/blocked ({e}). Training from scratch.")
            model = models.resnet18(weights=None)
    else:
        try:
            print("[Model] Loading pretrained weights using pretrained=True...")
            model = models.resnet18(pretrained=True)
        except Exception as e:
            print(f"[Model] Pretrained weight download failed/blocked ({e}). Training from scratch.")
            model = models.resnet18(pretrained=False)

    # Modify final layer for 2-class binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Move model to device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[Model] Model running on target device: {device}")

    # 6. Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7. Training Loop
    epochs = 3
    print(f"\n[Training] Starting training pipeline ({epochs} epochs)...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_train
        epoch_acc = (correct_train / total_train) * 100.0
        elapsed = time.time() - start_time

        print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Time: {elapsed:.2f}s")

    # 8. Evaluation on Validation Set
    print("\n[Evaluation] Running validation diagnostics...")
    model.eval()
    
    val_correct = 0
    val_total = 0

    # For metrics calculation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_accuracy = (val_correct / val_total) if val_total > 0 else 0.0
    print(f"[Evaluation] Validation Accuracy: {val_accuracy * 100.0:.2f}%")

    # Calculate metrics: Genuine is class 0, Tampered is class 1 (Positive class)
    # We want to identify 'tampered' (1) documents
    TP = 0  # True Positives: Actual tampered, predicted tampered
    FP = 0  # False Positives: Actual genuine, predicted tampered
    TN = 0  # True Negatives: Actual genuine, predicted genuine
    FN = 0  # False Negatives: Actual tampered, predicted genuine

    for pred, label in zip(all_preds, all_labels):
        if pred == tampered_idx and label == tampered_idx:
            TP += 1
        elif pred == tampered_idx and label == genuine_idx:
            FP += 1
        elif pred == genuine_idx and label == genuine_idx:
            TN += 1
        elif pred == genuine_idx and label == tampered_idx:
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"[Evaluation] Confusion Matrix: TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    print(f"[Evaluation] Accuracy : {val_accuracy:.4f}")
    print(f"[Evaluation] Precision: {precision:.4f}")
    print(f"[Evaluation] Recall   : {recall:.4f}")
    print(f"[Evaluation] F1 Score : {f1_score:.4f}")

    # 9. Save metrics.json
    metrics_data = {
        "accuracy": round(float(val_accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "confusion_matrix": {"tp": TP, "fp": FP, "tn": TN, "fn": FN}
    }

    # Save in root
    metrics_root_path = os.path.join(base_dir, "metrics.json")
    with open(metrics_root_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[Output] Exported metrics configuration to: {metrics_root_path}")

    # Save in backend
    metrics_backend_path = os.path.join(base_dir, "backend", "metrics.json")
    with open(metrics_backend_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[Output] Exported copy to backend directory: {metrics_backend_path}")

    # 10. Save model.pth
    model_path = os.path.join(models_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[Output] Model weights checkpoint saved successfully to: {model_path}")
    print("=" * 60)
    print("Training Pipeline Executed Successfully.\n")

if __name__ == "__main__":
    main()
