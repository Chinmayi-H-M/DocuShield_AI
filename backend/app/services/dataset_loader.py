import os
import glob
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger("docushield.ml.dataset")

class FraudDetectionDataset(Dataset):
    """
    Custom PyTorch Dataset for loading document images and their fraud labels.
    Genuine = 0, Tampered = 1
    """
    def __init__(self, file_paths: List[str], labels: List[int], transform: Optional[transforms.Compose] = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            # Open image and convert to RGB format
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image at {img_path}: {e}")
            # Fallback to a blank image if loading fails
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)


# Data Augmentation & Normalization Transforms
# ImageNet statistics are standard for pretrained ResNet50
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def load_dataset_splits(
    dataset_dir: str, 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Scans the dataset directories, matches image files, maps labels,
    and performs a stratified split into train, validation, and test sets.
    """
    genuine_dir = os.path.join(dataset_dir, "genuine")
    tampered_dir = os.path.join(dataset_dir, "tampered")
    
    # Verify/create paths to avoid directory not found errors
    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(tampered_dir, exist_ok=True)
    
    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    
    genuine_files: List[str] = []
    for ext in supported_extensions:
        genuine_files.extend(glob.glob(os.path.join(genuine_dir, ext)))
        
    tampered_files: List[str] = []
    for ext in supported_extensions:
        tampered_files.extend(glob.glob(os.path.join(tampered_dir, ext)))
        
    file_paths = genuine_files + tampered_files
    labels = [0] * len(genuine_files) + [1] * len(tampered_files)
    
    if not file_paths:
        logger.warning(f"No documents found in dataset path: {dataset_dir}")
        return [], [], [], [], [], []
        
    total_test_val = val_ratio + test_ratio
    
    try:
        # Stratified splits to preserve class distributions
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            file_paths, labels, test_size=total_test_val, random_state=42, stratify=labels
        )
        # Split remaining 30% equally into Val (15%) and Test (15%)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
    except ValueError:
        logger.warning("Dataset too small or imbalanced for stratified split. Falling back to default split.")
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            file_paths, labels, test_size=total_test_val, random_state=42
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=42
        )
        
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def get_data_loaders(
    dataset_dir: str, 
    batch_size: int = 16, 
    num_workers: int = 0
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Constructs PyTorch DataLoaders for the Train, Validation, and Test splits.
    """
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = load_dataset_splits(dataset_dir)
    
    if not train_paths:
        logger.error(f"Cannot create dataloaders because dataset is empty: {dataset_dir}")
        return None, None, None
        
    train_dataset = FraudDetectionDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = FraudDetectionDataset(val_paths, val_labels, transform=val_test_transforms)
    test_dataset = FraudDetectionDataset(test_paths, test_labels, transform=val_test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    logger.info(
        f"DataLoaders created. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}"
    )
    
    return train_loader, val_loader, test_loader
