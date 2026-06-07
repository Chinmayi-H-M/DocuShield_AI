import os
import glob
import random
import hashlib
import io
import numpy as np
from PIL import Image, ImageFilter
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger("docushield.ml.dataset")

class AddNoiseAndCompression(object):
    """
    Custom PyTorch transform to add Gaussian noise and simulate JPEG compression artifacts.
    Also randomly applies Gaussian blur to increase model robustness.
    """
    def __init__(self, noise_prob: float = 0.3, compress_prob: float = 0.3, blur_prob: float = 0.3):
        self.noise_prob = noise_prob
        self.compress_prob = compress_prob
        self.blur_prob = blur_prob

    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert PIL to numpy for noise injection
        if random.random() < self.noise_prob:
            img_arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, random.uniform(2, 10), img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr)
            
        # Simulate lossy JPEG compression artifacts
        if random.random() < self.compress_prob:
            buffer = io.BytesIO()
            quality = random.randint(30, 75)
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert("RGB")
            
        # Apply slight blur to simulate scan out-of-focus issues
        if random.random() < self.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            
        return img


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
# ImageNet statistics are standard for pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    AddNoiseAndCompression(noise_prob=0.3, compress_prob=0.3, blur_prob=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def get_document_type(file_path: str) -> str:
    """
    Maps a file path to its corresponding document type:
    Aadhaar, PAN, Passport, Invoice, Bank Statement, Property Document.
    """
    fn_lower = os.path.basename(file_path).lower()
    if "aadhaar" in fn_lower:
        return "Aadhaar"
    if "pan" in fn_lower:
        return "PAN"
    if "passport" in fn_lower:
        return "Passport"
    if "invoice" in fn_lower:
        return "Invoice"
    if "bank" in fn_lower or "statement" in fn_lower:
        return "Bank Statement"
    if "property" in fn_lower or "deed" in fn_lower:
        return "Property Document"
        
    # Fallback to a deterministic document type mapping using hash
    h = int(hashlib.md5(fn_lower.encode("utf-8")).hexdigest(), 16)
    doc_types = ["Aadhaar", "PAN", "Passport", "Invoice", "Bank Statement", "Property Document"]
    return doc_types[h % len(doc_types)]


def load_dataset_splits(
    dataset_dir: str, 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Scans the dataset directories, matches image files, maps labels,
    and performs a stratified split based on document type and fraud label.
    """
    genuine_dir = os.path.join(dataset_dir, "genuine")
    tampered_dir = os.path.join(dataset_dir, "tampered")
    
    # Verify/create paths to avoid directory not found errors
    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(tampered_dir, exist_ok=True)
    
    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    
    genuine_files: List[str] = []
    seen_genuine = set()
    for ext in supported_extensions:
        for f in glob.glob(os.path.join(genuine_dir, ext)):
            abs_p = os.path.abspath(f)
            norm_p = os.path.normcase(abs_p)
            if norm_p not in seen_genuine:
                seen_genuine.add(norm_p)
                genuine_files.append(abs_p)
        
    tampered_files: List[str] = []
    seen_tampered = set()
    for ext in supported_extensions:
        for f in glob.glob(os.path.join(tampered_dir, ext)):
            abs_p = os.path.abspath(f)
            norm_p = os.path.normcase(abs_p)
            if norm_p not in seen_tampered:
                seen_tampered.add(norm_p)
                tampered_files.append(abs_p)
        
    file_paths = genuine_files + tampered_files
    labels = [0] * len(genuine_files) + [1] * len(tampered_files)
    
    if not file_paths:
        logger.warning(f"No documents found in dataset path: {dataset_dir}")
        return [], [], [], [], [], []
        
    total_test_val = val_ratio + test_ratio
    
    try:
        # Create stratification keys combining class label and document type
        stratify_keys = [f"{lbl}_{get_document_type(fp)}" for fp, lbl in zip(file_paths, labels)]
        
        # Package file paths, labels, and keys into a list of tuples to keep alignment
        items = list(zip(file_paths, labels, stratify_keys))
        
        # Stratified splits to preserve class and document type distributions
        train_items, temp_items = train_test_split(
            items, test_size=total_test_val, random_state=42, stratify=[x[2] for x in items]
        )
        
        # Split remaining 30% equally into Val (15%) and Test (15%)
        val_items, test_items = train_test_split(
            temp_items, test_size=0.5, random_state=42, stratify=[x[2] for x in temp_items]
        )
        
        train_paths = [x[0] for x in train_items]
        train_labels = [x[1] for x in train_items]
        
        val_paths = [x[0] for x in val_items]
        val_labels = [x[1] for x in val_items]
        
        test_paths = [x[0] for x in test_items]
        test_labels = [x[1] for x in test_items]
        
    except ValueError:
        logger.warning("Dataset size or distribution too small for stratified splits on all document types. Falling back to default label split.")
        try:
            train_paths, temp_paths, train_labels, temp_labels = train_test_split(
                file_paths, labels, test_size=total_test_val, random_state=42, stratify=labels
            )
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
            )
        except ValueError:
            logger.warning("Falling back to unstratified split.")
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
