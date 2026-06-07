import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from app.services.train_model import get_model as init_model

_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing transforms matching the training setup exactly
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_model():
    """
    Loads and caches the trained model weights from models/model.pth dynamically matching metadata.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    # Resolve absolute path to model.pth
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "models", "model.pth"))
    metadata_path = os.path.abspath(os.path.join(current_dir, "..", "..", "models", "model_metadata.json"))
    
    # 1. Determine model architecture from metadata
    model_name = "resnet50"
    if os.path.exists(metadata_path):
        try:
            import json
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                model_name = meta.get("model_name", "resnet50")
        except Exception as e:
            print(f"[Model Inference] Failed to load metadata from {metadata_path}: {e}")
            
    # 2. Initialize the model matching the dynamic training settings
    model = init_model(model_name)
    
    # 3. Load weights mapping to the correct target device
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=_DEVICE)
            model.load_state_dict(state_dict)
            print(f"[Model Inference] Loaded model checkpoint from: {model_path}")
        except Exception as e:
            print(f"[Model Inference] Error loading state dict from {model_path}: {e}")
            raise e
    else:
        print(f"[Model Inference] Warning: Trained weights not found at '{model_path}'.")
        
    model.eval()
    model = model.to(_DEVICE)
    _MODEL = model
    return _MODEL

def predict_document(image_path: str) -> dict:
    """
    Predicts whether a document image is genuine or tampered.
    
    Parameters:
        image_path (str): Filepath to the document scan.
        
    Returns:
        dict: {"prediction": "genuine|tampered", "confidence": float}
    """
    try:
        if not os.path.exists(image_path):
            return {"prediction": "genuine", "confidence": 0.5, "error": f"File '{image_path}' not found"}

        # Attempt to load document scan as an image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            # Fallback for PDFs or other non-raster formats
            return {
                "prediction": "genuine", 
                "confidence": 0.5,
                "note": "Document format not supported directly by image classifier"
            }

        # Load cached model
        model = get_model()

        # Apply transforms and add batch dimension
        input_tensor = _TRANSFORM(img).unsqueeze(0).to(_DEVICE)

        # Run forward pass (sigmoid for BCE classifier output)
        with torch.no_grad():
            logits = model(input_tensor).squeeze(1)
            prob = torch.sigmoid(logits).item()

        # Genuine = 0, Tampered = 1
        if prob >= 0.5:
            prediction = "tampered"
            confidence = prob
        else:
            prediction = "genuine"
            confidence = 1.0 - prob

        return {
            "prediction": prediction,
            "confidence": round(float(confidence), 4)
        }
    except Exception as e:
        print(f"[Model Inference] Error during inference on '{image_path}': {e}")
        return {
            "prediction": "genuine",
            "confidence": 0.5,
            "error": str(e)
        }
