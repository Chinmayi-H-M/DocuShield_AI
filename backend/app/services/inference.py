import os
import io
from typing import Union, Dict, Any
from PIL import Image
import torch
import logging

from app.services.dataset_loader import val_test_transforms
from app.services.train_model import get_model

logger = logging.getLogger("docushield.ml.inference")


class InferenceService:
    """
    Handles loading the trained PyTorch model and running inference
    on provided document images (either from file paths or in-memory bytes).
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "models", "model.pth")
            
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """Loads and returns the model dynamically matching the metadata identifier."""
        # 1. Determine model architecture from metadata
        model_name = "resnet50"  # default fallback
        metadata_path = os.path.join(os.path.dirname(self.model_path), "model_metadata.json")
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    model_name = meta.get("model_name", "resnet50")
                logger.info(f"Detected model architecture from metadata: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model metadata from {metadata_path}: {e}")
                
        # 2. Initialize the model
        model = get_model(model_name)
        
        # 3. Load weights mapping to the correct target device
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"Successfully loaded trained model weights from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model weights from {self.model_path}: {e}")
                raise e
        else:
            raise FileNotFoundError(f"No trained model found at {self.model_path}")
            
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_input: Union[str, bytes]) -> Dict[str, Any]:
        """
        Executes prediction on the target image.
        Returns prediction label ("Tampered" or "Genuine"), classification confidence (%), and risk_level.
        """
        try:
            if isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input)).convert("RGB")
            else:
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image path not found: {image_input}")
                image = Image.open(image_input).convert("RGB")
            
            # Preprocess image and add batch dimension
            input_tensor = val_test_transforms(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_tensor).squeeze(1)
                prob = torch.sigmoid(logits).item()
                
            # Genuine = 0, Tampered = 1
            if prob >= 0.5:
                prediction = "Tampered"
                confidence = prob * 100.0
                
                # Risk level classification for Tampered
                if confidence > 80.0:
                    risk_level = "High"
                elif confidence > 50.0:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
            else:
                prediction = "Genuine"
                confidence = (1.0 - prob) * 100.0
                
                # Risk level classification for Genuine
                if confidence > 80.0:
                    risk_level = "Low"
                elif confidence > 50.0:
                    risk_level = "Medium"
                else:
                    risk_level = "High"
                
            return {
                "prediction": prediction,
                "confidence": round(confidence, 1),
                "risk_level": risk_level,
                "raw_score": round(prob * 100.0, 1)  # Raw score for combined score calculation
            }
        except Exception as e:
            logger.error(f"Prediction failed in InferenceService: {e}")
            raise e
