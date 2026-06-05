import os
import io
from typing import Union, Dict, Any
from PIL import Image
import torch
import logging

from app.services.dataset_loader import val_test_transforms
from app.services.train_model import get_resnet50_model

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
        """Loads and returns the ResNet50 model, restoring weights if available."""
        model = get_resnet50_model()
        if os.path.exists(self.model_path):
            try:
                # Load weights mapping to the correct target device
                state_dict = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"Successfully loaded trained model weights from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model weights from {self.model_path}: {e}. Fallback to initialized model.")
        else:
            logger.warning(f"No trained model found at {self.model_path}. Using initialized model state.")
            
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_input: Union[str, bytes]) -> Dict[str, Any]:
        """
        Executes prediction on the target image.
        Returns prediction label ("Tampered" or "Genuine") and classification confidence (%).
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
            else:
                prediction = "Genuine"
                confidence = (1.0 - prob) * 100.0
                
            return {
                "prediction": prediction,
                "confidence": round(confidence, 1)
            }
        except Exception as e:
            logger.error(f"Prediction failed in InferenceService: {e}")
            raise e
