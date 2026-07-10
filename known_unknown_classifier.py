from __future__ import annotations

import os
from typing import Dict

import timm
import torch
import torch.nn.functional as F
import numpy as np

from common import preprocess_cv2, load_model_from_models


class KnownUnknownClassifier:
    """
    Binary classifier that predicts whether a detected person is known or unknown.

    The architecture must match training:
    - timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
    """

    def __init__(
        self,
        model_name: str = None,
        device: torch.device = None,
        known_class_index: int = 0,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.known_class_index = known_class_index

        model_arch = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=2,
        ).to(self.device)

        self.model = self._load_classifier_model(
            model_name=model_name,
            model_arch=model_arch,
        )
        self.model.eval()
        self.model.to(self.device)

    def _load_classifier_model(self, model_name: str, model_arch: torch.nn.Module) -> torch.nn.Module:
        if model_name and os.path.isabs(model_name):
            if not os.path.exists(model_name):
                raise FileNotFoundError(f"Classifier model not found: {model_name}")

            checkpoint = torch.load(model_name, map_location=self.device)
            if isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            else:
                state_dict = checkpoint
                if isinstance(checkpoint, dict):
                    for key in ("model", "net", "module", "state_dict"):
                        if key in checkpoint:
                            candidate = checkpoint[key]
                            if isinstance(candidate, torch.nn.Module):
                                model = candidate
                                break
                            if isinstance(candidate, dict):
                                state_dict = candidate
                                break
                    else:
                        state_dict = checkpoint

                if "model" not in locals():
                    processed_state_dict = {}
                    for key, value in state_dict.items():
                        name = key[7:] if key.startswith("module.") else key
                        processed_state_dict[name] = value

                    model_arch.load_state_dict(processed_state_dict, strict=False)
                    model = model_arch

            model.eval()
            model.to(self.device)
            return model

        return load_model_from_models(
            filename=model_name,
            model_arch=model_arch,
            device=self.device,
        )

    def predict(self, image_bgr: np.ndarray) -> Dict:
        if not isinstance(image_bgr, np.ndarray):
            raise TypeError("image_bgr must be a numpy.ndarray")

        tensor = preprocess_cv2(image_bgr, device=self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            probs = F.softmax(logits, dim=1).squeeze(0)

        predicted_index = int(torch.argmax(probs).item())
        confidence = float(probs[predicted_index].item())
        is_known = predicted_index == self.known_class_index
        predicted_class = "known" if is_known else "unknown"

        return {
            "is_known": is_known,
            "predicted_class": predicted_class,
            "display_label": "quen" if is_known else "lạ",
            "confidence": confidence,
            "probabilities": {
                "known": float(probs[self.known_class_index].item()),
                "unknown": float(probs[1 - self.known_class_index].item()),
            },
        }
