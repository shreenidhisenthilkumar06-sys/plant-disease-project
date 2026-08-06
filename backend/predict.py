"""Model loading and inference service."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from disease_info import get_disease_info
from preprocess import preprocess_image
from schemas import PredictionResponse


class ModelUnavailableError(RuntimeError):
    """Raised when the saved model or class metadata cannot be loaded."""


class ModelService:
    """Owns the single in-memory TensorFlow model instance."""

    def __init__(self, model_path: Path, class_names_path: Path) -> None:
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model: tf.keras.Model | None = None
        self.class_names: list[str] = []

    def load(self) -> None:
        if not self.model_path.exists():
            raise ModelUnavailableError(f"Saved model was not found at {self.model_path}.")
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.class_names = json.loads(self.class_names_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ModelUnavailableError("Could not load the model or class_names.json.") from error
        if not isinstance(self.class_names, list) or not all(isinstance(item, str) for item in self.class_names):
            raise ModelUnavailableError("class_names.json must contain an ordered array of class names.")
        output_classes = int(self.model.output_shape[-1])
        if len(self.class_names) != output_classes:
            raise ModelUnavailableError(
                f"Model has {output_classes} output classes but class_names.json contains {len(self.class_names)} entries."
            )

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        if self.model is None:
            raise ModelUnavailableError("Model has not been loaded.")
        scores = np.asarray(self.model.predict(preprocess_image(image_bytes), verbose=0))[0]
        # Support models that emit logits as well as models that already emit probabilities.
        probabilities = self._to_probabilities(scores)
        index = int(np.argmax(probabilities))
        class_name = self.class_names[index]
        info = get_disease_info(class_name)
        return PredictionResponse(
            prediction=info.display_name,
            confidence=round(float(probabilities[index] * 100), 2),
            description=info.description,
            symptoms=info.symptoms,
            causes=info.causes,
            prevention=info.prevention,
            treatment=info.treatment,
        )

    @staticmethod
    def _to_probabilities(scores: np.ndarray) -> np.ndarray:
        if np.all(scores >= 0) and np.isclose(float(scores.sum()), 1.0, atol=1e-3):
            return scores
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum()
