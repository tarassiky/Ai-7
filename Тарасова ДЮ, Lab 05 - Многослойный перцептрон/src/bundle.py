"""InferenceBundle: модель + стандартизатор + имена + метки."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .mlp import MLPClassifier


class InferenceBundle:
    def __init__(self, model, standardizer, feature_names, class_labels):
        if standardizer.mean is None or standardizer.standard_deviation is None:
            raise ValueError("Standardizer должен быть обучен на train")
        self.model = model
        self.standardizer = standardizer
        self.feature_names = tuple(feature_names)
        self.class_labels = np.asarray(class_labels).copy()
        if len(self.feature_names) != model.layer_sizes[0]:
            raise ValueError("Число имён признаков не равно входной ширине")
        if self.class_labels.shape != (model.layer_sizes[-1],):
            raise ValueError("Число меток классов не равно выходной ширине")

    def predict_raw(self, raw_features):
        standardized = self.standardizer.transform(raw_features)
        probabilities = self.model.predict_proba(standardized)
        predictions = self.class_labels[np.argmax(probabilities, axis=1)]
        return probabilities, predictions

    def save(self, path):
        destination = Path(path)
        payload = {
            "format_version": np.array([1], dtype=int),
            "layer_sizes": np.array(self.model.layer_sizes, dtype=int),
            "hidden_activations": np.array(
                self.model.hidden_activations, dtype="U16"
            ),
            "feature_names": np.array(self.feature_names, dtype="U32"),
            "class_labels": self.class_labels,
            "feature_mean": self.standardizer.mean,
            "feature_standard_deviation": self.standardizer.standard_deviation,
        }
        for i, p in enumerate(self.model.parameters()):
            payload[f"weight_{i}"] = p.data
        np.savez(destination, **payload)
        return destination

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as stored:
            version = int(stored["format_version"][0])
            if version != 1:
                raise ValueError(f"Неподдерживаемая версия: {version}")
            layer_sizes = stored["layer_sizes"].astype(int).tolist()
            activations = stored["hidden_activations"].astype(str).tolist()
            model = MLPClassifier(layer_sizes, activations, seed=0)
            for i, p in enumerate(model.parameters()):
                w = np.asarray(stored[f"weight_{i}"], dtype=float)
                if w.shape != p.data.shape:
                    raise ValueError("Форма сохранённой матрицы не совпадает")
                p.data = w.copy()

            from .data_utils import Standardizer
            std = Standardizer()
            std.mean = np.asarray(stored["feature_mean"], dtype=float).copy()
            std.standard_deviation = np.asarray(
                stored["feature_standard_deviation"], dtype=float
            ).copy()
            if np.any(std.standard_deviation <= 0.0):
                raise ValueError("Сохранённые std должны быть положительными")

            feature_names = tuple(stored["feature_names"].astype(str).tolist())
            class_labels = np.asarray(stored["class_labels"]).copy()
            return cls(model, std, feature_names, class_labels)