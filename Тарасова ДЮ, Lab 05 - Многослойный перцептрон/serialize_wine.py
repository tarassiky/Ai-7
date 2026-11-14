"""Сохранение Wine-модели и проверка воспроизводимости inference."""
from __future__ import annotations

import numpy as np

from src.bundle import InferenceBundle
from src.data_utils import WINE_FEATURE_NAMES


def serialize(result, model_path="wine_mlp_model.npz"):
    bundle = InferenceBundle(
        result["model"], result["standardizer"],
        WINE_FEATURE_NAMES, np.unique(result["y_tr"]),
    )
    saved_path = bundle.save(model_path)
    loaded = InferenceBundle.load(saved_path)

    # Deployment-проверка — на отдельном batch сырых признаков.
    raw = result["X_tr_raw"][:3].copy()
    p_trained, pred_trained = bundle.predict_raw(raw)
    p_loaded, pred_loaded = loaded.predict_raw(raw)

    assert saved_path.is_file()
    assert loaded.feature_names == WINE_FEATURE_NAMES
    assert np.array_equal(loaded.class_labels, np.array([0, 1, 2]))
    assert np.allclose(
        loaded.standardizer.standard_deviation,
        result["standardizer"].standard_deviation,
    )
    assert np.allclose(p_loaded, p_trained)
    assert np.array_equal(pred_loaded, pred_trained)

    print(f"saved to {saved_path}")
    print("probabilities:\n", p_loaded)
    print("predictions:", pred_loaded)
    return bundle, loaded


if __name__ == "__main__":
    from train_wine import run_training
    result = run_training()
    serialize(result)