"""Обучение Wine: архитектура 13-16-8-3, mini-batch SGD."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.autograd import MatrixValue
from src.data_utils import (
    WINE_FEATURE_NAMES, Standardizer, load_wine_csv, stratified_split,
)
from src.mlp import (
    MLPClassifier, BestValidationCheckpoint, batch_count, count_parameters,
)


def run_training(seed=42, model_seed=7, epochs=100, batch_size=16,
                 learning_rate=0.03, shuffle_seed=2026,
                 data_path="data/wine.csv",
                 checkpoint_path="wine_best_checkpoint.npz"):
    features_all, labels_all = load_wine_csv(data_path)
    (X_tr_raw, y_tr, X_va_raw, y_va, X_te_raw, y_te) = stratified_split(
        features_all, labels_all, seed=seed,
    )
    std = Standardizer()
    X_tr = std.fit_transform(X_tr_raw)
    X_va = std.transform(X_va_raw)
    X_te = std.transform(X_te_raw)

    assert X_tr.shape == (125, 13)
    assert X_va.shape == (27, 13)
    assert X_te.shape == (26, 13)

    model = MLPClassifier(
        layer_sizes=[13, 16, 8, 3],
        hidden_activations=["relu", "tanh"],
        seed=model_seed,
    )
    assert count_parameters(model) == 387

    checkpoint = BestValidationCheckpoint(path=checkpoint_path)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        shuffle_seed=shuffle_seed,
        checkpoint=checkpoint,
    )

    assert len(history.history["loss"]) == epochs
    assert np.isfinite(history.history["loss"]).all()
    assert history.history["loss"][-1] < history.history["loss"][0]
    assert 0 <= history.best_epoch < epochs
    assert math.isclose(
        history.best_validation_loss,
        min(history.history["val_loss"]),
    )

    checkpoint.restore(model, source="memory")

    # Восстановление из файла должно дать те же веса.
    file_cp = BestValidationCheckpoint(path=checkpoint_path)
    file_cp.restore(model, source="file")
    assert file_cp.best_epoch == checkpoint.best_epoch

    return {
        "model": model,
        "standardizer": std,
        "history": history,
        "checkpoint": checkpoint,
        "X_tr": X_tr, "y_tr": y_tr,
        "X_va": X_va, "y_va": y_va,
        "X_te": X_te, "y_te": y_te,
        "X_tr_raw": X_tr_raw,
        "batch_updates": batch_count(125, batch_size),
    }


if __name__ == "__main__":
    result = run_training()
    h = result["history"]
    print(f"first_train_loss = {h.history['loss'][0]:.6f}")
    print(f"last_train_loss  = {h.history['loss'][-1]:.6f}")
    print(f"best_epoch       = {h.best_epoch + 1}")
    print(f"best_val_loss    = {h.best_validation_loss:.6f}")
    print(f"updates per epoch = {result['batch_updates']}")