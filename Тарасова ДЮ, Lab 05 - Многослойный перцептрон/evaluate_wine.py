"""Графики обучения и финальная оценка Wine."""
from __future__ import annotations

import numpy as np

from src.mlp import plot_training_history


def confusion_matrix(labels, predictions, class_count):
    matrix = np.zeros((class_count, class_count), dtype=int)
    for target, prediction in zip(labels, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def evaluate_all(result, save_figure="wine_training_history.png"):
    model = result["model"]
    history = result["history"]

    # Кривые обучения (без test).
    fig, axes = plot_training_history(
        history, "Wine: динамика train и validation по эпохам",
    )
    fig.savefig(save_figure, dpi=120)

    train_metrics = model.evaluate(result["X_tr"], result["y_tr"])
    val_metrics = model.evaluate(result["X_va"], result["y_va"])
    test_metrics = model.evaluate(result["X_te"], result["y_te"])
    test_predictions = model.predict(result["X_te"])
    test_confusion = confusion_matrix(result["y_te"], test_predictions, 3)

    print(f"train:      loss={train_metrics['loss']:.6f}, "
          f"acc={train_metrics['accuracy']:.6f}")
    print(f"validation: loss={val_metrics['loss']:.6f}, "
          f"acc={val_metrics['accuracy']:.6f}")
    print(f"test:       loss={test_metrics['loss']:.6f}, "
          f"acc={test_metrics['accuracy']:.6f}")
    print("test confusion matrix:\n", test_confusion)

    # Критерии Business Understanding.
    assert train_metrics["loss"] < 0.03
    assert train_metrics["accuracy"] == 1.0
    assert val_metrics["accuracy"] >= 0.95
    assert test_metrics["accuracy"] >= 0.95
    assert test_metrics["loss"] < 0.2
    assert test_confusion.shape == (3, 3)
    assert test_confusion.sum() == 26

    return {
        "train": train_metrics, "validation": val_metrics,
        "test": test_metrics, "confusion": test_confusion,
        "figure": save_figure,
    }


if __name__ == "__main__":
    from train_wine import run_training
    result = run_training()
    evaluate_all(result)