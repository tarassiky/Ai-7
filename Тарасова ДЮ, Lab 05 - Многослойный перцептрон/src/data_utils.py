"""Загрузка Wine, стратифицированный split, Standardizer."""
from __future__ import annotations

from pathlib import Path

import numpy as np

WINE_FEATURE_NAMES = (
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue", "od280_od315", "proline",
)


def load_wine_csv(path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(Path(path), delimiter=",", skiprows=1)
    labels = table[:, 0].astype(int)
    features = table[:, 1:].astype(float)

    if features.shape != (178, 13):
        raise ValueError(f"Ожидалась форма (178, 13), получена {features.shape}")
    if labels.shape != (178,):
        raise ValueError(f"Ожидалась форма (178,), получена {labels.shape}")
    if not np.isfinite(features).all():
        raise ValueError("В признаках обнаружены пропуски или бесконечности")
    if not np.array_equal(np.unique(labels), np.array([0, 1, 2])):
        raise ValueError("Метки должны быть 0, 1, 2")
    return features, labels


def stratified_split(features, labels, seed=42):
    values = np.asarray(features, dtype=float)
    targets = np.asarray(labels)

    if values.ndim != 2 or targets.ndim != 1 or len(values) != len(targets):
        raise ValueError("features и labels описывают разное число строк")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("labels должны быть целочисленными")

    classes, class_sizes = np.unique(targets, return_counts=True)
    if not len(targets):
        raise ValueError("Нельзя разделить пустой набор")

    def proportional_counts(part_size):
        ideal = class_sizes * part_size / len(targets)
        counts = np.floor(ideal).astype(int)
        remaining = int(part_size - counts.sum())
        order = np.argsort(-(ideal - counts), kind="stable")
        counts[order[:remaining]] += 1
        return counts

    validation_counts = proportional_counts(int(round(0.15 * len(targets))))
    test_counts = proportional_counts(int(np.floor(0.15 * len(targets))))
    train_counts = class_sizes - validation_counts - test_counts

    if min(validation_counts.min(), test_counts.min(), train_counts.min()) < 1:
        raise ValueError("Недостаточно объектов каждого класса")

    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label, test_count, val_count in zip(classes, test_counts, validation_counts):
        idx = np.flatnonzero(targets == label)
        rng.shuffle(idx)
        test_idx.extend(idx[:test_count])
        val_idx.extend(idx[test_count: test_count + val_count])
        train_idx.extend(idx[test_count + val_count:])

    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    combined = np.concatenate([train_idx, val_idx, test_idx])
    assert np.array_equal(np.sort(combined), np.arange(len(targets)))

    return (
        values[train_idx], targets[train_idx],
        values[val_idx], targets[val_idx],
        values[test_idx], targets[test_idx],
    )


class Standardizer:
    """Train-статистики для стандартизации (без утечки)."""

    def __init__(self):
        self.mean = None
        self.standard_deviation = None

    def fit(self, train):
        values = np.asarray(train, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("train должен быть конечной двумерной матрицей")
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=0)
        if np.any(std == 0.0):
            raise ValueError("Нельзя стандартизовать признак с нулевым std")
        self.mean = mean
        self.standard_deviation = std
        return self

    def transform(self, features):
        if self.mean is None or self.standard_deviation is None:
            raise RuntimeError("Сначала вызовите Standardizer.fit(train)")
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError("Число признаков не совпало с train")
        if not np.isfinite(values).all():
            raise ValueError("Признаки должны быть конечными")
        return (values - self.mean) / self.standard_deviation

    def fit_transform(self, train):
        return self.fit(train).transform(train)