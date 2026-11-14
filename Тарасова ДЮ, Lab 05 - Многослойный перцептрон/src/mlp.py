"""MLPClassifier, TrainingHistory, BestValidationCheckpoint, plot_training_history."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .autograd import MatrixValue, softmax_cross_entropy, softmax_rows


class TrainingHistory:
    def __init__(self):
        self.history = {
            "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [],
        }
        self.best_epoch = None
        self.best_validation_loss = math.inf

    def append(self, train_metrics, validation_metrics):
        self.history["loss"].append(float(train_metrics["loss"]))
        self.history["accuracy"].append(float(train_metrics["accuracy"]))
        self.history["val_loss"].append(float(validation_metrics["loss"]))
        self.history["val_accuracy"].append(
            float(validation_metrics["accuracy"])
        )


def plot_training_history(history, title):
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title должен быть непустой строкой")
    keys = ("loss", "val_loss", "accuracy", "val_accuracy")
    sequences = {k: np.array(history.history[k], dtype=float) for k in keys}
    n = len(sequences["loss"])
    if n == 0:
        raise ValueError("История пуста")
    if any(len(v) != n for v in sequences.values()):
        raise ValueError("Серии разной длины")
    if not all(np.isfinite(v).all() for v in sequences.values()):
        raise ValueError("Метрики должны быть конечными")
    if not (isinstance(history.best_epoch, (int, np.integer))
            and 0 <= history.best_epoch < n):
        raise ValueError("best_epoch вне диапазона")

    epochs = np.arange(1, n + 1)
    best_ep = int(history.best_epoch) + 1
    fig, (ax_loss, ax_acc) = plt.subplots(
        1, 2, figsize=(12, 4.5), constrained_layout=True,
    )
    ax_loss.plot(epochs, sequences["loss"], label="train loss", color="#1565C0")
    ax_loss.plot(epochs, sequences["val_loss"], label="validation loss", color="#EF6C00")
    ax_acc.plot(epochs, sequences["accuracy"], label="train accuracy", color="#1565C0")
    ax_acc.plot(epochs, sequences["val_accuracy"], label="validation accuracy", color="#EF6C00")
    for ax, ylabel in ((ax_loss, "Loss (CCE)"), (ax_acc, "Accuracy")):
        ax.axvline(best_ep, color="#A61C00", linestyle="--",
                   linewidth=1.5, label="лучшая validation-эпоха")
        ax.set_xlabel("Номер эпохи")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax_loss.set_title("Подгонка и validation loss")
    ax_acc.set_title("Подгонка и validation accuracy")
    fig.suptitle(title)
    return fig, (ax_loss, ax_acc)


class BestValidationCheckpoint:
    def __init__(self, path=None):
        self.path = None if path is None else Path(path)
        self.best_epoch = None
        self.best_validation_loss = math.inf
        self._best_weights = None

    def update(self, model, epoch, validation_metrics):
        loss = float(validation_metrics["loss"])
        if not math.isfinite(loss):
            raise FloatingPointError("Validation loss должна быть конечной")
        if loss >= self.best_validation_loss:
            return False
        self.best_epoch = int(epoch)
        self.best_validation_loss = loss
        self._best_weights = [p.data.copy() for p in model.parameters()]
        if self.path is not None:
            payload = {
                "best_epoch": np.array([self.best_epoch], dtype=int),
                "best_validation_loss": np.array([loss], dtype=float),
            }
            for i, w in enumerate(self._best_weights):
                payload[f"weight_{i}"] = w
            np.savez(self.path, **payload)
        return True

    def restore(self, model, source="memory"):
        if source == "memory":
            weights = self._best_weights
        elif source == "file":
            if self.path is None or not self.path.is_file():
                raise RuntimeError("Файловый checkpoint ещё не создан")
            with np.load(self.path, allow_pickle=False) as f:
                self.best_epoch = int(f["best_epoch"][0])
                self.best_validation_loss = float(
                    f["best_validation_loss"][0]
                )
                weights = [
                    np.asarray(f[f"weight_{i}"], dtype=float).copy()
                    for i in range(len(model.parameters()))
                ]
        else:
            raise ValueError("source должен быть 'memory' или 'file'")
        if weights is None:
            raise RuntimeError("Validation-checkpoint ещё не создан")
        for p, w in zip(model.parameters(), weights):
            if p.data.shape != w.shape:
                raise ValueError("Форма checkpoint не совпала с моделью")
            p.data = w.copy()
        return model


def iterate_minibatches(features, labels, batch_size, generator, shuffle=True):
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("Несогласованные формы features и labels")
    if batch_size <= 0:
        raise ValueError("batch_size должен быть положительным")
    indices = np.arange(features.shape[0])
    if shuffle:
        generator.shuffle(indices)
    for start in range(0, features.shape[0], batch_size):
        chunk = indices[start: start + batch_size]
        yield features[chunk], labels[chunk]


def batch_count(sample_count, batch_size):
    return math.ceil(sample_count / batch_size)


class MLPClassifier:
    def __init__(self, layer_sizes, hidden_activations, seed):
        sizes = tuple(int(s) for s in layer_sizes)
        activations = tuple(hidden_activations)
        if len(sizes) < 2 or any(s <= 0 for s in sizes):
            raise ValueError("layer_sizes должен содержать хотя бы d и C")
        if sizes[-1] < 2:
            raise ValueError("Нужно не менее двух классов")
        if len(activations) != len(sizes) - 2:
            raise ValueError("Нужна одна активация на каждый скрытый слой")
        if any(a not in {"relu", "tanh"} for a in activations):
            raise ValueError("Поддерживаются relu и tanh")

        self.layer_sizes = sizes
        self.hidden_activations = activations
        self.weights = []
        rng = np.random.default_rng(seed)

        for layer_index, (fan_in, fan_out) in enumerate(
            zip(sizes[:-1], sizes[1:])
        ):
            is_output = layer_index == len(sizes) - 2
            act = None if is_output else activations[layer_index]
            if act == "relu":
                scale = math.sqrt(2.0 / fan_in)
            else:
                scale = math.sqrt(2.0 / (fan_in + fan_out))
            data = np.zeros((fan_out, fan_in + 1), dtype=float)
            data[:, 1:] = rng.normal(loc=0.0, scale=scale,
                                     size=(fan_out, fan_in))
            self.weights.append(MatrixValue(data, op=f"weight_{layer_index}"))

    def __call__(self, features):
        current = (features if isinstance(features, MatrixValue)
                   else MatrixValue(features))
        if current.data.ndim != 2 or current.data.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Ожидался batch (*, {self.layer_sizes[0]}), "
                f"получена форма {current.data.shape}"
            )
        for i, weight in enumerate(self.weights):
            current = current.augment_with_ones() @ weight.T
            if i < len(self.hidden_activations):
                act = self.hidden_activations[i]
                current = current.relu() if act == "relu" else current.tanh()
        return current

    def parameters(self):
        return list(self.weights)

    def evaluate(self, features, labels):
        values = np.asarray(features, dtype=float)
        targets = np.asarray(labels, dtype=int)
        if values.ndim != 2 or targets.shape != (values.shape[0],):
            raise ValueError("Формы features и labels не согласованы")
        logits = self(values)
        loss = float(softmax_cross_entropy(logits, targets).data)
        predictions = np.argmax(softmax_rows(logits.data), axis=1)
        return {
            "loss": loss,
            "accuracy": float(np.mean(predictions == targets)),
        }

    def predict_proba(self, features):
        return softmax_rows(self(features).data)

    def predict(self, features):
        return np.argmax(self.predict_proba(features), axis=1)

    def fit(self, features, labels, validation_data, epochs, batch_size,
            learning_rate, shuffle_seed, checkpoint):
        train_features = np.asarray(features, dtype=float)
        train_labels = np.asarray(labels, dtype=int)
        val_features = np.asarray(validation_data[0], dtype=float)
        val_labels = np.asarray(validation_data[1], dtype=int)

        if learning_rate <= 0.0 or batch_size <= 0 or epochs <= 0:
            raise ValueError("learning_rate, batch_size, epochs должны быть > 0")
        if train_labels.shape != (train_features.shape[0],):
            raise ValueError("Train features/labels несогласованы")
        if val_labels.shape != (val_features.shape[0],):
            raise ValueError("Validation features/labels несогласованы")
        if not isinstance(checkpoint, BestValidationCheckpoint):
            raise ValueError("checkpoint должен быть BestValidationCheckpoint")

        rng = np.random.default_rng(shuffle_seed)
        history = TrainingHistory()

        for epoch in range(epochs):
            for batch_x, batch_y in iterate_minibatches(
                train_features, train_labels, batch_size, rng, shuffle=True,
            ):
                logits = self(batch_x)
                loss = softmax_cross_entropy(logits, batch_y)
                if not np.isfinite(loss.data).all():
                    raise FloatingPointError("Неконечная batch loss")
                loss.backward()
                params = self.parameters()
                for p in params:
                    if p.grad.shape != p.data.shape:
                        raise RuntimeError("Форма градиента не совпала")
                    if not np.isfinite(p.grad).all():
                        raise FloatingPointError("Неконечный градиент")
                # Все градиенты относятся к одному forward — обновляем разом.
                for p in params:
                    p.data -= learning_rate * p.grad

            train_metrics = self.evaluate(train_features, train_labels)
            val_metrics = self.evaluate(val_features, val_labels)
            history.append(train_metrics, val_metrics)
            checkpoint.update(self, epoch, val_metrics)
            history.best_epoch = checkpoint.best_epoch
            history.best_validation_loss = checkpoint.best_validation_loss
        return history


def count_parameters(model):
    return sum(p.data.size for p in model.parameters())