"""Матричный reverse-mode autograd и fused softmax–CCE.

Реализует этапы 4–10 лабораторной работы:
    * класс MatrixValue (DAG, callbacks, backward);
    * транспонирование, @, augment_with_ones, relu, tanh;
    * fused softmax + categorical cross-entropy.
"""
from __future__ import annotations

import numpy as np


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Устойчивый softmax по строкам (только для predict/evaluate)."""
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2:
        raise ValueError("logits должен иметь форму (B, C)")
    if values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("logits должен содержать B > 0 и C >= 2")
    if not np.isfinite(values).all():
        raise ValueError("logits должен содержать только конечные числа")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


class MatrixValue:
    """Одна матричная вершина динамического вычислительного графа."""

    def __init__(self, data, parents=(), op=":"):
        values = np.asarray(data, dtype=float)
        if values.ndim not in (0, 2):
            raise ValueError("MatrixValue поддерживает только scalar или 2D data")
        self.data = values.copy()
        self.grad = np.zeros_like(self.data)
        self.parents = tuple(parents)
        self.op = op
        self._backward = lambda: None

    # ---------- операции ----------
    def transpose(self):
        if self.data.ndim != 2:
            raise ValueError("Транспонировать можно только двумерный узел")
        result = MatrixValue(self.data.T, parents=(self,), op="transpose")

        def _backward():
            self.grad += result.grad.T

        result._backward = _backward
        return result

    @property
    def T(self):
        return self.transpose()

    def __matmul__(self, other):
        if not isinstance(other, MatrixValue):
            raise TypeError("Оператор @ принимает два MatrixValue")
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("Оператор @ принимает только двумерные матрицы")
        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError(
                f"Несогласованные формы для @: "
                f"{self.data.shape} и {other.data.shape}"
            )
        result = MatrixValue(
            self.data @ other.data, parents=(self, other), op="matmul",
        )

        def _backward():
            self.grad += result.grad @ other.data.T
            other.grad += self.data.T @ result.grad

        result._backward = _backward
        return result

    def augment_with_ones(self):
        if self.data.ndim != 2:
            raise ValueError("Расширять можно только двумерный узел")
        ones = np.ones((self.data.shape[0], 1), dtype=float)
        result = MatrixValue(
            np.concatenate((ones, self.data), axis=1),
            parents=(self,), op="augment_with_ones",
        )

        def _backward():
            self.grad += result.grad[:, 1:]

        result._backward = _backward
        return result

    def relu(self):
        if self.data.ndim != 2:
            raise ValueError("ReLU применяется к двумерной матрице")
        result = MatrixValue(
            np.maximum(0.0, self.data), parents=(self,), op="relu",
        )

        def _backward():
            self.grad += result.grad * (self.data > 0.0)

        result._backward = _backward
        return result

    def tanh(self):
        if self.data.ndim != 2:
            raise ValueError("tanh применяется к двумерной матрице")
        result_data = np.tanh(self.data)
        result = MatrixValue(result_data, parents=(self,), op="tanh")

        def _backward():
            self.grad += result.grad * (1.0 - result_data ** 2)

        result._backward = _backward
        return result

    # ---------- обратный проход ----------
    def backward(self, seed=None):
        topo, visited = [], set()

        def build(node):
            if node in visited:
                return
            visited.add(node)
            for parent in node.parents:
                build(parent)
            topo.append(node)

        build(self)
        for node in topo:
            node.grad = np.zeros_like(node.data)

        if seed is None:
            if self.data.ndim != 0:
                raise ValueError("Для нескалярного результата требуется seed")
            seed_values = np.ones_like(self.data)
        else:
            seed_values = np.asarray(seed, dtype=float)
            if seed_values.shape != self.data.shape:
                raise ValueError(
                    f"Форма seed {seed_values.shape} не равна {self.data.shape}"
                )

        self.grad = seed_values.copy()
        for node in reversed(topo):
            node._backward()


def softmax_cross_entropy(logits: MatrixValue, targets) -> MatrixValue:
    """Fused softmax–CCE: возвращает скалярный узел средней потери.

    Callback использует строго выведенный градиент ``(P − Y) / B``.
    """
    if not isinstance(logits, MatrixValue):
        raise TypeError("logits должен быть MatrixValue")
    if logits.data.ndim != 2:
        raise ValueError("logits должен иметь форму (B, C)")
    if not np.isfinite(logits.data).all():
        raise ValueError("logits должен содержать только конечные числа")

    batch_size, class_count = logits.data.shape
    if batch_size == 0 or class_count < 2:
        raise ValueError("logits должен содержать B > 0 и C >= 2")

    raw_labels = np.asarray(targets)
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError("targets должен содержать целочисленные метки")
    labels = raw_labels.astype(int, copy=False)
    if labels.shape != (batch_size,):
        raise ValueError("targets должен иметь форму (B,)")
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError("В targets есть класс вне диапазона [0, C)")

    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    log_normalizer = np.log(exponentials.sum(axis=1))
    selected_shifted_logits = shifted[np.arange(batch_size), labels]
    loss_value = np.mean(log_normalizer - selected_shifted_logits)

    result = MatrixValue(
        np.array(loss_value), parents=(logits,), op="softmax_cross_entropy",
    )

    def _backward():
        logits_gradient = probabilities.copy()
        logits_gradient[np.arange(batch_size), labels] -= 1.0
        logits_gradient /= batch_size
        logits.grad += result.grad * logits_gradient

    result._backward = _backward
    return result