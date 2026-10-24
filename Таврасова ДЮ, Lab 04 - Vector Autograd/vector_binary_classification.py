"""
Лабораторная работа: векторный вход и бинарная классификация
на скалярном autograd.

Запуск:
    python vector_binary_classification.py

Все проверки выполняются через assert; при успехе печатаются сообщения OK.
"""

import math


# ---------------------------------------------------------------------------
# Вспомогательная функция сравнения вещественных чисел
# ---------------------------------------------------------------------------

def assert_close(actual, expected, tolerance=1e-9):
    difference = abs(actual - expected)
    assert difference <= tolerance, (
        f"Ожидалось {expected}, получено {actual}, "
        f"абсолютная ошибка {difference}"
    )


# ---------------------------------------------------------------------------
# Скалярная вершина вычислительного графа (из предыдущей лабораторной)
# ---------------------------------------------------------------------------

class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents
        self.op = op
        self._backward = lambda: None

    def __repr__(self):
        return (
            f"Value(data={self.data:.6g}, grad={self.grad:.6g}, "
            f"op={self.op!r})"
        )

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, parents=(self, other), op="+")
        def backward():
            self.grad += result.grad
            other.grad += result.grad
        result._backward = backward
        return result

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, parents=(self, other), op="*")
        def backward():
            self.grad += result.grad * other.data
            other.grad += result.grad * self.data
        result._backward = backward
        return result

    __rmul__ = __mul__

    def __neg__(self):
        result = Value(-self.data, parents=(self,), op="neg")
        def backward():
            self.grad += result.grad * (-1)
        result._backward = backward
        return result

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, power):
        result = Value(
            self.data ** power, parents=(self,), op=f"pow({power})"
        )
        def backward():
            local_derivative = power * self.data ** (power - 1)
            self.grad += result.grad * local_derivative
        result._backward = backward
        return result

    def sin(self):
        result = Value(math.sin(self.data), parents=(self,), op="sin")
        def backward():
            self.grad += result.grad * math.cos(self.data)
        result._backward = backward
        return result

    def exp(self):
        result = Value(math.exp(self.data), parents=(self,), op="exp")
        def backward():
            self.grad += result.grad * result.data
        result._backward = backward
        return result

    def log(self):
        result = Value(math.log(self.data), parents=(self,), op="log")
        def backward():
            self.grad += result.grad * (1 / self.data)
        result._backward = backward
        return result

    def tanh(self):
        result = Value(math.tanh(self.data), parents=(self,), op="tanh")
        def backward():
            self.grad += result.grad * (1 - result.data ** 2)
        result._backward = backward
        return result

    def backward(self):
        topo = []
        visited = set()
        def build(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    build(parent)
                topo.append(node)
        build(self)
        for node in topo:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# ---------------------------------------------------------------------------
# Векторные вспомогательные функции (на самом деле — цикл скалярных операций)
# ---------------------------------------------------------------------------

def affine(weights, features):
    """Скалярное развёртывание w0*1 + sum_j w_j * x_j."""
    augmented_features = [1.0, *features]
    assert len(weights) == len(augmented_features), (
        "Число весов должно включать вес постоянной координаты"
    )
    total = weights[0] * augmented_features[0]
    for weight, feature in zip(weights[1:], augmented_features[1:]):
        total = total + weight * feature
    return total


def sigmoid(logit):
    return (1.0 + (-logit).exp()) ** -1


def binary_cross_entropy(probability, target):
    target = float(target)
    assert target in (0.0, 1.0), (
        "Метка бинарной классификации должна быть 0 или 1"
    )
    return -(
        target * probability.log()
        + (1.0 - target) * (1.0 - probability).log()
    )


def mean_binary_cross_entropy(model, inputs, targets):
    assert len(inputs) == len(targets), (
        "Число объектов и меток должно совпадать"
    )
    assert len(inputs) > 0, "Нужен хотя бы один объект"
    total = Value(0.0)
    for features, target in zip(inputs, targets):
        probability = model(features)
        total = total + binary_cross_entropy(probability, target)
    return total * (1.0 / len(inputs))


def predict_label(probability):
    return 1 if probability.data >= 0.5 else 0


def classification_accuracy(model, inputs, targets):
    predictions = [predict_label(model(features)) for features in inputs]
    correct = sum(
        prediction == target
        for prediction, target in zip(predictions, targets)
    )
    return correct / len(targets)


def numerical_gradient(model, parameter, inputs, targets, step=1e-6):
    original_data = parameter.data
    parameter.data = original_data + step
    loss_plus = mean_binary_cross_entropy(model, inputs, targets).data
    parameter.data = original_data - step
    loss_minus = mean_binary_cross_entropy(model, inputs, targets).data
    parameter.data = original_data
    return (loss_plus - loss_minus) / (2.0 * step)


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------

class VectorBinaryNeuron:
    """Один sigmoid-нейрон с векторным входом."""

    def __init__(self, weights):
        assert len(weights) > 1, (
            "Нужны вес смещения и хотя бы один входной вес"
        )
        self.weights = [Value(weight) for weight in weights]

    def __call__(self, features):
        logit = affine(self.weights, features)
        return sigmoid(logit)

    def parameters(self):
        return self.weights


class OneHiddenBinaryClassifier:
    """Один скрытый tanh-нейрон и sigmoid-выход."""

    def __init__(self, hidden_weights, output_weights):
        assert len(hidden_weights) > 1, (
            "Нужны вес смещения и хотя бы один входной вес"
        )
        assert len(output_weights) == 2, (
            "Выходу нужны вес смещения и вес скрытого состояния"
        )
        self.hidden_weights = [Value(weight) for weight in hidden_weights]
        self.output_weights = [Value(weight) for weight in output_weights]

    def hidden_state(self, features):
        hidden_logit = affine(self.hidden_weights, features)
        return hidden_logit.tanh()

    def __call__(self, features):
        hidden = self.hidden_state(features)
        output_logit = affine(self.output_weights, [hidden])
        return sigmoid(output_logit)

    def parameters(self):
        return self.hidden_weights + self.output_weights


class HiddenLayerBinaryClassifier:
    """Слой из произвольного числа скрытых tanh-нейронов и sigmoid-выход."""

    def __init__(self, hidden_layer_weights, output_weights):
        assert len(hidden_layer_weights) > 0, "Нужен хотя бы один скрытый нейрон"
        feature_count = len(hidden_layer_weights[0])
        assert feature_count > 1, (
            "У скрытого нейрона нужны смещение и хотя бы один вход"
        )
        assert all(
            len(weights) == feature_count
            for weights in hidden_layer_weights
        ), "Все скрытые нейроны должны иметь одинаковое число входов"
        assert len(output_weights) == len(hidden_layer_weights) + 1, (
            "Выходу нужны вес смещения и веса всех скрытых состояний"
        )
        self.hidden_weights = [
            [Value(weight) for weight in weights]
            for weights in hidden_layer_weights
        ]
        self.output_weights = [Value(weight) for weight in output_weights]

    def hidden_states(self, features):
        return [
            affine(weights, features).tanh()
            for weights in self.hidden_weights
        ]

    def __call__(self, features):
        hidden = self.hidden_states(features)
        output_logit = affine(self.output_weights, hidden)
        return sigmoid(output_logit)

    def parameters(self):
        parameters = []
        for weights in self.hidden_weights:
            parameters.extend(weights)
        parameters.extend(self.output_weights)
        return parameters


# ---------------------------------------------------------------------------
# Датасеты
# ---------------------------------------------------------------------------

OR_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
OR_TARGETS = [0, 1, 1, 1]

AND_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
AND_TARGETS = [0, 0, 0, 1]

XOR_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_TARGETS = [0, 1, 1, 0]


# ---------------------------------------------------------------------------
# Эксперименты
# ---------------------------------------------------------------------------

def experiment_scalar_graph_refresh():
    a = Value(2.0)
    objective = a * a + a
    objective.backward()
    assert_close(objective.data, 6.0)
    assert_close(a.grad, 5.0)
    print("Скалярный Value готов: J =", objective.data, ", dJ/da =", a.grad)


def experiment_or_forward():
    or_neuron = VectorBinaryNeuron(weights=[0.0, 0.1, -0.2])
    or_probabilities = [or_neuron(features).data for features in OR_INPUTS]

    assert_close(or_probabilities[0], 0.5)
    assert_close(or_probabilities[1], 1.0 / (1.0 + math.exp(0.2)))
    assert len(or_neuron.parameters()) == 3
    assert all(isinstance(p, Value) for p in or_neuron.parameters())
    print("Начальные вероятности OR:", or_probabilities)
    print("Прямой проход OR: OK")


def experiment_or_gradients():
    gradient_or = VectorBinaryNeuron(weights=[0.0, 0.1, -0.2])
    or_loss = mean_binary_cross_entropy(gradient_or, OR_INPUTS, OR_TARGETS)
    or_loss.backward()

    for parameter in gradient_or.parameters():
        expected_gradient = numerical_gradient(
            gradient_or, parameter, OR_INPUTS, OR_TARGETS
        )
        assert_close(parameter.grad, expected_gradient, tolerance=1e-5)

    assert any(abs(p.grad) > 1e-8 for p in gradient_or.parameters())
    print("BCE для OR до обучения:", or_loss.data)
    print("Градиенты OR:", [p.grad for p in gradient_or.parameters()])
    print("Градиенты OR: OK")


def experiment_train_or():
    learning_rate = 0.5
    epochs = 300

    trained_or = VectorBinaryNeuron(weights=[0.0, 0.1, -0.2])
    or_losses = []

    for epoch in range(epochs):
        loss = mean_binary_cross_entropy(trained_or, OR_INPUTS, OR_TARGETS)
        loss.backward()
        or_losses.append(loss.data)
        for parameter in trained_or.parameters():
            parameter.data -= learning_rate * parameter.grad

    or_predictions = [trained_or(f).data for f in OR_INPUTS]
    or_accuracy = classification_accuracy(trained_or, OR_INPUTS, OR_TARGETS)

    assert or_losses[-1] < 0.1, or_losses[-1]
    assert or_losses[-1] < or_losses[0]
    assert or_accuracy == 1.0, or_accuracy
    assert all(abs(p.data) > 0.1 for p in trained_or.parameters())

    print("OR: loss в первой эпохе:", or_losses[0])
    print("OR: loss в последней эпохе:", or_losses[-1])
    print("OR: вероятности:", or_predictions)
    print("OR: точность:", or_accuracy)
    print("Обучение OR: OK")


def experiment_and_gradients():
    gradient_and = OneHiddenBinaryClassifier(
        hidden_weights=[0.0, 0.1, -0.2],
        output_weights=[0.0, 0.3],
    )
    and_loss = mean_binary_cross_entropy(
        gradient_and, AND_INPUTS, AND_TARGETS
    )
    and_loss.backward()

    for parameter in gradient_and.parameters():
        expected_gradient = numerical_gradient(
            gradient_and, parameter, AND_INPUTS, AND_TARGETS
        )
        assert_close(parameter.grad, expected_gradient, tolerance=1e-5)

    assert_close(gradient_and.hidden_weights[1].grad, 0.0)
    assert any(abs(p.grad) > 1e-8 for p in gradient_and.parameters())
    print("BCE для AND до обучения:", and_loss.data)
    print("Градиенты AND:", [p.grad for p in gradient_and.parameters()])
    print("Градиенты AND: OK")


def experiment_train_and():
    learning_rate = 0.3
    epochs = 300

    trained_and = OneHiddenBinaryClassifier(
        hidden_weights=[0.0, 0.1, -0.2],
        output_weights=[0.0, 0.3],
    )
    and_losses = []

    for epoch in range(epochs):
        loss = mean_binary_cross_entropy(
            trained_and, AND_INPUTS, AND_TARGETS
        )
        loss.backward()
        and_losses.append(loss.data)
        for parameter in trained_and.parameters():
            parameter.data -= learning_rate * parameter.grad

    and_predictions = [trained_and(f).data for f in AND_INPUTS]
    and_accuracy = classification_accuracy(
        trained_and, AND_INPUTS, AND_TARGETS
    )

    assert and_losses[-1] < 0.05, and_losses[-1]
    assert and_losses[-1] < and_losses[0]
    assert and_accuracy == 1.0, and_accuracy
    assert all(abs(p.grad) > 1e-8 for p in trained_and.parameters())

    print("AND: loss в первой эпохе:", and_losses[0])
    print("AND: loss в последней эпохе:", and_losses[-1])
    print("AND: вероятности:", and_predictions)
    print("AND: точность:", and_accuracy)
    print("Обучение AND: OK")


def experiment_xor_gradients():
    gradient_xor = HiddenLayerBinaryClassifier(
        hidden_layer_weights=[[0.1, 0.4, -0.3], [-0.1, -0.2, 0.5]],
        output_weights=[0.0, 0.3, -0.4],
    )
    xor_loss = mean_binary_cross_entropy(
        gradient_xor, XOR_INPUTS, XOR_TARGETS
    )
    xor_loss.backward()

    for parameter in gradient_xor.parameters():
        expected_gradient = numerical_gradient(
            gradient_xor, parameter, XOR_INPUTS, XOR_TARGETS
        )
        assert_close(parameter.grad, expected_gradient, tolerance=1e-5)

    assert len(gradient_xor.parameters()) == 9
    assert any(abs(p.grad) > 1e-8 for p in gradient_xor.parameters())
    print("BCE для XOR до обучения:", xor_loss.data)
    print("Градиенты XOR:", [p.grad for p in gradient_xor.parameters()])
    print("Градиенты XOR: OK")


def experiment_train_xor():
    learning_rate = 0.5
    epochs = 500

    trained_xor = HiddenLayerBinaryClassifier(
        hidden_layer_weights=[[0.1, 0.4, -0.3], [-0.1, -0.2, 0.5]],
        output_weights=[0.0, 0.3, -0.4],
    )
    xor_losses = []

    for epoch in range(epochs):
        loss = mean_binary_cross_entropy(
            trained_xor, XOR_INPUTS, XOR_TARGETS
        )
        loss.backward()
        xor_losses.append(loss.data)
        for parameter in trained_xor.parameters():
            parameter.data -= learning_rate * parameter.grad

    xor_predictions = [trained_xor(f).data for f in XOR_INPUTS]
    xor_accuracy = classification_accuracy(
        trained_xor, XOR_INPUTS, XOR_TARGETS
    )

    assert xor_losses[-1] < 0.03, xor_losses[-1]
    assert xor_losses[-1] < xor_losses[0]
    assert xor_accuracy == 1.0, xor_accuracy
    assert all(abs(p.data) > 0.1 for p in trained_xor.parameters())

    print("XOR: loss в первой эпохе:", xor_losses[0])
    print("XOR: loss в последней эпохе:", xor_losses[-1])
    print("XOR: вероятности:", xor_predictions)
    print("XOR: точность:", xor_accuracy)
    print("Обучение XOR: OK")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Подготовка: скалярный граф")
    print("=" * 60)
    experiment_scalar_graph_refresh()

    print()
    print("=" * 60)
    print("Этап 1: один sigmoid-нейрон на OR")
    print("=" * 60)
    experiment_or_forward()
    experiment_or_gradients()
    experiment_train_or()

    print()
    print("=" * 60)
    print("Этап 2: один скрытый нейрон на AND")
    print("=" * 60)
    experiment_and_gradients()
    experiment_train_and()

    print()
    print("=" * 60)
    print("Этап 3: два скрытых нейрона на XOR")
    print("=" * 60)
    experiment_xor_gradients()
    experiment_train_xor()

    print()
    print("Все проверки пройдены успешно.")


if __name__ == "__main__":
    main()