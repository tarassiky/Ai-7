"""
Лабораторная работа: скалярное автоматическое дифференцирование
и обучение одного нейрона.

Запуск:
    python autograd_lab.py

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
# Класс Value: скалярная вершина вычислительного графа
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

    # --- сложение ---------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(
            self.data + other.data,
            parents=(self, other),
            op="+",
        )
        def backward():
            self.grad += result.grad
            other.grad += result.grad
        result._backward = backward
        return result

    __radd__ = __add__

    # --- умножение --------------------------------------------------------
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(
            self.data * other.data,
            parents=(self, other),
            op="*",
        )
        def backward():
            self.grad += result.grad * other.data
            other.grad += result.grad * self.data
        result._backward = backward
        return result

    __rmul__ = __mul__

    # --- отрицание и вычитание -------------------------------------------
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

    # --- степень с постоянным показателем ---------------------------------
    def __pow__(self, power):
        result = Value(
            self.data ** power,
            parents=(self,),
            op=f"pow({power})",
        )
        def backward():
            local_derivative = power * self.data ** (power - 1)
            self.grad += result.grad * local_derivative
        result._backward = backward
        return result

    # --- элементарные функции ---------------------------------------------
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

    # --- обратный проход --------------------------------------------------
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
# Нейрон с гиперболическим тангенсом
# ---------------------------------------------------------------------------

class TanhNeuron:
    def __init__(self, weight, bias):
        self.weight = Value(weight)
        self.bias = Value(bias)

    def __call__(self, x):
        return (self.weight * x + self.bias).tanh()

    def parameters(self):
        return [self.weight, self.bias]


# ---------------------------------------------------------------------------
# Функция потерь (MSE)
# ---------------------------------------------------------------------------

def mean_squared_loss(neuron, inputs, targets):
    total = Value(0.0)
    for input_number, target_number in zip(inputs, targets):
        prediction = neuron(input_number)
        error = prediction - target_number
        total = total + error ** 2
    return total * (1.0 / len(inputs))


def plain_loss(weight, bias, inputs, targets):
    total = 0.0
    for input_number, target_number in zip(inputs, targets):
        prediction = math.tanh(weight * input_number + bias)
        total += (prediction - target_number) ** 2
    return total / len(inputs)


# ---------------------------------------------------------------------------
# Эксперименты
# ---------------------------------------------------------------------------

def experiment_addition():
    left = Value(2.5)
    right = Value(-1.0)
    sum_value = left + right
    assert_close(sum_value.data, 1.5)
    assert sum_value.parents == (left, right)
    assert sum_value.op == "+"
    print("Сложение в прямом проходе: OK")


def experiment_multiplication():
    left = Value(4.0)
    right = Value(-0.5)
    product_value = left * right
    scaled_value = 3.0 * left
    assert_close(product_value.data, -2.0)
    assert_close(scaled_value.data, 12.0)
    assert product_value.parents == (left, right)
    assert product_value.op == "*"
    print("Умножение в прямом проходе: OK")


def experiment_neg_sub_power():
    source = Value(3.0)
    negative = -source
    difference = 5.0 - source
    cube = source ** 3
    assert_close(negative.data, -3.0)
    assert_close(difference.data, 2.0)
    assert_close(cube.data, 27.0)
    assert negative.op == "neg"
    assert cube.op == "pow(3)"
    print("Отрицание, вычитание и степень в прямом проходе: OK")


def experiment_elementary():
    angle = Value(0.0)
    positive = Value(2.0)
    activation_input = Value(0.7)
    assert_close(angle.sin().data, 0.0)
    assert_close(positive.exp().log().data, 2.0)
    assert_close(activation_input.tanh().data, math.tanh(0.7))
    print("Элементарные функции в прямом проходе: OK")


def experiment_backward_shared_graph():
    a = Value(2.0)
    objective = a * a + a
    objective.backward()
    assert_close(objective.data, 6.0)
    assert_close(a.grad, 5.0)
    assert_close(objective.grad, 1.0)
    print("Обратный проход и накопление двух вкладов: OK")


def experiment_manual_backward_trace():
    a = Value(0.5)
    b = Value(1.0)
    m = a * b
    s = m + a
    j = s.tanh()

    for node in (a, b, m, s, j):
        node.grad = 0.0
    j.grad = 1.0

    expected_gs = 1.0 - j.data ** 2

    j._backward()
    assert_close(s.grad, expected_gs)
    assert_close(m.grad, 0.0)
    assert_close(a.grad, 0.0)
    print("После J._backward():", "g_s =", s.grad)

    s._backward()
    assert_close(m.grad, expected_gs)
    assert_close(a.grad, expected_gs)
    print("После s._backward():", "g_m =", m.grad, "; первый вклад в g_a =", a.grad)

    m._backward()
    assert_close(a.grad, expected_gs * (1.0 + b.data))
    assert_close(b.grad, expected_gs * a.data)
    print("После m._backward():", "g_a =", a.grad, "; g_b =", b.grad)
    print("Ручная трассировка: OK")


def experiment_finite_difference():
    def plain_function(x):
        return math.sin(x) * math.exp(x) + x ** 3

    point = 0.37
    step = 1e-6

    input_value = Value(point)
    objective = input_value.sin() * input_value.exp() + input_value ** 3
    objective.backward()

    numerical_gradient = (
        plain_function(point + step) - plain_function(point - step)
    ) / (2 * step)
    absolute_error = abs(input_value.grad - numerical_gradient)

    assert absolute_error < 1e-5, absolute_error
    print("Градиент Value :", input_value.grad)
    print("Градиент разностью :", numerical_gradient)
    print("Абсолютная ошибка :", absolute_error)
    print("Центральная разность: OK")


def experiment_neuron_forward():
    neuron = TanhNeuron(weight=2.0, bias=-1.0)
    prediction = neuron(0.5)
    assert_close(prediction.data, 0.0)
    assert neuron.parameters() == [neuron.weight, neuron.bias]
    print("Прямой проход одного нейрона: OK")


def experiment_loss_and_gradient():
    inputs = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
    targets = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0]

    gradient_neuron = TanhNeuron(weight=0.0, bias=0.0)
    initial_loss = mean_squared_loss(gradient_neuron, inputs, targets)
    initial_loss.backward()

    step = 1e-6
    numerical_weight_gradient = (
        plain_loss(step, 0.0, inputs, targets)
        - plain_loss(-step, 0.0, inputs, targets)
    ) / (2 * step)
    numerical_bias_gradient = (
        plain_loss(0.0, step, inputs, targets)
        - plain_loss(0.0, -step, inputs, targets)
    ) / (2 * step)

    assert_close(initial_loss.data, 1.0)
    assert_close(
        gradient_neuron.weight.grad,
        numerical_weight_gradient,
        tolerance=1e-5,
    )
    assert_close(
        gradient_neuron.bias.grad,
        numerical_bias_gradient,
        tolerance=1e-5,
    )
    print("Начальная ошибка:", initial_loss.data)
    print("Градиент по весу:", gradient_neuron.weight.grad)
    print("Градиент по смещению:", gradient_neuron.bias.grad)
    print("Проверка градиентов параметров: OK")


def experiment_train_neuron():
    inputs = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
    targets = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0]

    learning_rate = 0.2
    epochs = 80

    trained_neuron = TanhNeuron(weight=0.0, bias=0.0)
    losses = []

    for epoch in range(epochs):
        loss = mean_squared_loss(trained_neuron, inputs, targets)
        loss.backward()
        losses.append(loss.data)
        for parameter in trained_neuron.parameters():
            parameter.data -= learning_rate * parameter.grad

    final_predictions = [
        trained_neuron(input_number).data for input_number in inputs
    ]
    predicted_labels = [
        1.0 if prediction >= 0.0 else -1.0
        for prediction in final_predictions
    ]
    accuracy = sum(
        prediction == target
        for prediction, target in zip(predicted_labels, targets)
    ) / len(targets)

    assert losses[-1] < 0.03, losses[-1]
    assert accuracy == 1.0, accuracy
    assert abs(trained_neuron.weight.data) > 0.1
    assert abs(trained_neuron.bias.data) > 0.1

    print("Ошибка в первой эпохе:", losses[0])
    print("Ошибка в последней эпохе:", losses[-1])
    print("Вес после обучения:", trained_neuron.weight.data)
    print("Смещение после обучения:", trained_neuron.bias.data)
    print("Прогнозы:", final_predictions)
    print("Точность:", accuracy)
    print("Обучение нейрона: OK")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Прямой проход")
    print("=" * 60)
    experiment_addition()
    experiment_multiplication()
    experiment_neg_sub_power()
    experiment_elementary()

    print()
    print("=" * 60)
    print("Обратный проход")
    print("=" * 60)
    experiment_backward_shared_graph()
    experiment_manual_backward_trace()
    experiment_finite_difference()

    print()
    print("=" * 60)
    print("Нейрон")
    print("=" * 60)
    experiment_neuron_forward()
    experiment_loss_and_gradient()
    experiment_train_neuron()

    print()
    print("Все проверки пройдены успешно.")


if __name__ == "__main__":
    main()