import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


train_inputs = np.array([[0, 0, 1],
                         [1, 1, 1],
                         [1, 0, 1],
                         [0, 1, 1]])

print("Введенные данные")
print(train_inputs)

train_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Случайные веса:")
print(synaptic_weights)

# Метод обратного распространения
# где range - число повторений обучения
for i in range(500000):
    input_layer = train_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    err = train_outputs - outputs
    adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))

    synaptic_weights += adjustments

print("Веса после обучения:")
print(synaptic_weights)

print("Результат после обучения:")
# noinspection PyUnboundLocalVariable
print(outputs)
