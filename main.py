import random
import numpy as np
import matplotlib.pyplot as plt
from BPNetwork import BPNetwork

HIDDEN_UNIT_NUM = 50

# nn = BPNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35,
#                output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
# for i in range(10000):
#     nn.train([0.05, 0.1], [0.01, 0.09])
#     print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.09]]]), 9))

# training_sets = [
#     [[0, 0], [0]],
#     [[0, 1], [1]],
#     [[1, 0], [1]],
#     [[1, 1], [0]]
# ]
#
# nn = BPNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
# for i in range(50000):
#     training_inputs, training_outputs = random.choice(training_sets)
#     nn.train(training_inputs, training_outputs)
#     print(i, nn.calculate_total_error(training_sets))

hlw_list = []
olw_list = []
for i in range(HIDDEN_UNIT_NUM):
    hlw_list.append((random.random() * 2 - 1) / 100)
    olw_list.append((random.random() * 2 - 1) / 100)
hlw = np.array(hlw_list)
olw = np.array(olw_list)

hlb = -random.random() / 100
olb = -random.random() / 100

# training_sets = []
nn = BPNetwork(1, HIDDEN_UNIT_NUM, 1, hidden_layer_weights=hlw, hidden_layer_bias=hlb,
               output_layer_weights=olw, output_layer_bias=olb)
for i in range(3000000):
    # training_inputs, training_outputs = random.choice(training_sets)
    input = [random.random() * np.pi * 2 - np.pi]
    output = [np.sin(input[0])]
    training_inputs, training_outputs = [input, output]
    # training_sets.append([input, output])
    nn.train(training_inputs, training_outputs)
    # print(i, nn.calculate_total_error(training_sets))
    print(i)

x = np.arange(-np.pi, np.pi, 0.01)
y = []
Y = []
for i in x:
    y.append(np.sin(i))
    Y.append(nn.feed_forward([i]))
plt.plot(x, y)
plt.plot(x, Y, color='red', linestyle='--')
plt.show()

diffAverage = 0
for i in range(len(y)):
    diffAverage += np.abs(Y[i] - y[i])
diffAverage = diffAverage / len(y)
print(diffAverage)
