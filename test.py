import random
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

from BPNetwork import BPNetwork

HIDDEN_UNIT_NUM = 50


# nn = BPNetwork(2, [2, 2], 2, hidden_layers_weights=[[0.15, 0.2, 0.25, 0.3], [0.15, 0.2, 0.25, 0.3]],
#                hidden_layers_bias=[0.35, 0.35],
#                output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
# for i in range(10000):
#     nn.train([0.05, 0.1], [0.01, 0.09])
#     print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.09]]]), 9))
# print(nn.feed_forward([0.05, 0.1]))

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

# # sinx 拟合
# hlw = []
# olw = []
# for i in range(HIDDEN_UNIT_NUM):
#     hlw.append((random.random() * 2 - 1) / 1000)
#     olw.append((random.random() * 2 - 1) / 1000)
#
# hlb = -random.random() / 100
# olb = -random.random() / 100
#
# # training_sets = []
# nn = BPNetwork(1, [HIDDEN_UNIT_NUM], 1, hidden_layers_weights=[hlw], hidden_layers_bias=[hlb],
#                output_layer_weights=olw, output_layer_bias=olb)
# for i in range(100000):
#     # training_inputs, training_outputs = random.choice(training_sets)
#     input = [random.random() * np.pi * 2 - np.pi]
#     output = [np.sin(input[0])]
#     training_inputs, training_outputs = [input, output]
#     # training_sets.append([input, output])
#     nn.train(training_inputs, training_outputs)
#     # print(i, nn.calculate_total_error(training_sets))
#     print(i)
#
# x = np.arange(-np.pi, np.pi, 0.01)
# y = []
# Y = []
# for i in x:
#     y.append(np.sin(i))
#     Y.append(nn.feed_forward([i]))
# plt.plot(x, y)
# plt.plot(x, Y, color='red', linestyle='--')
# plt.show()
#
# diffAverage = 0
# for i in range(len(y)):
#     diffAverage += np.abs(Y[i] - y[i])
# diffAverage = diffAverage / len(y)
# print(diffAverage)


# 汉字分类
# 产生训练数据集(600）
def convert_to_bw(im, path):
    WHITE = 255
    BLACK = 0
    im = im.convert("L")
    im.save(path)
    im = im.point(lambda x: WHITE if x > 196 else BLACK)
    im = im.convert('1')
    im.save(path)
    return im


training_inputs = np.empty([12 * 600, 28 * 28])
training_outputs = np.empty([12 * 600, 12])
count = 0
for c in range(12):
    ns = np.random.randint(600, size=600)
    for n in ns:
        img = Image.open('train/' + str(c + 1) + '/' + str(n + 1) + '.bmp')
        training_inputs[count] = np.resize(convert_to_bw(img, 'train/' + str(c + 1) + '/' + str(n + 1) + '.bmp'),
                                           (28 * 28))
        output = np.zeros(12)
        output[c] = 1
        training_outputs[count] = output
        count += 1

# 随机化训练数据
ns = np.random.randint(12 * 600, size=12 * 600)
randomized_training_inputs = np.empty([12 * 600, 28 * 28])
randomized_training_outputs = np.empty([12 * 600, 12])
randomized_count = 0
for n in ns:
    randomized_training_inputs[randomized_count] = training_inputs[n]
    randomized_training_outputs[randomized_count] = training_outputs[n]
    randomized_count += 1

# 产生测试集数据（20）
test_inputs = np.empty([12 * 20, 28 * 28])
test_outputs = np.empty([12 * 20])
count = 0
for c in range(12):
    for n in range(20):
        img = Image.open('train/' + str(c + 1) + '/' + str(600 + n + 1) + '.bmp')
        test_inputs[count] = np.resize(convert_to_bw(img, 'train/' + str(c + 1) + '/' + str(600 + n + 1) + '.bmp'),
                                       (28 * 28))
        test_outputs[count] = c
        count += 1

# 准备网络参数、开始训练
INPUT_NUM = 28 * 28
OUTPUT_NUM = 12
# 随机生成：
hlw = []
olw = []
for i in range(HIDDEN_UNIT_NUM * INPUT_NUM):
    hlw.append((random.random() * 2 - 1) / 1000)
for j in range(HIDDEN_UNIT_NUM * OUTPUT_NUM):
    olw.append((random.random() * 2 - 1) / 1000)

hlb = -random.random() / 100
olb = -random.random() / 100

# 从文件读取
# hlw = []
# with open('data/hlw_output_1604071206.449504.txt') as f:
#     for line in f:
#         hlw.append(float(line[0:-3]))
# print(hlw)
# olw = []
# with open('data/olw_output_1604071206.5246542.txt') as f:
#     for line in f:
#         olw.append(float(line[0:-3]))
# print(olw)
#
# b = []
# with open('data/olb_output_1604071206.617167.txt') as f:
#     for line in f:
#         b.append(float(line[0:-3]))
# hlb = b[0]
# olb = b[1]
# print(b)

nn = BPNetwork(INPUT_NUM, [HIDDEN_UNIT_NUM], OUTPUT_NUM, hidden_layers_weights=[hlw], hidden_layers_bias=[hlb],
               output_layer_weights=olw, output_layer_bias=olb, softmax_enabled=1)

TIMES = 8

for i in range(TIMES):
    for c in range(12 * 600):
        nn.train(randomized_training_inputs[c], randomized_training_outputs[c])
        print(str(i) + ": " + str(np.floor(c / 600)) + "/" + str(c % 600))

# 计算正确率
success = 0
for i in range(12 * 20):
    index = np.random.randint(12 * 20)
    output_list = np.array(nn.softmax_feed_forward(test_inputs[index])).tolist()
    print(output_list)
    max_index = output_list.index(max(output_list))
    print(max_index)
    if max_index == test_outputs[index]:
        success += 1
print(success)
result = success / (12 * 20)
print(result)

# 持久化
hl_weights = []
for hl in nn.hidden_layers:
    for n in hl.neurons:
        hl_weights.append(n.weights)
hlw_output = np.resize(np.array(hl_weights), (len(nn.hidden_layers) * HIDDEN_UNIT_NUM * INPUT_NUM))
np.savetxt('data/hlw_output_' + str(time.time()) + '.txt', hlw_output, fmt="%.17f", delimiter=',')

ol_weights = []
for n in nn.output_layer.neurons:
    ol_weights.append(n.weights)
ol_output = np.resize(np.array(ol_weights), HIDDEN_UNIT_NUM * OUTPUT_NUM)
np.savetxt('data/olw_output_' + str(time.time()) + '.txt', hlw_output, fmt="%.17f", delimiter=',')

b = []
for hl in nn.hidden_layers:
    b.append(hl.bias)
b.append(nn.output_layer.bias)
np.savetxt('data/b_output_' + str(time.time()) + '.txt', b, fmt="%.17f", delimiter=',')

# 过拟合测试
# print(randomized_training_inputs[1])
# print(randomized_training_outputs[1])
# for i in range(100):
#     nn.train(randomized_training_inputs[1], randomized_training_outputs[1])
#     print(i)
# print(nn.softmax_feed_forward(randomized_training_inputs[1]))
# output_list = np.array(nn.softmax_feed_forward(randomized_training_inputs[1])).tolist()
# max_index = output_list.index(max(output_list))
# print(max_index)
