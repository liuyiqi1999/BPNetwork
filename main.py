import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from PIL import Image

from BPNetwork import BPNetwork

HIDDEN_UNIT_NUM = 50


def sinx(path):
    INPUT_NUM = 1
    OUTPUT_NUM = 1

    if path == '':
        hlw = []
        olw = []
        for i in range(HIDDEN_UNIT_NUM):
            hlw.append((random.random() * 2 - 1) / 100)
            olw.append((random.random() * 2 - 1) / 100)

        hlb = -random.random() / 1
        olb = -random.random() / 1
    else:
        file1 = open(path, 'rb')
        nn = pickle.load(file1)

    # TIMES = 300000
    #
    # if path == '':
    #     nn = BPNetwork(1, [HIDDEN_UNIT_NUM], 1, hidden_layers_weights=[hlw], hidden_layers_bias=[hlb],
    #                    output_layer_weights=olw, output_layer_bias=olb)
    #     for i in range(TIMES):
    #         # training_inputs, training_outputs = random.choice(training_sets)
    #         input = [random.random() * np.pi * 2 - np.pi]
    #         output = [np.sin(input[0])]
    #         training_inputs, training_outputs = [input, output]
    #         # training_sets.append([input, output])
    #         nn.train(training_inputs, training_outputs)
    #         # print(i, nn.calculate_total_error(training_sets))
    #         print(i)

    # if path == '':
    #     file = open('data/s_'+str(TIMES)+'_' + str(time.time()) + '.data', 'wb')
    #     pickle.dump(nn, file)
    #     file.close()

    x = np.arange(-np.pi, np.pi, 0.01)
    y = []
    Y = []
    for i in x:
        y.append(np.sin(i))
        Y.append(nn.feed_forward([i]))
    plt.plot(x, y)
    plt.plot(x, Y, color='red', linestyle='--')
    plt.show()

    diff = 0
    for i in range(len(y)):
        diff += np.square(Y[i] - y[i])
    diff_average = diff / len(y)
    print(diff_average)


def convert_to_bw(im, path):
    WHITE = 255
    BLACK = 0
    im = im.convert("L")
    im.save(path)
    im = im.point(lambda x: WHITE if x > 196 else BLACK)
    im = im.convert('1')
    im.save(path)
    return im


def character(path):
    # 产生训练数据集(600）
    training_inputs = np.empty([12 * 600, 28 * 28])
    training_outputs = np.empty([12 * 600, 12])
    count = 0
    for c in range(12):
        ns = np.random.randint(600, size=600)
        for n in ns:
            img = Image.open('train/' + str(c + 1) + '/' + str(n + 1) + '.bmp')
            img = img.resize((28, 28) * Image.ANTIALIAS)
            training_inputs[count] = np.resize(
                convert_to_bw(img, 'train_cache/' + str(c + 1) + '/' + str(n + 1) + '.bmp'),
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
    # test_inputs = np.empty([12 * 20, 28 * 28])
    # test_outputs = np.empty([12 * 20])
    # count = 0
    # for c in range(12):
    #     for n in range(20):
    #         img = Image.open('train/' + str(c + 1) + '/' + str(600 + n + 1) + '.bmp')
    #         test_inputs[count] = np.resize(
    #             convert_to_bw(img, 'train_cache/' + str(c + 1) + '/' + str(600 + n + 1) + '.bmp'),
    #             (28 * 28))
    #         test_outputs[count] = c
    #         count += 1

    test_inputs = np.empty([1, 28*28])
    test_outputs = np.empty([1])
    img = Image.open('train/IMG_0855.png')
    test_inputs[0] = np.resize(
        convert_to_bw(img, 'train_cache/IMG_0855.png'),
        (28 * 28))
    print(test_inputs[0])
    test_outputs[0] = 0

    # 准备网络参数、开始训练
    INPUT_NUM = 28 * 28
    OUTPUT_NUM = 12

    if path == '':
        # 随机生成
        hlw = []
        olw = []
        for i in range(HIDDEN_UNIT_NUM * INPUT_NUM):
            hlw.append((random.random() * 2 - 1) / 100)
        for j in range(HIDDEN_UNIT_NUM * OUTPUT_NUM):
            olw.append((random.random() * 2 - 1) / 100)

        hlb = -random.random() / 1
        olb = -random.random() / 1

        nn = BPNetwork(INPUT_NUM, [HIDDEN_UNIT_NUM], OUTPUT_NUM, hidden_layers_weights=[hlw], hidden_layers_bias=[hlb],
                       output_layer_weights=olw, output_layer_bias=olb, softmax_enabled=1)
    else:
        file1 = open(path, 'rb')
        nn = pickle.load(file1)

    # TIMES = 15
    # for i in range(TIMES):
    #     for c in range(12 * 600):
    #         nn.train(randomized_training_inputs[c], randomized_training_outputs[c])
    #         print(str(i) + ": " + str(np.floor(c / 600)) + "/" + str(c % 600))
    #
    # # 持久化
    # file = open('data/h_nwb_' + str(15) + '_' + str(time.time()) + '.data', 'wb')
    # pickle.dump(nn, file)
    # file.close()

    # 计算正确率
    # success = 0
    # for i in range(12 * 20):
    #     index = np.random.randint(12 * 20)
    #     output_list = np.array(nn.softmax_feed_forward(test_inputs[index])).tolist()
    #     print(output_list)
    #     max_index = output_list.index(max(output_list))
    #     print(max_index)
    #     if max_index == test_outputs[index]:
    #         success += 1
    # print(success)
    # result = success / (12 * 20)
    # print(result)

    output_list = np.array(nn.softmax_feed_forward(test_inputs[0])).tolist()
    print(output_list)
    max_index = output_list.index(max(output_list))
    print(max_index)

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


sinx('data/s_10000000_1604149114.362782.data')
