import random
import numpy as np

from Neuron import Neuron


class NeuronLayer:
    def __init__(self, neurons_num, bias):
        self.bias = bias if bias else random.random()
        self.neurons = []
        for i in range(neurons_num):
            self.neurons.append(Neuron(self.bias))

    # 输出层信息
    def inspect(self):
        print('Neurons: ', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print(' Weight:', self.neurons[n].weights[w])
            print(' Bias:', self.bias)

    # 计算对下一层的输出
    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    # 获得对下一层的输出
    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

    def output_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_test_output(inputs))
        return outputs

    def softmax_output_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_test_output(inputs))
        return self.softmax(outputs)

    def get_softmax_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return self.softmax(outputs)

    def softmax(self, outputs):
        total = 0
        for o in outputs:
            total += np.exp(o)
        result = [0] * len(outputs)
        for o in range(len(outputs)):
            result[o] = np.exp(outputs[o]) / total
        return result

    # ∂E_cross/∂o
    def calculate_pd_cross_error_wrt_softmax_output(self, target_output):
        pd_cross_error_wrt_output = 0
        softmax_outputs = self.get_softmax_outputs()
        for i in range(len(self.neurons)):
            pd_cross_error_wrt_output -= target_output[i] / softmax_outputs[i]
        return pd_cross_error_wrt_output

    # ∂Softmax_output/∂i
    def calculate_pd_cross_error_wrt_total_net_input(self, pd_cross_error_wrt_softmax_output_array):
        softmax_outputs = self.get_softmax_outputs()
        for j in range(len(pd_cross_error_wrt_softmax_output_array)):
            for i in range(len(self.neurons)):
                if i == j:
                    # ∂Softmax_output/∂i = Si(1-Si)
                    pd_cross_error_wrt_softmax_output_array[j] *= softmax_outputs[j] * (1 - softmax_outputs[j])
                else:
                    pd_cross_error_wrt_softmax_output_array[j] *= -softmax_outputs[j] * softmax_outputs[i]
        return pd_cross_error_wrt_softmax_output_array
