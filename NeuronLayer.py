import random

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