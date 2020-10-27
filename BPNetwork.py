import random

from NeuronLayer import NeuronLayer
import numpy as np


class BPNetwork:
    LEARNING_RATE = 0.01

    def __init__(self, inputs_num, hidden_num, outputs_num, hidden_layer_weights=None, hidden_layer_bias=None,
                 output_layer_weights=None, output_layer_bias=None):
        self.inputs_num = inputs_num

        self.hidden_layer = NeuronLayer(hidden_num, hidden_layer_bias)
        self.output_layer = NeuronLayer(outputs_num, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    # 初始化输入层到隐含层的 weight
    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.inputs_num):
                self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

        # weight_num = 0
        # for h in range(len(self.hidden_layer.neurons)):
        #     weights_list = np.empty([self.inputs_num], dtype=int)
        #     for i in np.nditer(weights_list, op_flags=['readwrite']):
        #         i[...] = hidden_layer_weights[weight_num]
        #         weight_num += 1
        #     self.hidden_layer.neurons[h].weights = np.array(weights_list)

    # 初始化隐含层到输出层的 weight
    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        # weight_num = 0
        # for o in range(len(self.output_layer.neurons)):
        #     weights_list = np.empty([len(self.hidden_layer.neurons)], dtype=int)
        #     for h in np.nditer(weights_list, op_flags=['readwrite']):
        #         h[...] = output_layer_weights[weight_num]
        #         weight_num += 1
        #     self.output_layer.neurons[o].weights = np.array(weights_list)

        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    # 输出网络信息
    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.inputs_num))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs) # TODO：可伸缩的隐含层
        return self.output_layer.output_forward(hidden_layer_outputs)

    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 正向传播：获得输出层的值（输入 -> 输出）
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            # ∂E/∂iⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[
                o].calculate_test_pd_error_wrt_total_net_input(training_outputs[o])

        # 反向传播：获得隐含层的值（输出 -> 隐含）
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # dE/dhⱼ = Σ ∂E/∂oⱼ * ∂o/∂hⱼ = Σ ∂E/∂oⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * \
                                                    self.output_layer.neurons[o].weights[h]

            # ∂E/∂oⱼ = ∂E/∂hⱼ * ∂hⱼ/∂oⱼ
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * \
                                                             self.hidden_layer.neurons[
                                                                 h].calculate_pd_total_net_input_wrt_input()

        # 更新输出层 weight，bias
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # weight: ∂Eⱼ/∂wᵢⱼ = ∂E/∂oⱼ * ∂oⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[
                    o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # weight: Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

            # bias: ∂Eⱼ/∂bᵢⱼ = ∂E/∂oⱼ * ∂oⱼ/∂bᵢⱼ = ∂E/∂oⱼ * 1
            pd_error_wrt_output_layer_bias = pd_errors_wrt_output_neuron_total_net_input[o] # TODO: 隐含层可伸缩后，bias 的求导公式需要重新计算

            # bias: Δb = α * ∂Eⱼ/∂bᵢⱼ
            self.output_layer.bias -= self.LEARNING_RATE * pd_error_wrt_output_layer_bias

        # 更新隐含层 weight，bias
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                # weight: ∂Eⱼ/∂wᵢ = ∂E/∂hⱼ * ∂hⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[
                    h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # weight: Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

            # bias: ∂Eⱼ/∂bᵢⱼ = ∂E/∂hⱼ * ∂hⱼ/∂bᵢⱼ = ∂E/∂hⱼ * 1
            pd_error_wrt_hidden_layer_bias = pd_errors_wrt_hidden_neuron_total_net_input[h]

            # bias: Δb = α * ∂Eⱼ/∂bᵢⱼ
            self.output_layer.bias -= self.LEARNING_RATE * pd_error_wrt_hidden_layer_bias

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error
