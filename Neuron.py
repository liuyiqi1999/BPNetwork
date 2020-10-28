import math

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    # 计算总输出
    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.activation(self.calculate_total_net_input())
        return self.output

    # 去除 sigmoid 的输出
    def calculate_test_output(self, inputs):
        self.inputs = inputs
        self.output = self.calculate_total_net_input()
        return self.output

    # 计算线性输出
    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # 激活函数
    def activation(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # ∂E/∂i = ∂E/∂o * ∂o/∂i
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input()

    # 没有 sigmoid 函数
    def calculate_test_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output)

    # 计算 error（平方差）
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # ∂E/∂o
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # ∂o/∂i
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    # ∂i/∂w
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]
