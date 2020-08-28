# -*- coding: utf-8 -*-
# @Author: Xia Hanyu
# @Date:   2020-08-24 19:40:12
# @Last Modified by:   Xia Hanyu
# @Last Modified time: 2020-08-28 00:04:15

import scipy.special as sci
import numpy as np

class NeuralNetwork:
    """
    一个简易神经网络，可以识别手写数字
    仅有输入层、输出层和一层隐藏层
    """

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_node = input_nodes
        self.hidden_node = hidden_nodes
        self.output_node = output_nodes
        self.learning_rate = learning_rate
        
        # 链接权重矩阵，采用随机正态分布，均值为0，方差为1 / squareroot(下一层节点个数)
        self.W_ih = np.random.normal(
            0.0, pow(self.hidden_node, -0.5), (self.input_node, self.hidden_node))
        self.W_ho = np.random.normal(
            0.0, pow(self.output_node, -0.5), (self.hidden_node, self.output_node))
        
    def activateFunc(self, x):
        # 激活函数选用sigmoid
        return sci.expit(x)
    
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin = 2).T

        hidden_inputs = np.dot(self.W_ih.T, inputs)
        hidden_outputs = self.activateFunc(hidden_inputs)

        final_inputs = np.dot(self.W_ho.T, hidden_outputs)
        final_outputs = self.activateFunc(final_inputs)

        # 输出误差
        output_errors = targets - final_outputs
        # 隐藏层误差
        hidden_errors = np.dot(self.W_ho, output_errors)
        # 权重矩阵修正
        self.W_ho += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_errors)).T
        self.W_ih += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs)).T

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.W_ih.T, inputs)
        hidden_outputs = self.activateFunc(hidden_inputs)

        final_inputs = np.dot(self.W_ho.T, hidden_outputs)
        final_outputs = self.activateFunc(final_inputs)

        return final_outputs