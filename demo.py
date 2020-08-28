# -*- coding: utf-8 -*-
# @Author: Xia Hanyu
# @Date:   2020-08-26 21:58:22
# @Last Modified by:   Xia Hanyu
# @Last Modified time: 2020-08-28 00:04:07

from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci

# 隐藏层100个神经元，学习率0.1
network = NeuralNetwork(28 * 28, 100, 10, 0.1)

# 使用mnist_train.csv 前1000条数据进行训练
train_file = open('mnist_train.csv', 'r')
for i in range(1000):
    data = train_file.readline().split(',')
    # 标准输出认为0.01为未激活，0.99为完全激活，因为sigmoid(x)范围在(0, 1)
    targets = np.zeros(10) + 0.01
    targets[int(data[0])] = 0.99

    # 输入数据范围控制在(0, 1)
    inputs = np.asfarray(data[1:]) / 255.0 * 0.99 + 0.01

    network.train(inputs, targets)
train_file.close()
print("train done")

# 使用mnist_test.csv 第一条数据测试
test_file = open('mnist_test.csv', 'r')
data = test_file.readline().split(',')
test_file.close()
result = network.query(np.asfarray(data[1:]) / 255.0 * 0.99 + 0.01)
print(result) # 输出层得到数组
answer = np.argmax(result)
print("识别结果：", answer)

# 读取其他图片形式（PNG JPG）的手写数字
# img_array = sci.misc.imread('file_name', flatten=True)
# img_data = 255.0 - img_array.reshape(28 * 28)
# img_data = img_data / 255.0 * 0.99 + 0.01

# 显示图片
image_array = np.asfarray(data[1:]).reshape(28, 28)
plt.imshow(image_array, cmap="Greys")
plt.show()
plt.close()