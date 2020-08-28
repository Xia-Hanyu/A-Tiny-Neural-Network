# <center>简易神经网络</center>

此文档 Latex 公式编辑依赖于 Visual Studio Code 插件 Markdown Preview Enhanced，无法直接在 github 中显示，文档请参阅 README.pdf

## 简介

可以识别 0-9 手写字体的简易神经网络
读取$m \times n$像素的图片文件，本神经网络采用$m = n = 28$，你自然可以修改
提供训练集`mnist_train.csv`，拥有 60000 条数据，每条数据第一个为标签，其余 28 \* 28 = 784 个数字为每点像素值
提供测试集`mnist_test.csv`，格式与训练集相同

http://www.pjreddie.com/media/files/mnist_train.csv
http://www.pjreddie.com/media/files/mnist_test.csv

你也可以自己利用图片组成测试数据

## 原理

#### 概述

神经网络由输入层、隐藏层和输出层组成，隐藏层仅有一层，规模为 784 \* m(隐藏层神经元数量) \* 10，输出是大小为 10 的数组，代表各个数字为答案的概率，取概率最高者为最终识别结果

输入层与隐藏层、隐藏层与输出层之间链接由权重矩阵 $W_{ih(784 \times m)}, W_{ho(m \times 10)} $ 构成。输入为 784 个元素的向量 $X_{in}$，代表 784 个像素值

$$
W_{mn} = \begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1n} \\
\vdots & \vdots & \ddots & \vdots \\
w_{m1} & w_{m2} & \cdots & w_{mn}
\end{bmatrix}
$$

#### 激活函数

除输入层外，每层神经元激活函数选用`sigmoid()`，即
$$sigmoid(x) = \frac{1}{1 + e^{-x}} $$

$sigmoid(x) \in (0, 1) $，因此保持输入值在 $(0, 1)$为佳

每层输入、输出关系为

$$X_{out} = sigmoid(X_{in}) $$

#### 信号传递

$X$层、$Y$层之间信号传递表示为
$$Y_{in} = W_{XY}^TX_{out} $$

对于此简易网络（$I$层、$H$层、$O$层），可以完整表示为
$$O_{out} = sigmoid(W_{ho}^Tsigmoid(W_{ih}^TI_{out})) $$
$$I_{in} = I_{out} $$

#### 误差

对于输出层，误差即为与标签的差值，对于隐藏层，误差为输出层误差根据$W_{ho}$权重按比例分配

$$
e_{hidden, i} = \frac{\sum_k w_{ik}e_{out,k}}{\sum_k w_{ik}}
$$

同一个神经元计算式中分母均相同，为了显式按比例分配，分母可以不要，使得表示、计算简便，因而有

$$
e_{hidden, i} = \sum_k w_{ik}e_{out,k}
$$

矩阵表示为

$$
E_{hidden} = W_{ho}E_{out}
$$

#### 权重修正

权重初始采用随机值，通过训练修正

采用梯度下降方法对$i$神经元到$j$神经元链接权重$w_{ij}$更新。设学习率为 $\alpha$，误差函数为 $e_j$，则有
$$w_{ij} = w_{ij} - \alpha \frac{\partial e_j}{\partial w_{ij}} $$
其中
$$e_j = (target_j - output_j)^2$$

得到

$$
\frac{\partial e_j}{\partial w_{ij}} = \frac{\partial e_j}{\partial output_j} \cdot \frac{\partial output_j}{\partial w_{ij}} = -2(target_j - output_j) \cdot \frac{\partial sigmoid(\sum_k w_{kj} \cdot output_k)}{\partial w_{ij}}
$$

又有

$$
\frac{\partial sigmoid(x)}{\partial x} = sigmoid(x) \cdot (1 - sidmoid(x))
$$

$$
sigmoid(\sum_k w_{kj} \cdot output_k) = sigmoid(input_j) = output_j
$$

$$
\frac{\partial \sum_k w_{kj} \cdot output_k}{\partial w_{ij}} = output_i
$$

得到

$$
\frac{\partial e_j}{\partial w_{ij}} = -2(target_j - output_j) \cdot sigmoid(\sum_k w_{kj} \cdot output_k) \cdot (1 - sigmoid(\sum_k w_{kj} \cdot output_k)) \cdot output_i
$$

$$
= -2(target_j - output_j) \cdot output_j \cdot (1 - output_j) \cdot output_i
$$

写为矩阵形式，$E_j = Target_j - Output_j$为误差向量，舍弃系数 2，得到权重更新方程

$$
W_{ij} = W_{ij} + \alpha \cdot E_j \cdot Output_j(I - Output_j) \cdot Output_i^T
$$

其中

$$
I = \begin{bmatrix}
1 \\
1 \\
\vdots \\
1
\end{bmatrix}
$$

## 结果

隐藏层神经元数量设为 100，学习率 0.1，采用 1000 条数据训练的神经网络识别准确率可达到 95%以上
