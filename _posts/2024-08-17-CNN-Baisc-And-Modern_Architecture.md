---
title: CNN Basics And Modern Architecture
tags:
  - CNN
layout: single
toc: true
toc_label: "目录"
---

卷积神经网络（CNN）是深度学习发展过程中不可忽视的重要结构之一，在图像处理领域，是以深度学习为基础的方法的基础结构。
本文意在梳理CNN的基本原理和现代演进结构，巩固基础，展望未来。

# 从全连接网络(FCN)到卷积神经网络(CNN)
全连接神经网路（Fully connected network）没有对特征之间如何交互有任何假设，每一个神经元都与其他所有神经元有连接。
当处理表格类型的数据时（e.g. 图像），一个$$10^6$$像素的图像，即使做Reduction操作，和1000维的隐藏层做全连接，需要$$10^6 \times 10^e = 10^9$$个参数，当网络更加复杂时，数据量更大时，参数量将会剧烈增加，训练网络需要更多的资源、耐心和技巧。

图像数据本身包含丰富的结构信息，当人类区分图片中的动物是猫或狗时，不会逐个检查每个像素之间的关系，而是利用图像块之间的关系和信息来识别，相邻的像素包含相似的语义信息（Locality），图像语义块之间的关系不会变换(Translation Invariance)。
卷积神经网络(Convolution neural network)，系统化的利用Locality和Invariance，可以使用更少的参数学习到有用的信息：

- 在浅层网络中，网路应该关注局部区域(Locality)，不用关系其他区域的内容；对于同一个区域，不论其处在图片中的哪个位置，网络应该要作出相似的反应(Translation Invariance/Equivariance)。
- 深层网络，汇聚浅层网络的信息，捕捉图片中长距离的特征间的关系。

## 改造MLP为CNN

MLP关注每个像素和其他所有像素的关系，每个像素和其他所有像素之间都存在权重。考虑二维输入图片 $$X$$ 和 $$H$$ 二维隐藏层权重 $$H_{i,j}$$, $$X_{i,j}$$分别图片和隐藏层表示位置`(i,j)`处的像素, $$U_{i,j}$$代表bias：

$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned}$$

在第二行，对公式做改写，并对下标重新索引，令$$k=i+1, l=j+b$$，因此 $$\mathbf{V}_{i,j,a,b}=\mathbf{W}_{i,j,i+a,j+b}$$，通过改变`a`和`b`的范围，就能覆盖整张图像，做到全连接。

**Notice:** $$\mathbf{W}$$在MLP中一般表示为二维权重，此处用四维权重是为了表达让隐藏层的每一个神经元都能接受到来自输入图像的每一个像素，因此每一个隐藏层的神经元都有一个二维权重矩阵，神经元之间的权重矩阵不共享。一个 `1000 x 1000`的输入图像映射到 `1000 x 1000`的隐藏层，需要 $$10^{12}$$的参数量。
{: .notice}


### 利用Translation Invariance
平移不变性暗示平移输入`X`，在隐藏层`H`中也会有相应的平移。即：在`U`和`V`都不依赖于位置`(i,j)`时, 才平移不变性才可能发生。因此，让$$\mathbf{V}_{i,j,a,b}=\mathbf{V}_{a,b}$$, $$U$$是常量`u`,因此`H`可简化为：

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$

这就是卷积！权重`V`不再依赖图像中的位置变量`(i,j)`，所需参数量 $$4 \times 10^6$$。(**why 4?**{: style="color: red"})。 此时 `a` 和 `b`的取值范围是 `[-1000, 1000]`，接下来需要利用局部性原理限制`a`和`b`的取值范围。
### 利用Locality

局部性原理表明，隐藏层中位置`(i,j)`出的值之和输入中`(i,j)`的邻域有关系，因此`a`和`b`的取值需要限制在一定的范围内，超过范围，$$\mathbf{V}_{a,b}=0$$。

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$


参数量为$$4 \times \Delta^2$$，$$\Delta$$通常小于10。$$V$$被叫做可学习的**卷积核**，**滤波器**或者权重。至此，参数量从$$10^12$$降低到$$4\Delta^2$$，这是一个巨大的下降。

### 卷积通道

基于MLP改造的卷积层，输入和输出都只有一个通道，实际上，图片有长、宽、通道数量三种维度，为了表达更多维度，隐藏层也可以拥有很多通道维度，来捕捉通道之间的特征。因此卷积核$$\mathbf{V}_{a,b}$$需要表示为$$\mathbf{V}_{a,b,c,d}$$，`c,d`分别表示输入和输出通道下标。

$$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$


## 实现卷积

**Notice**：上文描述的操作严格来说不是卷积，而是互相关性，本质上他们的计算结果是相同的，遵从深度学习的术语，我们仍称其为卷积。详见[Cross-Correlation and Convolution](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#cross-correlation-and-convolution)
{: .notice}

忽略通道量，在二维图像$$(n_h \times n_w)$$上卷积，卷积核 $$(k_h \times k_w)$$，输出维度为$$ (n_h-k_h+1) \times (n_w-k_w+1) $$

![二维卷积图示](/assets/images/deep_learning/2d_conv_op.png){: .align-center}


$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

```python
# Pytorch implementation
def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# Conv2d Layer
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```


- 特征图(Feature Map): 卷积层的输出的通常称谓。
- 感受野(Receptive Field): 影响某层`l`中的神经元`x`的输出值的前`l-1`层的所有元素。


## Padding 和 Stride

使用卷积时，图像边缘的像素信息会一定程度丢失，卷积核越大，图片边缘的信息丢失越严重。如下图所示，卷积核为`1x1, 2x2, 3x3`时，图片边缘的像素点的使用次数。单次卷积虽然损失不大，但随着网络的加深，累计效应会放大。


<figure class="align-center">
  <img src="/assets/images/deep_learning/conv_pixel_util.png" alt="">
</figure>


- 直接的解决方法：输入四周补充像素点，补充的像素点设置为0。例如下图

![pading样例](/assets/images/deep_learning/conv_pad.png){: .align-center}


增加padding后，卷积输出shape变为：

$$(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+1).$$

在许多应用中，为了让输入输出shape一致，$$p_h = k_h - 1, p_w = k_w - 1$$，这样更容易预估输出的shape。当 $$k_h$$为奇数时，我们将会在两边添加$$\frac{p_h}{2}$$列，当 $$k_h$$为偶数时，一种pad方式是在左边增加$\lceil p_\textrm{h}/2\rceil$，右边增加$\lfloor p_\textrm{h}/2\rfloor$列。

通常，CNN使用高和宽为奇数的卷积核，这样可以让pad的行数或者列数在图片上下左右相同，保持维度大小、比例不变。


执行卷积操作时，卷积核每次移动一步。我们也可以移动多步，以增加计算效率或者想做下采样。当卷积核很大的时候，可以捕捉更大的区域，此时，我们可以适当增加移动步长。

![conv_stride](/assets/images/deep_learning/conv_stride.png){: .align-center}

strides of 3 and 2 for height and width
{: style="text-align: center;"}


当stride $(s_h,s_w)$被设置后，卷积的输出shape为：
$$\lfloor(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+s_\textrm{h})/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+s_\textrm{w})/s_\textrm{w}\rfloor.$$

当 $p_h = k_h - 1, p_w = k_w - 1$时，卷积输出shape为：

$$\lfloor(n_\textrm{h}+s_\textrm{h}-1)/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}+s_\textrm{w}-1)/s_\textrm{w}\rfloor$$

当输入的宽和高可以被stride整除，卷积输出shape为：

$$(\frac{n_h}{s_h} \times \frac{n_w}{s_w})$$


## 多通道输入输出卷积实现


### 多通道输入-单通道输出卷积

![multi_input_channel](/assets/images/deep_learning/conv_multi_ic.png){: .align-center}

两个输入通道的卷积计算，卷积核也同样具有两个通道，分别对应输入的两个通道，每个通道卷积的结果相加，作为最终结果, e.g. $(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$.

```python
# Pytorch Implementation
def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return sum(corr2d(x, k) for x, k in zip(X, K))
```

### 多通道输入-多通道输出卷积

$c_i, c_o$分别为输入输出的通道数量，卷积核的shape为 $c_o \times c_i \times k_h \times k_w$。卷积结果是`concat`起来每个输出通道对应的卷积核和*所有*的输入通道做卷积。

```python
# Pytorch implementation
def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of K, and each time, perform
    # cross-correlation operations with input X. All of the results are
    # stacked together
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
```

### $1 \times 1$卷积核

$1 \times 1$卷积核在单通道卷积中没有作用，它不能让输出神经元看到输入的相邻区域。在多通道输入中，$1\times1$卷积可以在`channel`维度做计算，相当于在channel维度做全连接，因此它需要的参数量为$c_i \times c_o$

![conv_1x1](/assets/images/deep_learning/conv_1x1.png){: .align-center}

三输入channel, 两输出channel，$1\times1$卷积图
{: style="text-align:center;"}


```python
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # Matrix multiplication in the fully connected layer
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6 # True, Y1.shape == Y2.shape
```


## 池化


池化层没有参数，它主要两个作用：
- 减缓卷积对局部的敏感, 网络越深，隐藏层元素的感受野越大，就更敏感
- 空间下采样，降低空间分辨率，加速计算


常用两种池化层：
- 平均池化(Average Pooling)：将局部区域的均值作为输出
- 最大池化(Maximum Pooling)：将局部区域的最大值最为输出，输出越大特征越强越能表征该局部区域的重要性，大多数case中，最大池化效果更好


```python
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```


**对多channel输入，池化对每个channel分别生效**，而不是沿着channel维度加起来。

池化层和卷积核一样，具有pad和stride参数，含义同卷积核一致。



# 参考

- [Diving into Deep Learning](https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html)
