---
title: BatchNorm and LayerNorm
tags:
  - CNN
layout: single
toc: true
toc_sticky: true
toc_label: "目录"
---

训练深度的神经网络在指定的时间内收敛是困难的。BatchNorm和LayerNorm是加速深度网络收敛的正则化技术，广泛应用于现代的视觉和自然语言处理领域。
接下来我们将介绍其细节、区别、使用位置以及应用场景。

# BatchNorm

变量说明: 
- $\mathcal{B}$ minibatch
- $\mathbf{x} \in \mathcal{B}$ input
- $\textrm{BN}$ 代表batchnorm 
- $\epsilon > 0$ 保证不会除0
- $\gamma, \beta$ 分别是scale参数和shift参数，均是可学习参数，其shape和$\mathbf{x}$一致，其目的是恢复标准化后的自由度。


$$\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$


$$\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ and }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.$$


显然，应用BatchNorm后，在训练中中的网络权重大小不会发散(diverge)，因为BatchNorm会动态缩放标准化他们。


经验表明，不同来源的噪声能够加速训练、减少过拟合问题。BatchNorm中对 均值和方差 的估计，看起来是引入了某种形式的正则化
，[Teye](https://arxiv.org/abs/1802.06455)和[Luo](https://arxiv.org/abs/1809.00846)将BatchNorm的性质分别和贝叶斯先验和惩罚关联，他们解释了为什么BatchNorm在50~100大小的minibatch上表现最好。
特定大小的minibatch像是在每层注入了某种*正确*的噪声一样：较大的minibatch由于更稳定的估计正则化较少，较小的minibatch由于高方差而破坏有用的信号。

在训练过程中，网络权重随着我们更新模型会动态变化，因此**训练时，只能基于minibatch计算均值和方差**；
训练完成后，进入预估阶段，模型权重不再变化，因此**预估时，能够基于整个数据集计算均值和方差**。 这种表现和Dropout正则非常像。

**一句话总结**BatchNorm作用：正则化，提升训练稳定性。

但如果想让模型对输入扰动不敏感，则可以考虑去掉BatchNorm。 (Wang et al. 2022)[https://arxiv.org/abs/2207.01156]

### BatchNorm用在全连接网络层

用于Linear层之后，激活层之前

$$\mathbf{h} = \phi(\textrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

### BatchNorm用在卷积层

- 用于卷积层之后，激活层之前
- 每个通道都会计算单独的均值和方差, 因此每个通道都会在单独$B \times h \times w$个元素上做BatchNorm，并且有自己单独的 scale 和 shift 参数（标量）

当BatchNorm用在卷积层的时候，minibatch即使为1，也不会有问题,因为 $\frac{x-\hat{\mu}}{\sigma} \neq 0$
因此，卷积层LayerNorm可以用作BatchNorm的替代。


```python
# 每一层，都需要保存 gamma, beta, moving_mean, moving_var参数
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data

  class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
```



### BatchNorm 和 Dropout在网络中的顺序

参考论文[Understand the Disharmony between Dropout and BatchNorm by Variance Shift](https://arxiv.org/pdf/1801.05134) 和[stackoverflow](https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout)
BatchNorm和Dropout的使用顺序为：

**Conv -> BatchNorm -> Activation -> DropOut -> Pool**
{: .notice--info}


## LayerNorm


BatchNorm计算minibatch中的所有样本的均值和方差，LayerNorm只计算单个样本对应向量的均值和方差

$$\mathbf{x} \rightarrow \textrm{LN}(\mathbf{x}) =  \frac{\mathbf{x} - \hat{\mu}}{\hat\sigma},$$

$$\hat{\mu} \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n x_i \textrm{ and }
\hat{\sigma}^2 \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2 + \epsilon.$$

LayerNorm可以防止权重发散（divergence），因为网络层的输出和尺度无关。也就是说，$\alpha \neq 0$时， $\textrm{LN(x)} \approx \textrm{LN}(\alpha \mathbf{x})$
当$|\alpha| \to \infty$ 时，等号成立

此外，LayrNorm和minibatch以及训练模式无关，同时也能够防止权重发散。
