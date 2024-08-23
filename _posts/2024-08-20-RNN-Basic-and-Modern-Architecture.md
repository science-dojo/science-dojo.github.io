---
title: RNN Basic and Modern Architecture  
tags:
  - CNN
layout: single
toc: true
toc_sticky: true
toc_label: "目录"
---

在分类、回归等任务中（eg. 点击率预估、图像类别预估），给定特征$X$后，预估固定的目标$y$，例如在点击率预估任务中，$y$有两种选择：点击、不点击。
而现实中，我们经常有需要预估一个结构化的目标的需求，例如给定一个图片，自动描述其内容；给定一句中文，自动翻译出其英文表达；给定一段音频，自动识别
出不同语言的文字表达。这种我们通常称**sequence-to-sequence**任务，其有两种形式。其一，对齐(aligned)，每个时间步的输入对应特定的目标(e.g., part of speach tagging)；其二，非对齐(unaligned)，每个时间步的输入和输出不是完全对齐(e.g. 语言翻译)。接下来，我们将探究处理这类问题的思路、方法和工具。

# Autoregressive Model

在非结构化的预估任务中（e.g. 点击率预估），通常假设输入$X$是从某个确定未知的分布$P(X)$**独立同分布**采样出来。在结构化预估任务中，
仍然可以假设全部序列（e.g. 全部文档集）独立同分布的从某确定未知的分布中采样出来，但是我们无法假设每个时间步的数据都是相互独立的。
例如，一个文档中单词，后出现的单词通常更依赖于其前面的单词，并且时间步越近，受影响越大。因此，大多数的序列模型都不会假设每个时间步的数据是相互独立的，
只需要假设这些序列数据是从一个固定分布中独立同分布采样的。因此，引出了一个最直接的问题 --> unsupervised density model (or sequence modeling): 

> 给定一个序列的集合，预估指定序列$x_1,x_2,...,x_T$出现的概率，如何找出这个概率质量函数 $P$？


我们先考虑一个简单的问题: 如何预估$x_t$，假设我们只有一个序列集和，没有其他任何数据，能影响$x_t$的，只有时间步$t$前的数据$x_{t-1}, ..., x_1$,
因此问题建模为计算条件概率分布:

$$P(x_t | x_{t-1},...,x_1)$$

虽然预估连续值随机变量的整个分布可能很困难，但预估出一个变量的值后，其他变量也可以类似预估出来。
这种根据同一个信号之前时刻取值预估当前时刻取值的模型，称为**自回归模型(Autoregressive model)**。

实际处理时，通常只会选取时刻`T`之前$\tau$个时刻的值作为条件，每次预估的参数个数都是相同的，实际操作中方便处理输入；
此外，模型可以维护一个表示过去状态的隐藏变量$h_t$，预估$\hat{x_t}$时, $h_t$作为输入: $\hat{x_t} = P(x_t|h_t)$，
同时更新用$h_{t-1}, x_{t-1}$ 更新 $h_t = g(h_{t-1}, x_{t-1})$。计算$Loss(x_t, \hat{x_t})$，回传错误信号，更新模型。
由于$h_t$不是可见变量，这类模型也称为**隐状态自回归模型(latent autoregressive model)**。

# Sequence Model

如何计算概率质量函数$P$预估序列出现的概率？应用*chain rule of probability*，将序列模型（语言模型）建模分解为自回归预估的乘积：

$$
P(x_1,...,x_T)=P(x_1) \prod_{t=2}^{T}P(x_t|x_{t-1},...,x_1)
$$

因此序列模型完成了*预估下一个词* 和 *预估序列出现概率*的双重任务。


## Markov Models


Markov condition
: 序列在时间步$x_t$只依赖前$\tau$步的取值，即：去掉$\tau$步之前的时间步，而不影响预估能力。


Markov model
: 满足Markov condition的模型


$\tau =1$时, *first-order Markov model*, $\tau =k$时, *$k^{th}-order Markov model$*

$\tao$越大，效果越好；随着$\tao$逐渐增大，效果增益快速降低，计算量增大。实际使用时，根据计算资源，灵活选择取值。
# Measure languate model

 Entroy
: 给定一个分布$P(x)$，其 entropy 定义为 $H[P] = \sum_{j}^{n}-P(j)logP(j)$
  $-Log(P(j))$用于量化观测到事件$j$时的惊讶程度，或者传输信息需要的比特数，其被称为 **Information content**。概率为1时, 信息量为0，概率越低，信息量越大，因为当一个低概率的事件发生时，通常能传达重要的信息。

Cross Entropy
: 

Perpexity
:


# Recurrent Neural Networks 


