---
layout: post
author: Erik
---

Sources
- [ml-cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
- [Visual Information](https://colah.github.io/posts/2015-09-Visual-Information)
- [Blogpost on loss functions](https://rohanvarma.me/Loss-Functions/)
- [Cross Entropy](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/)
- [BCE](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
- [PyTorch](https://pytorch.org/docs/stable/nn.html#loss-functions)

# General

## Entropy, H

\\[ H(y) = - \sum_{i} y_i \log \frac{1}{y_i} = \sum_{i} y_i \log y_i  \\]


The minimum , theoretical lower limit, cost to send information (maximum amount of information per bit). Using encoding based on prior knowledge about the source.



## Cross-Entropy, CE


Maximizing the the log-likelihood is the same as minimizing the cross entropy


\\[ H_{y} (\hat{y}) = \sum_{i} \hat{y_i} \log \frac{1}{y_i} = \sum_{i} \hat{y}_i \log y_i \\]

\\[ H_{\hat{y}} (y) = \sum_{i} y_i \log \frac{1}{\hat{y_i}} = \sum_{i} y_i \log \hat{y_i} \\]


## KLD

\\[ D_{KL}(y || \hat{y}) = D_{ \hat{y} }(y) = H_{\hat{y}}(y) - H(y) \\]


## Binary-Cross-Entropy, BCE

\\[ L(y, \hat{y} ) = BCE(x, y) = \frac{1}{N} \sum_{n = 0}^{N} \\]



# Loss Functions (Pytorch)

Using PyTorch as the deep learning framework so lets investigate what kinds of
loss-functions are available by default.


### L1-Loss

The L1-loss uses the absolute difference (L1-norm) for each of the \((y_{pred}, y\))
pairs.


\\[ L(y_{pred}, y) = L \{l_1, ..., l_n\}^T, l_n = |y_{pred, n} - y_n|  \\]

In PyTorch you have the ability to then choose what type of reduction scheme to use.
This reduction is either 'none', 'mean' or 'sum' where the 'mean' is the default.


Mean:

\\[ L_{mean}(y_{pred}, y) = \frac{1}{N}\sum_{n=0}^N l_n \\]


Sum:

\\[ L_{sum}(y_{pred}, y)  = \sum_{n=0}^N l_n \\]


None:

Returns the tensor with the absolute difference for each point.


### Mean Square Eroor Loss (L2-Loss)

The mean square error loss function is one of the most common in deep learning and uses
the error (distance between true and predicted) squared. Again we may choose what type
of reduction schema to use.


\\[ L(y_{pred}, y) = L \{l_1, ..., l_n\}^T, l_n = (y_{pred, n} - y_n)^2  \\]

Mean:

\\[ L_{mean}(y_{pred}, y) = \frac{1}{N}\sum_{n=0}^N l_n \\]


Sum:

\\[ L_{sum}(y_{pred}, y) = \sum_{n=0}^N l_n \\]


None:

Returns the tensor with the absolute difference for each point.


### Binary Cross Entropy, BCELoss

Measures the binary cross entropy between the target and output. Used in training
autoencoders as the reconstruction error. The targets and the output are required to be
between 0 and 1. 

Notice the term with \\( \log(y_{pred}) \\) here we see that we need our output to be
in the domain of log. This means that the output must not be of zero or negative
values. However,  we also have the term with \\( \log(1 - y_{pred}) \\) and for this to
be in the domain of log the output cannot be equal to or larger than 1.

Thus the most common thing to do is to use the Sigmoid activation for the output.

\\[ L(y_{pred}, y) = L \{l_1, ..., l_n\}^T, l_n = -w_n [ y_n * \log(y_{pred, n}) + (1-y_n) * \log(1 - x_n ) ] \\]


Mean:

\\[ L_{mean}(y_{pred}, y) = \frac{1}{N}\sum_{n=0}^N l_n \\]


Sum:

\\[ L_{sum}(y_{pred}, y) = \sum_{n=0}^N l_n \\]


### BCEWithLogitsLoss

Combines the Sigmoid Layer of the output and the BCELoss function in a numerically  stable manner.



## What are the differences?


