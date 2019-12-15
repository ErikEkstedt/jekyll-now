---
layout: default
title: Glow
---

## Glow: Generative Flow with Invertible 1x1 Convolutions

Invertible flows are useful because they allow for a model to directly maximize the
log-likelihood instead of doing variational inference. 
[Paper](https://arxiv.org/pdf/1807.03039.pdf)

<!--more-->

Glow is an architecture decomposed into **two** parts. The **high-level multi-scale
architecture** that contains **L** layers. Each of these layers consists of **K** **steps
of flow**. In this architecture a step of flow is defined as 3 sequential layers.

<img height="300" src="/assets/images/Glow/Glow-overall-architecture.png" alt="" class="centerImg"/>

<!--excerpt-->

#### ActNorm

A normalization layer closely related to batchnorm which is
inititalized to normalize the post-activation of this layer to zero mean and
unit variance (mean: 0, variance: 1). The layer accomplishes this by a scaling
and shifting operation (affine transformation). After the inititalization the
weights used for the transformation are regarded as standard parameters of the
network and is part of the computational graph and optimization process.

#### Permutation

The second step in the flow process is some sort of permutation of the
input data. The most important feature of the step of flow used is that it
should be invertible. This is accomplished by keeping half of the
information unprocessed and using it to transform the other half of the
data. Previous work has implemented this permutation by a fixed permutation
scheme used in the training pipeline. The work in Glow was to instead learn
a permutation as part of the training. This is implemented by a 1x1
invertible convolution operation (a general permutation operation).

#### Affine Coupling

The last step in flow is the actual transformation of one part of the data
given the other part (with optional addition of some conditional data). The
data flowing through the flow step unprocessed is also used as input to
some neural network, **NN()**, which outputs are used to scale and shift the
other part of the input data. The important thing here is that **NN()** does
not have to be invertible and could be as complex as needed.


-------------------------------------

### Multi-scale Architecture

This architecture comes from the paper [Density Estimation using Real-NVP](https://arxiv.org/pdf/1605.08803.pdf).
(Real = real valued. NVP = Non Volume preserving. Referring to the probability density function (volumes) at
each layer.)

![Multiscale architecture](/assets/images/Glow/Glow-multi-scale-architecture.png)

The multi-scale architecture is a way to progressively learn features on different scales
of the data. Instead of sending everything through the network for a final output
intermittent steps provide outputs for different levels of scale which makes it possible
to make the volume of parameters/variables, being processed, "smaller and smaller". The main
idea is depicted in the figure (b). The final output of the network is the concatenation
of the outputs at the different level of scale. The multi-scale architecture starts with a
**squeeze** operation, followed by steps of flow, then ends with a **split** operation.
Below is the equation for these layers where each f^(i+1) is at the end of the split.
Thus, inside the network we get two tensors after each split, the data and the output at
this scale.

![Multiscale architecture equations](/assets/images/Glow/Glow-multi-scale-architecture-equations.png)

-------------------------------------

## Training Glow

The training of glow is somewhat backwards in regard to "regular" deep
learning. The input to the network is the data (image, audio, text, etc...) and
the outputs are the correlated latent variable z. Because the entire structure
of the model is invertible when applying the model for inference the whole
thing is run in reverse. This means that all components in the architecture
will have one transformation forward as well as backwards. In most
implementation this is named regular/reverse. But one can also talk about it as
encoding/decoding where the encoding takes a point in the data space, X, and encodes it to the
latent space, Z. The decoding does the reverse and takes a point from Z and
decodes it to X.


