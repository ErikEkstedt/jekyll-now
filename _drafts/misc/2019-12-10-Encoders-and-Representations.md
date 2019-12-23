---
layout: post
author: Erik
---

<div class='row'>
  <div class='column'>
    <b>Wavenet</b>
    <img class='centerImg' src="/images/turntaking/tt_cnn/wavenet-caus-convs.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/turntaking/tt_cnn/wavenet-dilconvs.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/turntaking/tt_cnn/wavenet_gates.png" alt="" width="100%"/>
  </div>
  <div class='column'>
  <b>VQVAE</b>
    <img class='centerImg' src="/images/turntaking/tt_cnn/VQVAE.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/turntaking/tt_cnn/VQVAE2.png" alt="" width="100%"/>
  </div>
  <div class='column'>
    <b>CPC</b>
    <img class='centerImg' src="/images/turntaking/tt_cnn/CPC.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/turntaking/tt_cnn/CPC2.png" alt="" width="100%"/>
  </div>
</div>

* [WaveNet, van den Oord, 2016](https://arxiv.org/abs/1609.03499)
    - [vincentherrmann/pytorch-wavenet)](https://github.com/vincentherrmann/pytorch-wavenet)
* [VAE](https://arxiv.org/pdf/1312.6114.pdf)
* [VQVAE](https://arxiv.org/pdf/1711.00937.pdf, vad den Oord)
* [VQVAE 2](https://arxiv.org/pdf/1906.00446.pdf)
  - [rosanality/vq-vae-2-pytorch](https://github.com/rosinality/vq-vae-2-pytorch)
* [CPC](https://arxiv.org/pdf/1807.03748.pdf)
* [Data-Efficient Image Recognition with CPC](https://arxiv.org/pdf/1905.09272.pdf)


<!--more-->

## Introduction

* CNNs are parameter efficient and has been ubiquitous in deep learning tasks concerning vision
* CNNs have been shown to do well on time sequences with long range dependencies (wavenet)
* Learning a latent space that approximates the underlying distribution of data has been a goal for
    machine learning algorithms for a long time. (VAE)
* Approximating the underlying distribution discretely is a powerful idea shown to work very well
    (VQVQVAE, VQVAE2)
* Autoencoder structures are a common way to design algorithms to learn representations
    - The last part of an autoencoder transforms a latent space back to the input space and
        calculates a reconstruction error
    - "not every bit is equal" is a phrase indicating that reconstructing the input space is
        problamatic
* CPC learns to reconstruct latent space instead of input space.
  * This can be more efficient (CPC2)



## Input Space & Output Space for Conversations

* Audio
  * Waveform, Spectrograms
  * Extracted features (pitch, duration, intensity, ...)
  * Language: Text
* Vision
  * gaze, gestures etc


We first concentrate on audio because it includes all the necessary information for most
conversations. The original form of audio is pressure differences in air wich is measured by
micropohones to discrete intensity samples over time. Fourier transformation of the waveform creates
a spectrogram, frequency space. It is common to map the pure frequencies to mel-frequencies, which is more suitable for speech. 

Given a (clean) spectrogram a trained linguist can read/extract phonetic and prosodic information.
That is to say that if we treat the spectrogram as an image, then reading left to right, there are
patterns that form discrete tokens (phonemes). 

<img class='centerImg' src="/images/turntaking/tt_analysis/nobody_understands_quantum_mechanics_spec.png" alt="ALL" style='width: 40%'>

The idea to treat the spectrogram as an image is straight-forward if we only care about the content
of the spectrogram and are not concern about any time constraints. However, if we need to reason
about the content continuously we have to be careful about not letting any future information effect
our current predictions.

* How to use a CNN over arbitrary long sequences as encoder to a continuous model (e.g RNN)? 
* How to make sure that no information flows "backwards" in time (regardless of the depth of the network)
* The content of a spectrogram is $$Frequency \times Time$$, for regular images it is $$ Spatial \times Spatial $$. 
  * Is this a problem for encoding spectrograms?  
  * Can one use 2D-convs on spectrograms effectively?
  * Is 1D-convs enough (better than 2D-convs)?


## Continuously, time constrained encoding of spectrograms with CNNs

In order to use a CNN based encoder for continuous data we want to find out how to achieve two
things. First that the encoder representations at each step only depends on the previous up and to
the current input. Secondly, we want to find out how much previous information is used, that is the
receptive field of the encoder for each time step. This value will give us an indication for what
type of information the encoder can possible encode.

A Convolutional Neural Network, CNN, are defined by a set of paramters.
* kernel size, the size of the matrix/window processing the information
* stride, the step of which to jump for each 
* dilation, the spread of the elements being processed by a single krenel operation
* padding, the amount of padding to add to the input
* Receptive Field, RF
  - How much does a feature at any given time step see?
  - [How to compute the receptive field](https://distill.pub/2019/computing-receptive-fields/)

The kernel size defines how many features to process from the previous layer. Stride defines the
"step" of which to jump when applying the convolution. The dilation defines how spread out the
elements used in the kernel multiplication is. A dilation of 1 yields a normal convolution, all
features processed by the kernel, at any given step, is seperated by 1 i.e they are in consecutive
order. For a dilation of 2 the elements are now separated by two, that is every other element in the
previous layer is processed by the kernel, for any given kernel operation.

The receptive field of a convolutional output is the size of the interval of which it processes
information. For example a single layer with kernel-size 3, stride 1 and dilation 1, each step
processes 3 consecutive input elements and thus the receptive field is 3. If the dilation is 2 then
we process 3 input elements separated by 1 and the receptive field becomes, $$3*2=6$$. The stride
does not effect the receptive field of a single layer output but does so very much for multiple
layers.

[How to compute the receptive field](https://distill.pub/2019/computing-receptive-fields/) gives a
much in depth analysis (and better) analysis for convolutional networks and their receptive fields.


## Causality

In their famous paper [WaveNet (van den Oord, 2016)](https://arxiv.org/abs/1609.03499) the authors
use causal 1D convolutions on a waveform.

<img class='centerImg' src="/images/turntaking/tt_cnn/wavenet-caus-convs.png" alt="" width="50%"/>

This causation is quite simple achieved by applying a padding to the input elements. In a regular
convolution we think about the output as being "in the middle" of the kernel with information about
both sides (past and future). When we apply padding to the left we shift the data and can now view
the convolutional output as in the image above.

We create a casual convolutional layer by applying paddin to the left and look at the output for
varying amount of layers. The images below depict the input and output of the different models.

* kernel size: 3
* stride: 1
* dilation: 1
* All weights are initialized to 1 and no bias
* 1D convs: 15 feature maps are shown
* 2D convs: 1 feature map is shown

<div class='row'>
  <div class='column'>
  <center> <b> Regular 1D</b> </center>
    <img class='centerImg' src="/images/cnn/1_regular_conv1d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/2_regular_conv1d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/3_regular_conv1d.png" alt="" width="100%"/>
  </div>
  <div class='column'>
  <center> <b> Regular 2D</b> </center>
    <img class='centerImg' src="/images/cnn/1_regular_conv2d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/2_regular_conv2d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/3_regular_conv2d.png" alt="" width="100%"/>
  </div>
  <div class='column'>
  <center><b>Casual 1D</b></center>
    <img class='centerImg' src="/images/cnn/1_causal_conv1d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/2_causal_conv1d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/3_causal_conv1d.png" alt="" width="100%"/>
  </div>
  <div class='column'>
  <center><b>Casual 2D</b></center>
    <img class='centerImg' src="/images/cnn/1_causal_conv2d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/2_causal_conv2d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/3_causal_conv2d.png" alt="" width="100%"/>
  </div>
</div>

Here we see that the only nonzero input frame produces different features for the causal and the
regular convolutions. The import thing is for the information in the output layer to **ONLY** depend
on information from the current or previous frames. All activations should be to the right of the
nonzero input. Imagine going backwards from the top then going towards the right (downwards) is to
go forward in time which is not allowed. The causal cnns satisfy this condition.

### Dilation

In deepminds work the input space of the model is a 1D time sequence of intensities (waveform).
These sequences contain commonly 16000 samples every second. Thus in order to keep the number of
parameters low and possible get better representations they utilize dilation. This dilation makes it
so that the receptive field is vastly increased with fewer layers.

<img class='centerImg' src="/images/turntaking/tt_cnn/wavenet-dilconvs.png" alt="" width="50%"/>

However, if the input space is defined by spectrograms the time axis is greatly decreased and might
contain 20-100 frames each second. This is a great reduction in time steps and thus dilation might
not be as important.


<div class='row'>
  <div class='column'>
  <center> <b> Regular 1D</b> </center>
    <img class='centerImg' src="/images/cnn/1_regular_dilation_conv1d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/2_regular_dilation_conv1d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/3_regular_dilation_conv1d.png" alt="" width="100%"/>
  </div>
  <div class='column'>
  <center> <b> Regular 2D</b> </center>
    <img class='centerImg' src="/images/cnn/1_regular_dilation_conv2d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/2_regular_dilation_conv2d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/3_regular_dilation_conv2d.png" alt="" width="100%"/>
  </div>
  <div class='column'>
  <center><b>Casual 1D</b></center>
    <img class='centerImg' src="/images/cnn/1_causal_dilation_conv1d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/2_causal_dilation_conv1d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/3_causal_dilation_conv1d.png" alt="" width="100%"/>
  </div>
  <div class='column'>
  <center><b>Casual 2D</b></center>
    <img class='centerImg' src="/images/cnn/1_causal_dilation_conv2d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/2_causal_dilation_conv2d.png" alt="" width="100%"/>
    <img class='centerImg' src="/images/cnn/3_causal_dilation_conv2d.png" alt="" width="100%"/>
  </div>
</div>


The output is looking the way we want. However, because of the nature of convoloutions some output values
are smaller than others and from this point we will treat all the nonzero outputs as one in order to
see more clearly.

* strides ?

## Stacks of Causal (dilated) CNNs

Given that we now may process information casually we need to define how our encoder atchitecture. A
common "block" in many convolutional networks is the "ResNet block" introduced in [Deep Residual
Learning for Image Recognition, He et al 2015](https://arxiv.org/pdf/1512.03385.pdf). In both
[VQVAE](https://arxiv.org/pdf/1711.00937.pdf, vad den Oord) and [VQVAE
2](https://arxiv.org/pdf/1906.00446.pdf) the first two layers of the encoder are regular CNNs with a
stride of two and kernel size of 4. This means that the spatial dimensions are halfed through each
layer. Following this "downsampling" the data flows through a two layered Resnet block. A resnet
block is defined by two weight layers separated by an activation (relu) followed by a skip
connection. (The image below is from the resnet paper).

<img class='centerImg' src="/images/cnn/ResnetBlock.png" alt="" width="50%"/>

Lets construct a casual ResNet Block and test that causality holds. In the images below we see the
output from multiple Casual ResNet Blocks (5 layers). The outpu to the left is the randomly
initialized output and to the right is the output initialized to all ones and all the nonzero output
has a value of one.

<div class='row'>
  <div class='column'>
  <center> <b> Causal Resnet Block D: 3, 2 K: 3</b> </center>
  <img class='centerImg' src="/images/cnn/causal_resnet_block_d3_k3.png" alt="" width="80%"/>
  </div>
  <div class='column'>
  <center> <b> (all nonzero output = 1) Causal Resnet Block D: 3, 2 K: 3</b> </center>
  <img class='centerImg' src="/images/cnn/causal_resnet_block_d3_k3_thresh.png" alt="" width="80%"/>
  </div>
</div>

## Encoder

The ResNet block output seems to be okay so lets add the first downsampling layers which defines the
hidden channels we want to use (ResNet blocks has the same in and out channels). After this initial
layer we put our resnet block followed by an output layer (conv1D) to project the data to any
desired output space. This is our initial encoder structure. Lets see how the output looks but also
lets backprop through the encoder to the ouput and see where we get gradients. For the images below
we use a 5 layer encoder (1 pre-layer, 2 layer resnet blocks, 1 output layer) with a kernel size of
3 and dilation 1. The gradients are calculated from the time step indicated by the black line.

<div class='row'>
  <div class='column'>
  <center> <b> Small dilation (1) </b> </center>
  <img class='centerImg' src="/images/cnn/encoder_grad_L5_K3_D1.png" alt="" width="90%"/>
  </div>
  <div class='column'>
  <center> <b> Larger dilation (3) </b> </center>
  <img class='centerImg' src="/images/cnn/encoder_grad_L5_K3_D3.png" alt="" width="90%"/>
  </div>
</div>


Here we see that the gradient is zero "in the future" to the right of the black line which is
exactly what we want.

What about a larger receptive field through dilation? Well, the gradient reaches further into the
past (larger receptive field) but we actually don't get any information from every other time step,
there is no gradient values here. This means that the information we could possibly extract for this
specific time step has to be in the yellow frames in the gradient plot. 


### Multiple Stacks

If we require more information at every time step then we could try to have multiple resnet stacks
(pre-layer, resnet blocks, output layer) and then combine the total output.  The output may be
combined through concatenation or addition. In the image below we use both the stacks from the
previous images, combine them, and calculate the gradient.

<div class='row'>
  <div class='column'>
  <center> <b> Encoder Kernel size: 3 & Dilation: 1, 3 </b> </center>
  <img class='centerImg' src="/images/cnn/encoder_grad_L5_K3_D1-3_add.png" alt="" width="90%"/>
  </div>
  <div class='column'>
  <center> <b> Encoder Kernel size: 3, 5 & Dilation: 1, 3 </b> </center>
  <img class='centerImg' src="/images/cnn/encoder_grad_L5_K3-5_D1-3_add.png" alt="" width="90%"/>
  </div>
</div>

Not surprisingly we end up with the combined output and gradient from the previous cases. The
gradient shows us what information is available at any given time step and now we can reason about
how to design our encoder further. The left image above shows the gradient for two stacks both with
kernel size 3 and a dilation of 1 and 3 respectively. The right image above shows the gradient for
an encoder with 4 stacks with kernel sizes 3 and 5 and dilation 1 and 3.

Additions:
* Combine the output from all stacks at each layer not just last


## Receptive Field

TODO: add receptive field function to know the receptive field beforehand. Check against gradient to
see that they match.

* [ ] Receptive Field

## Test

* In order to encode a latent space containing the information of the past we may want to learn a representation through an AE structure
* What is the receptive field for the latent space for any given time step?
* What information in the receptive field are we interested in? Reconstruct
* VQVAE
* CPC

Okay so now we have a Causal 1D-Conv encoder. How do we know what metrics to use? 




## Test


Train on the turntaking task.
* CNNEncoder + fc -> output
  - one stack (different kernel and dilation)
  - multiple stacks
* CNNEncoder + rnn -> output
  - one stack (different kernel and dilation)
  - multiple stacks
  - layers of rnn



