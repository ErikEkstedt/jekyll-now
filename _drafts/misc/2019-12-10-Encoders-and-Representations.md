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
* [VQVAE 2]( https://arxiv.org/pdf/1906.00446.pdf )
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

Given that we now may process information causally we need to further define our encoder atchitecture. A
common "block" in many convolutional networks is the "ResNet block" introduced in [Deep Residual
Learning for Image Recognition, He et al 2015](https://arxiv.org/pdf/1512.03385.pdf). In both
[VQVAE](https://arxiv.org/pdf/1711.00937.pdf, vad den Oord) and [VQVAE
2](https://arxiv.org/pdf/1906.00446.pdf) the first two layers of the encoder are regular CNNs with a
stride of two and kernel size of 4. This means that the spatial dimensions are halfed through each
layer. Following this "downsampling" the data flows through a two layered Resnet block. A resnet
block is defined by two weight layers separated by an activation (relu) followed by a skip
connection. (The image below is from the resnet paper).

<img class='centerImg' src="/images/cnn/ResnetBlock.png" alt="" width="50%"/>

Lets construct a causal ResNet Block and test that causality holds. In the images below we see the
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

## Learn What Representations in What Space?

Learn distributions from Autoregressive Probabilistic Models (LM) or Reconstruction Auto Encoder models

## Space

What to reconstruct?
- Wavenet "reconstructs"/"generates" discrete values of samples. 
    - 0-255 classes at Hz
- Reconstruct next frame in spectrogram?
    - Sigmoid at output for each channel -> MSE
    - Discretize between min-max values for the intensity of each channel
      - 80 Frequencies
      - 256 values each -> 20480 total classes
      - How did melnet do it?




## Probabilistic Autoregressive Model, density optimization

* [PixelCNN](https://arxiv.org/pdf/1601.06759.pdf) Models images through maximum likelhood where each color channel value is defined by an $$N = 2^8$$
    categorical distribution (softmax) 
* [WaveNet, van den Oord, 2016](https://arxiv.org/abs/1609.03499) outputs a categorical distribution over the sample value for each timestep
  * The catagorical ditribution is defined by a softmax over N ($$2^8 = 256 \in [0, 255] $$) categories
  * Optimize the maximum likelihood of the data w.r.t the parameters
  * "Because log-likelihoods are tractable, we tune hyperparameters on a validation set and can easily measure if the model is overfitting or underfitting"
* [Melnet](https://arxiv.org/pdf/1906.01083.pdf) 
  * Factorize a joint distribution over a spectrogram x as a product over conditional distributions
  * Define an order for the conditional distributions $$ P(x) = \prod P(x_i \| x_{<i}) $$
    * Melnet define from low to high frequency for each spectrogram frame
    * They model their distribution by $$ \theta $$. Which outputs the vector $$ \theta \in R^{3K} $$.
    * A mixture of $$K$$ distributions
    * That is $$ \theta = \{ \mu, \sigma, \pi \}^{K}_{k=1}$$gg
    * Constraints on the outputs to define "simple" Gaussian Misxtures
      * linear for mean $$\mu $$, Sigmoid for $$\sigma$$ positive standard deviations and mixture
          coefficients $$\pi$$ summing to one by a softmax
  * Have the parameters be defined by the maximum likelihood estimate with regard to the data
  * This maximum likelihood estimation of parameters is the same as optimizing the negative log
      likelihood 

### [VQVAE 2]( https://arxiv.org/pdf/1906.00446.pdf )

Much is paraphrased or directly quoted from the paper.


* VQVAE models can be "better" understood as a communications system.
    - Why is communication viewpoint advantageuous over ... What are the alternative view points?
* An Encoder maps Observations onto a sequence of discrete variables.
* A Decoder that reconstructs the observations from discrete variables.
* Encoder/Decoder uses a shared codebook
* Compare output vector of encoder with codes in the codebook (prototypes)
    - distance metric
* The encoder output vector is quantized into indicies, an index, of the shared codebook 
* The Decoder reads the index, gets the prototype code then generates an output
    - the output can be reconstruction of the input, partial parts of the input, missing things in
        the input, etx
    - The output can be to generate future segments (becomes autoregressive)
* The output loss of the Decoder (can be reconstruction or next frame(partial input)) is
    backpropagated to learn the mappings
* I goes from the decoder to the encoder via the "straight-through" gradient estimator

The loss used is the reconstruction mean square error to learn the appropriate mappings for
reconstruction. The generative or reconstructive capabilities measured in $$L_{capabilities}$$


$$ L_{capabilities} = \| \mathbf{x} - D(\mathbf{e}) \| ^2_2$$ 

Then adds two auxiliary losses used to regularize the learning. 

> "The *codebook loss*, which only applies to the codebook variables, brings the selected codebook
> $$\mathbf{e}$$ close to the output of the encoder, $$E(\mathbf{x})$$." 


$$ L_{codebook} = \| sg(E(\mathbf{x})) - \mathbf{e}) \| ^2_2$$ 

> "The *commitment loss*, which only applies to the encoder weights, encourages the output of the
> encoder to stay close to the chosen codebook vector to prevent it from fluctuating too frequently
> from one code vector to another"

$$ L_{commitment} = \| sg(\mathbf{e}) - E(\mathbf{x}) \| ^2_2$$ 

Let $$\beta$$ be a scaler controlling the reluctance to change the code corresponding to the
encoder and $$sg(·)$$ is the straight-through gradient estimator. Then the total objective is to
minimize the loss.

$$ L(\mathbf{x}, D(\mathbf{e})) = L_{capabilities} + L_{codebook} + \beta L_{commitment} $$ 


The codebook loss $$L_{codebook}$$ is replaced by exponential moving
averages([VQVAE](https://arxiv.org/pdf/1711.00937.pdf, vad den Oord)) updates for the codebook as a
replacement for the codebook loss.
* why? meta learning updates averaged over mini batch samples

### Stage 1: VQ-VAE training


1. Map Input X to Z
  - Z consists of to hierarchies top and bottom
2. Quantize the top encoded vectors to a top prototype
3. Read the top prototype and the input to encode the bottom vector 
4. Quantize the bottom encoded vector to a bottom prototype
5. The Decoder reads both the top code and the bottom code and reconstructs the input.
6. Update all the parameters $$ Update(L(\mathbf{x}, \mathbf{\hat{\mathbf{x}}}))) $$ 
  - Probably Adam


### Causal VQVAE

The original VQVAE2 networks encodes a color image $$(3, 256, 256 )$$ down to a top and bottom
representation of size $$(32, 32)$$ and $$(64, 64)$$ respectively. Then discretizes the
representations to codes stored in a codebook(learnable), one for the top codes and one for the
bottom. Then decodes those discrete repesentations through a decoder (residual stacks and
ConvTranspose) in order to reconstruct the original image. 

For our purposes we wish to compress a melspectrogram, $$(1, Time, 80)$$, where 80 is the number of
mel channels used. We want to quantize the data in a causal way such that no encoding at any time
step depends on future frames. Given that the input data is of the form $$(Batch, Channels, Height,
Width)$$, the way PyTorch defines it for their Conv2d, we want to compress the information such that
no information in the reconstructed output depends on any information above the current "height". Of
course height and width are just words and all we care about is that the information at $$(B, 1, t_0,
N_{mels})$$ does not depend on $$(B, 1, t_1, N_{mels})$$ for any $$ t_1 > t_0 $$.


**Cuda vs Cpu ?**

Using Cuda for the convolutional operation seems to introduce a small problem. By default our causal
network does not work. When using CPU the gradient looks as expected, however, using `to('cuda')`
changes the gradient for the worse. For cuda the receptive field and the gradient interval is larger
but worse the gradient at time step 50 depends on future frames. It seems that pytorch requires
`torch.backend.cudnn.deterministic=True` to work properly for causal implementation.


<div class='row'>
  <div class='column'>
    <b>CPU</b>
    <img class='centerImg' src="/images/cnn/vqvae_encoding_grad_cpu.png" alt="ALL" style='width: 90%'>
  </div>
  <div class='column'>
    <b>cudnn.deterministic=False</b>
    <img class='centerImg' src="/images/cnn/vqvae_encoding_grad_cuda.png" alt="ALL" style='width: 90%'>
  </div>
  <div class='column'>
    <b>cudnn.deterministic=True</b>
    <img class='centerImg' src="/images/cnn/vqvae_encoding_grad_cuda_deterministic.png" alt="ALL" style='width: 90%'>
  </div>
</div>

**Moving on**

Now when the cuda causality error is out of the way we change the Encoder part of the VQVAE to only
use causal Convolutions and residual stacks. The hierarchical nature of the algorithm works by first
encoding the input into a bottom representation and a top representation. The bottom being
downsampled by a factor of 4 (in $$N_{mels}$$) and the top representation downsamples the bottom
representation further by 2(in $$N_{mels}$$), that is 8 in total. 

The top representation is then fed through a convolutional layer with a kernel size of 1 in order to
have the same numbers of channels as the dimensionality of codes in the codebook. This
representation is then compared to the codes in the codebook, through a distance metric (MSE), and
the closest code, for each time and frequency element, is chosen for the quantized representation.

The quantized top representation is then decoded, through a residual stack and a convtranspose
layer, back to the shape of the bottom representation. This decoded top representation is then
concatenated with the bottom representation.

This combined representation, like the top representation, is passed through a Conv2d with a kernel
size of 1 to match the dimensionality of the codebook. The quantization is done in the same way as
for the top, albeit with another codebook, to produce a quantized representation for the bottom
features.

In the original implementation the top and bottom quantized tensors are fed through the decoder to
reconstruct the input image. However, for our purposes we need to check that the encoder works as we
hope, is it causal?

Below are the gradients for the top and bottom quantized encoder output respectively.

<div class='row'>
  <div class='column'>
    <b>Bottom</b>
    <img class='centerImg' src="/images/cnn/vqvae_encoder_bottom_rf.png" alt="ALL" style='width: 90%'>
  </div>
  <div class='column'>
    <b>Top</b>
    <img class='centerImg' src="/images/cnn/vqvae_encoder_top_rf.png" alt="ALL" style='width: 90%'>
  </div>
</div>

The gradient is calculated with respect to the output at the time step indicated by the black line
(step 30). The output, the top of the images, is the first value of the 64 dim code for each time
step. The bottom representation contains 20 codes at each time step $$n_{mels} = 80 \to 20$$ and the
top representation contains 10 codes $$n_{mels} = 80 \to 10$$. In other words, each time step
contains 20 and 10 codes for the bottom and top representation respectively.


The receptive fields are not the same for the top and the bottom representations? Why?

Now, the next question is what we wish these codes should represent?

* Can the $$ (20+10) \times code_{dim}$$ reconstruct the entire receptive field?
  * If not how many frames, inside the receptive field closest to the current frame, can it
      reconstruct?
* Can this discrete latent code be used in an autoregressive way?
    - Can the representations from each step generate the next frame?



## Setup

Given that we want to reconstruct/generate the frames of a spectrogram we could do this continuously
(MSE) or discretely (softmax / NLL). Lets follow the arguments from PixelCNN and discretize the
space. 

A spectrogram is a representation of audio recorded under different circumstances. Because of this
we want to normalize them in order for them all to be more similar, i.e act as samples from a
smaller distribution. We wish to discretize the values for each bin so in order to do that we need
to normalize the spectrograms in a min-max fashion.

* Normalize
  - min-max normalize a speaker recording (over the entire recording) and then discretize each frequency channel into N bins.
  - 8 bits $$2^8 = 256 $$,  6 bits $$2^6 = 64 $$, and so on
  - A common size of a mel spectrogram is 80 channel
  - 8 bits * 80 channels = 640 bits / frame -> $$2^{640} \approx 4.56e+192 $$ different states
* Discretize
  - The total space of each frame is huge
  - Following the Melnet paper we define an order, i.e dependency, (e.g low to high frequency). 
  - RNN conditioned on encoding of previous information start from low and given the previous value output the next for $$n_{mels}$$ steps.

<div class='row'>
  <div class='column'>
    <b> Discrete Spectrogram vs normal </b>
    <img class='centerImg' src="/images/cnn/discreteSpectrograms.png" alt="" width="100%"/>
  </div>
  <div class='column'>
    <b> A single Frame</b>
    <img class='centerImg' src="/images/cnn/discreteSpectrogramsCategories.png" alt="" width="100%"/>
  </div>
</div>

How well does the simple CNN encoder do at spectrogram generation? (hard task so we do not expect that much)


#### Model

* Encode the history with the causal 1D-CNN-Encoder
* The output of the encoder initializes the hidden state of an RNN
* The RNN autoregressively generate the output frame
  - Teacher forcing ?



## Reconstruct the already known

* Reconstrcut receptive field. 
  * Each representation should include the acoustic information in the
    receptive field.
* What is the receptive field for the latent space for any given time step?
* What information in the receptive field are we interested in? Reconstruct
* VQVAE
* CPC


## Test


Train on the turntaking task.
* CNNEncoder + fc -> output
  - one stack (different kernel and dilation)
  - multiple stacks
* CNNEncoder + rnn -> output
  - one stack (different kernel and dilation)
  - multiple stacks
  - layers of rnn



