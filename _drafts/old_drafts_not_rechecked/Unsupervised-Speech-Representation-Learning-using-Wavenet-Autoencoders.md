---
layout: post
title: Unsupervised Speech Representation Learning using Wavenet Autoencoders
---


[Unsupervised Speech Representation Learning using Wavenet Autoencoders](https://arxiv.org/pdf/1901.08810.pdf)


<!--more-->
## Model


* Encoder
  - MFCC + 1 derivative + 2 derivative  -> 39D
    - features every 10ms -> 100 Hz
  - Conv, kernel=3, stride=1, channels=768
  - Conv, kernel=3, stride=1, channels=768 + residual
  - Conv kernel=4, stride=2, channels=768 
    - Downsamples the signal by half
    - 20ms -> 50 Hz
  - Conv, kernel=3, stride=1, channels=768
  - Conv, kernel=3, stride=1, channels=768 + residual
  - 3 Layer FC 768, relu, residual connection

