---
layout: post
title: SincNet
---

SincNet is using knowledge about what frequencies are more important for speaker
identification. The main contribution is to redefine a regular `Conv1D` layer in deep
learning to learn only a bandpass filter. This greatly reduces the number of parameters
needed as the first step of encoding a raw audio-sample signal.

<!--more-->

<img src="/assets/images/SincNet/SincNet.png" alt="SincNet" height='600' style='float:right'/>


Contributions mentioned by the authors
* Fast Convergence
* Few Parameters
* Computational Efficiency
* Interpretability


## Data

Dataset used in paper:
* LibriSpeech
* TIMIT

Cleaning:
* "Non-speech intervals at beginning and end of each sentence were removed"
* "(LibriSpeech) internal silences lasting more than 125ms were split into multiple chunks"
* "To address text-independentspeaker recognition, the calibration sentences of TIMIT (i.e.,the utterances with the same text for all speakers) have beenremoved"
* "(TIMIT) five  sentences  for  eachspeaker  were  used  for  training,  ... three were  used  for  test"
* "(Librispeech) training and test material have been randomly selected to exploit 12-15 seconds of training material for each speaker and test sentences lasting 2-6 seconds"
