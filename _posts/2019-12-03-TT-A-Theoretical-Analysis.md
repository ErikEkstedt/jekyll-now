---
layout: post
author: Erik
---

## Data: Real, dataset, model and output

What kinds of data are we going to work with? How to process this data? What do we wish to achieve


<!--more-->


# Data

## Input space

## Acoustic Information

Spectrogram Fourier transform of 1D audio (intensity) signal -> 2D frequency/power space.

### Mel-Spectrogram

A melspectrogram is a constructed from a frequency power spectrogram from an audio signal mapped onto a mel space.
The mel mapping maps frequencies into bins more akin to how humans seem to experience audio.

A regular power spectrogram over 3 seconds of spoken audio


-> mapped onto the mel space

#### Spectrum

The power for each frequency band in a single frame is a spectrum

We note that the lower frequencies are more distinguishable in the mel space than in the power spectrogram. Important frequencies for speech such as the
fundamental frequency F0 and its harmonics F1, F2 are given more importance in the mel space. The higher frequencies (noise) is less prominent.

In Deep Learning it is very common to process melspectrograms instead of the original power specotrogram.


## Example

*"Nobody understands quantum mechanics"* - Richard Feynman 1964

Parameters
* step time: 10ms (step length of fourier transform)
* window time: 50ms (window size of fourier transform)

<div class="row">
  <div class="column">
    <h3> Waveform </h3>
  </div>
  <div class="column">
    <h3> MelSpectrogram </h3>
  </div>
  <div class="column">
    <h3> Spectrum over time</h3>
  </div>
</div>
<div class="row">
  <div class="column">
    <img src="/images/turntaking/tt_analysis/nobody_understands_quantum_mechanics_waveform.png" alt="ALL" width='100%'>
  </div>
  <div class="column">
    <img src="/images/turntaking/tt_analysis/nobody_understands_quantum_mechanics_spec.png" alt="ALL" width='100%'>
  </div>
  <div class="column">
    <video width="100%" controls loop>
      <source src="/images/turntaking/tt_analysis/feynman_spectrum.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>


# Datasets

Parameters
* Sample Rate: 8000 Hz
* max frequency: 4000Hz (Sample rate // 2)
* step size: 50ms
* window size: 50ms
* n mels: 80

**Switchboard**

<img src="/images/turntaking/tt_analysis/spectrum_swb.png" alt="ALL" width='400px'>

**Maptask**

<img src="/images/turntaking/tt_analysis/spectrum_maptask.png" alt="ALL" width='400px'>


#### Single Dialog Speaker 0

<img src="/images/turntaking/tt_analysis/Melspectrum_Single_Speaker_0_(all_vs_vad_vs_novad).png" alt="Channel 0" width='400px'>

#### Single Dialog Speaker 1

<img src="/images/turntaking/tt_analysis/Melspectrum_Single_Speaker_1_(all_vs_vad_vs_novad).png" alt="Channel 1" width='400px'>



### Intensity

### Pitch Semitone

### Words
  - Glove Embedding
    - Avg
  - Stop Word rate

Output Space
- VAD
- Turn-taking Events
