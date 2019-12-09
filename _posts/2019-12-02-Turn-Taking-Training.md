---
layout: post
author: Erik
---


Goal: Predict the future voice activation from diadic dialogues.


## From Simplest Towards Complex


How should we think about the problem? What input should we use and why?

<!--more-->


## Starting Point

Based on the fact that out output space consists of binary values for voice activation the
simplest model would use the same space as input space.


## Non Discrete Feature, Intensity

Instead of actually training on the VAD (requires some preprocessing step which
could be better or worse) we instead train on the next best thing.  The voice
activity will most probably be defined by taking into account intensity, the
amount of energy or power, in each audio frame. Lets see how well the model does
on only the intensity as intput.  


| **Inputs** | **Description**  |
| :----- | :------  |
| Intensity | Well defined everywhere. bounded [0, normalized_max(power, energy) ].  Continuous  |
| **Outputs** | **Description**  |
| Binary Voice activity (VAD) | Well defined everywhere. Bounded [0, 1]. Discontinuous  |


* Model
  - [ ] Window FC
  - [ ] Window Conv
  - [x] frame RNN
  - [ ] window RNN


Training
- Do we converge?
- How does the predictions look?



## Adding Prosody

Lets use the model from the previous section but now we add some prosodic information we
guess should be useful. We extract the intensity for each frame along with the pitch. The
intensity is straight forward to implement and is defined everywhere, it is a continuous
number between 0-no intensity and max, the maximum intensity possible (bounded by the
maximum energy of the microphone etc. maximum intensity of the audio signal). The maximum
could be normalized speaker wise or across the dataset. The pitch, however, is another
story. The basis for the pitch is the fundamental frequency, the first harmonic, of voiced
speech. Usually in the region 85-180Hz for males and 165 - 255Hz for females but it is
only present during voiced regions of the signal. A frequency $ \frac{1}{n} != 0 $


| **Inputs** | **Description** |
| :----- | :------ |
| Intensity | Well defined everywhere. bounded [0, normalized_max(power, energy)].  Continuous |
| Pitch (F0) | Semitones. Well defined on voiced segment. Not defined at unvoiced. Z-normalized to disentangle speaker info |
| **Outputs** | **Description** |
| Binary Voice activity (VAD) | Well defined everywhere. Bounded [0, 1]. Discontinuous |


* Model
  - [ ] Window FC
  - [ ] Window Conv
  - [x] frame RNN
  - [ ] window RNN

- Normalize/no-normalization of pitch
  - Effects convergence?
- Did adding the intensity and pitch improve the model

## Adding MFCC (first four)


| **Inputs** | **Description** |
| :---------- | :------ |
| Intensity | Well defined everywhere. bounded [0, normalized_max(power, energy)].  Continuous |
| Pitch (F0) | Semitones. Well defined on voiced segment. Not defined at unvoiced. Z-normalized to disentangle speaker info |
| MFCC (0-3) | Defined everywhere. Different magnitudes for the different coefficients |
| **Outputs** | **Description** |
| Binary Voice activity (VAD) | Well defined everywhere. Bounded [0, 1]. Discontinuous |


* Model
  - [ ] Window FC
  - [ ] Window Conv
  - [x] frame RNN
  - [ ] window RNN


## Results

The results from training these 3 types of network showed something interesting. The
losses seems to be very similar but the scoring metric is different! All models converges
towards the same loss for both training and validation set (underfitting?) but although
intensity and intensity+pitch does about the same the added MFCC feature does much better
on the validation metric.

### Loss

The loss is very similar for all three models. They converge in about the same time and
reached the same loss (validation and training).

![training loss](/images/turntaking/tt_training/loss_small.png)

### Fscore on shift and hold events

The fscore metric shows that although the losses are similar the performance on the metric
is not. The model with the added MFCC feature doessignificantly better than the other
iterations.

![training fscore 50, 100ms](/images/turntaking/tt_training/fscore_small_05_1.png)
![training fscore 200, 500ms](/images/turntaking/tt_training/fscore_small_2_5.png)


## Prosody Information important when using MFCC features?


One straightforward follow up question is now whether the prosody information (pitch and
intensity) does anything valuable when using the MFCC features. Lets investigate that by
only training on the MFCC features.

For the training with only MFCC features the losses were again pretty much
exactly the same but as with the pitch, intensity, mfcc model it score better
than only the intensity and pitch model.

<img src="/images/turntaking/tt_training/fscore_small_all_05_1.png" alt="" width="100%">
<img src="/images/turntaking/tt_training/fscore_small_all_2_5.png" alt="" width="100%">




### MelSpectrogram, Looking at the complete spectrum

| **Inputs** | **Description** |
| :---------- | :------ |
| MelSpectrogram | Well defined everywhere. Continuous. Using Frequency z normalization over each speaker |
| **Outputs** | **Description** |
| Binary Voice activity (VAD) | Well defined everywhere. Bounded [0, 1]. Discontinuous |

