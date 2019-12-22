---
layout: post
author: Erik
---

## Data: Real, dataset, model and output

What kinds of data are we going to work with? How to process this data? What do we wish to achieve


<!--more-->


# Data


<img src="/images/turntaking/turntaking.png" alt="ALL">


## Input space

* Audio is sampled by pressure changes in the air. 
* These changes are measured by a device to produce discrete values at certain time intervals
* 1D sequence over time
* 1D -> 2D, $$R^1 \to R^2$$, fourier transformation.
  - A fuzzier perspective (to a degree) over time
  - Image space where left to right carry different meaning than right to left; information is not symmetric about the frequency-axis
* Samples - Spectrogram
    - Frequenciy Intensity over Time
    - Human hearing $$20-20KHz$$
    - 16Khz for speech intuitevely seems plenty to have conversations
    - 8KHz is arguably enugh but the quality degregation is at least annoying. Too much noice is
        difficult to encode
* more complex the content of the information in the audio the more other capabilities more than audio quality is needed. For simpler speech activities I would argue 4Khz is mostly enough even though here you might be bothered (and lose concentration) by the audio quality.


On Deepminds [blog](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) they show this gif zooming in on an audio waveform.
<img width='200' src="https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig1-Anim-160908-r01.gif" alt="" align='middle'> 
A sequence of intensity over time.  A [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform)
decomposes the signal in to a frequency domain representation, precicely a sum over frequencies and their magnitude. A linear set of
frequencies played with different magnitudes.
<iframe width="200" src="https://www.youtube.com/embed/spUNpyF58BY?ecver=1" frameborder="0" allow="autoplay; encrypted-media" align='middle' allowfullscreen> All possible sounds span sound space  </iframe> 
<iframe width="200" src="https://s3.envato.com/h264-video-previews/87821136-dccf-4270-a5a3-57f6cca7fde8/20404750.mp4" align='middle' frameborder="90" allow="autoplay; encrypted-media" allowfullscreen></iframe>


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
    <img src="/images/turntaking/tt_analysis/nobody_understands_quantum_mechanics_waveform.png" alt="ALL" style='height: 240px'>
  </div>
  <div class="column">
    <img src="/images/turntaking/tt_analysis/nobody_understands_quantum_mechanics_spec.png" alt="ALL" style='width: 100%'>
  </div>
  <div class="column">
    <video width="100%" controls loop>
      <source src="/images/turntaking/tt_analysis/feynman_spectrum.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>


# Datasets

#### Parameters

| Feature:          | Value     |
| :---------------- | :-------- |
| Sample Rate:      |   8000 Hz |
|    max frequency: | 4000Hz (Sample rate // 2) |
|   step size:      |   50ms |
| window size:      |   50ms |
|      n mels:      |     80 |


<div class='row'>
  <div class='columns'>
    <b>Switchboard</b>
    <img src="/images/turntaking/tt_analysis/spectrum_swb.png" alt="ALL" style='flex: 50%; width: 90%'>
  </div> 
  <div class='columns'>
    <b>Maptask</b>
    <img src="/images/turntaking/tt_analysis/spectrum_maptask.png" alt="ALL" style='flex: 50%; width: 91%'>
</div> 
</div>



<div class='row'>
  <div class='columns'>
  <b>Single Dialog Speaker 0</b>
  <img src="/images/turntaking/tt_analysis/Melspectrum_Single_Speaker_0_(all_vs_vad_vs_novad).png" alt="Channel 0" style='flex: 50%; width: 91%'>
  </div> 
  <div class='columns'>
    <b>Single Dialog Speaker 1</b>
    <img src="/images/turntaking/tt_analysis/Melspectrum_Single_Speaker_1_(all_vs_vad_vs_novad).png" alt="Channel 1" style='flex: 50%; width: 91%'>
</div> 
</div>

### Intensity

### Pitch Semitone

### Words
  - Glove Embedding
    - Avg
  - Stop Word rate

Output Space
- VAD
- Turn-taking Events


## Conversation & Prosody

<button onclick="nonSense()"> The Nonsensicality of Prosody with seemingly random spoken information.  </button>


<div id="nonSense">
  <li> The Books makes cool art.  </li>
  <blockquote>
    "It is a compendium on mini CD of four pieces created for the "1%" art and sound installation in the Ministry of Culture in Paris, France in 2004."
  </blockquote>
  <blockquote>
    I laughed really hard one night at around 22.30 in front of my computer
    "Can you shut the door!"
  </blockquote>
  <iframe src="https://open.spotify.com/embed/track/6q6EXUhoujkQCkcAwe1jEK" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
  Music For a French and Other short format oddities, The Books, 2004 [pitchfork article](https://pitchfork.com/reviews/albums/857-music-for-a-french-elevator/)
  We hear some conjunctions "and", "or", some other "not totally random" words but mostly it is
  numbers. (min - max)

  [Wikipedia](https://en.wikipedia.org/wiki/Music_for_a_French_Elevator_and_Other_Short_Format_Oddities_by_the_Books)
  Wikipedia contributors. "Music_for_a_French_Elevator_and_Other_Short_Format_Oddities_by_the_Books" Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 20 Dec. 2019. Web. 20 Dec. 2019.
</div>

