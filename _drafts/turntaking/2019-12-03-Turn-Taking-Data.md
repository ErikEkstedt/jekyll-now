---
layout: post
author: Erik
---

## Data: Real, dataset, model and output

What kinds of data are we going to work with? How to process this data? What do we wish to achieve


<!--more-->

# Data

<img src="/images/turntaking/turntaking.png" alt="ALL">


## Conversation

* Audio: spoken language
    - 1D samples over time (fast)
      - 2D Frequency representation over time (slower)
          - Signal -> sum of frequencies
          - Lossy Nyquist
    - Prosody
        - duration
        - intensity
        - pitch
    - phonemes (slower still)
* Text:  written language
    - phonemes
    - words
    - phrases
    - utterances / turns


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

Audio is represented digitally as waveforms, intensity samples over time, as seen below (source: [Deepmindblog](https://deepmind.com/blog/wavenet-generative-model-raw-audio/))

<center>
<img width='400' src="https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig1-Anim-160908-r01.gif" alt="" align='middle'>
</center>
A [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform) decomposes the signal in to a
frequency domain representation, precicely a sum over frequencies and their magnitude. A linear set
of frequencies with different magnitudes.

<center>
<iframe height="200" src="https://www.youtube.com/embed/spUNpyF58BY?ecver=1" frameborder="0" allow="autoplay; encrypted-media" align='middle' allowfullscreen>
</iframe>
<iframe height="200" src="https://s3.envato.com/h264-video-previews/87821136-dccf-4270-a5a3-57f6cca7fde8/20404750.mp4" align='middle' frameborder="90" controls='loop' allow="autoplay; encrypted-media" allowfullscreen>
</iframe>
</center>


## Acoustic Information

Spectrogram Fourier transform of 1D audio (intensity) signal -> 2D frequency/power space.
Step time: 10ms (step length of fourier transform),  window time: 50ms (window size of fourier transform)


<div class="row">
  <div class="column" style='flex: 45%'>
    <center><h3> Waveform </h3></center>
    <img src="/images/turntaking/tt_analysis/nobody_understands_quantum_mechanics_waveform.png" alt="ALL" style='width: 100%'>
  </div>
  <div class="column" style='flex: 5%'>
    <center>
      <span style='font-size:100px;'>&#8594;</span>
    </center>
  </div>
  <div class="column" style='flex: 45%'>
    <center><h3> MelSpectrogram</h3></center>
    <img src="/images/turntaking/tt_analysis/nobody_understands_quantum_mechanics_spec.png" alt="ALL" style='height: 200px'>
  </div>
</div>


<center>
<div class="row">
<div class="column">
<video width="100%" controls>
<source src="/images/turntaking/tt_analysis/feynman_spectrum.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>
</div>
</div>
</center>

> "Nobody understands quantum mechanics"* - Richard Feynman 1964


### Mel-Spectrogram

A melspectrogram is a constructed from a frequency power spectrogram from an audio signal mapped onto a mel space.
The mel mapping maps frequencies into bins more akin to how humans seem to experience audio.

A regular power spectrogram over 3 seconds of spoken audio -> mapped onto the mel space

#### Spectrum

The power for each frequency band in a single frame is a spectrum. We note that the lower
frequencies are more distinguishable in the mel space than in the power spectrogram. Important
frequencies for speech such as the fundamental frequency F0 and its harmonics F1, F2 are given more
importance in the mel space. The higher frequencies (noise) is less prominent.

In Deep Learning it is very common to process melspectrograms, for speech, instead of the original power specotrogram.

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
* Total Vocabulary over datasets
* top 100 words 
* Define stopwords ~100-300
* Distribution of stopwords in dialogs
* Stop Word rate
* Glove Embedding
  * Avg


### Conversation

* Taking/Yielding Turns
* Heuristics
  * Minimize silence ?
  * Minimize talking over each other ?
  * Convey/Recieve information ?
* Understanding and Coordinating
  - Dance - timing
  - Beats - duration
  - Music - prosody
  - Why do people like rap/spoken word? Rhymes?


### Dialog

A conversation with only 2 interlocutors. Less possabilities, Less Coordination required.

* What is the context?
    - Games
    - Work
    - Casual
    - Familiarity with the other
* What is being talked about?
* What is the intent of the other? The joint intent of the conversation?
* Dialog Acts/Speech Acts
* Explanations, Questions, Statements
* Sentiment


I predict what you are saying by infering what I think you mean. I simultaniously associate your
meaning to by own based on some convergence between what I intent/want and mean and what I infer
that you intend and mean, given some notion of uncertainty I have about all preceding "objects", and
decide on what actions to implement now or later (perhaps even what I should have implemented
before).

Close friendship: If I want you to feel good and I want you to be happy and I am willing to invest in a certain amount of energy into making that a reality and actually do it, I might like you you might be really close to me.


Lots of associations any given moment. Lucky for us that are brains are highly parallel.

### Brains

* Energy Efficient
* Evolutionary designed
  * Purpose driven (in a sense)
  * Traits that are favourable for gene survival (procreation by proxy), Selfish Gene
* Generates behaviour
  - Sapolsky

The brain is very sparse

* Reptile Brain / Old Brain
* Neocortex (mammals)
    - Homogenous
    - Smallest unit of replication -> cortical columns
        - Excitory cells
        - Inhibitory cells
        - Grid Cells
    - Numenta (Thousand Brain Theory)
* Units of replication
    - Predictive
    - Sparse
    - NOISY <-> Noise robust
      - unreliable
      - chemistry
      - cells and concentrations in biological systems are not perfect... at all.
* Predictability -> Unsupervised Learning
    - Movement
    - Saccades
    - action control
    - Know what you are about to experience given what your are going to do.
    - Generates lots of data
* Dendrites & Axons
* Connectome
* Very many representations possible given the interconnectivety between and the number of neurons
* One representation for each "state" in your life
    - grid cells
    - sparse


### Dataset Dialog Metrics

<center> 

<b>Dialog</b>

<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/data/dialog_duration_all.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/data/speaker_ratio_all.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
</div>

<b>Gaps</b>

<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/data/gaps_ratio_total_all.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/data/gaps_ratio_silence_all.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
</div>

<b>Total Frames, 10ms</b>

<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_states_10ms_frames.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_states_10ms_frames_dset.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
</div>

<b>Total Frames combine overlap, 10ms</b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_5states_1second_10ms_frames.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_5states_1second_10ms_frames.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
</div>

<b>Class Labels, 50ms frames, 1s prediction</b>

<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_labels_1second_50ms_frames.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_labels_1second_50ms_frames_dset.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
</div>


<b>Class Labels, 10ms frames, 1s prediction</b>

<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_labels_1second_10ms_frames.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_labels_1second_10ms_frames_dset.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
</div>


<b>Class Label Segment, 50ms frames, 1s prediction</b>

<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_labels_segment_vad.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
</div>

<b>Class Labels combine overlap, 10ms frames, 1s prediction</b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_5labels_1second_10ms_frames.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_5labels_1second_10ms_frames_dset.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
</div>

<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_5labels_segment_vad.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
</div>

</center> 

## Turns


<center> 
<table style="width:50%">
  <tr>
    <th>Event params</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>ipu_segment_frames </td>
    <td>4</td>
  </tr>
  <tr>
    <td>turn_min_duration </td>
    <td>4</td>
  </tr>
  <tr>
    <td>min_turn_solo </td>
    <td>6</td>
  </tr>
  <tr>
    <td>turn_long_cutoff </td>
    <td>30</td>
  </tr>
  <tr>
    <td>turn_min_gap </td>
    <td>6</td>
  </tr>
  <tr>
    <td>turn_answer_min </td>
    <td>10</td> 
  </tr>
  <tr>
    <td>turn_answer_good_min </td>
    <td>60</td>
  </tr>
  <tr>
    <td>pause_min_length </td>
    <td>10</td>
  </tr>
  <tr>
    <td>pause_continuation_min </td>
    <td>20</td>
  </tr>
</table>
</center> 

<br>

<center>
<div class='row'>
  <div class='columns' style='width: 50%'>
    <img src="/images/turntaking/turns_new/turn_duration_swb.png" alt="ALL" style='width: 100%'>
  </div>
  <div class='columns' style='width: 50%'>
    <img src="/images/turntaking/turns_new/events_swb.png" alt="ALL" style='width: 100%'>
  </div>
</div>

<table style='width: 80%; pad: 5%'>
  <tr>
    <th>Events</th>
    <th>n</th>
  </tr>
  <tr>
    <td>Duration</td>
    <td>107634</td>
  </tr>
  <tr>
    <td>Duration Long</td>
    <td>67998</td>
  </tr>
  <tr>
    <td>Duration Short</td>
    <td>39636</td>
  </tr>
  <tr>
    <td>Gaps</td>
    <td>43197</td>
  </tr>
  <tr>
    <td>Gaps Good</td>
    <td>25547</td>
  </tr>
  <tr>
    <td>Pauses</td>
    <td>70218</td>
  </tr>
</table>

</center>


## Conversation & Prosody

<button onclick="nonSense()"> The Nonsensicality of Prosody with seemingly random spoken information.  </button>


<div id="nonSense">
  <li> The Books makes cool art.  </li>
  <blockquote>
    "It is a compendium on mini CD of four pieces created for the "1%" art and sound installation in the Ministry of Culture in Paris, France in 2004."
  </blockquote>
  <blockquote>
    I laughed really hard one night at around 22.30pm alone in front of my computer
    "Can you shut the door!"
  </blockquote>
  <iframe src="https://open.spotify.com/embed/track/6q6EXUhoujkQCkcAwe1jEK" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
</div>

[Wikipedia](https://en.wikipedia.org/wiki/Music_for_a_French_Elevator_and_Other_Short_Format_Oddities_by_the_Books)

Music For a French and Other short format oddities, The Books, 2004 [pitchfork article](https://pitchfork.com/reviews/albums/857-music-for-a-french-elevator/) We hear some
conjunctions "and", "or", some other "not totally random" words but mostly it is numbers.


