---
layout: post
author: Erik
---


Exp


<!--more-->


## Introduction
* States of Turn-Taking
* Distribution over states and datasets
    - MT
    - SWB
    - Robot
* Likelihood training
* Continuous, Predictive, Autoregressive
* Representation

<center>
<div class="row">
  <div class="column">
    <center><h3> States of Turn-Taking </h3></center>
    <img src="/images/turntaking/experiment1/labels.png" alt="ALL" style='width: 100%'>
  </div>
</div>
</center>

<center>

<b> State distribution in datasets </b>

<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_states_10ms_frames.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/labels/class_states_10ms_frames_dset.png" alt="ALL" style='flex: 50%; width: 99%'>
  </div>
</div>
</center>


## 




### Experiment 1


We are interested in states so lets try and learn those directly


Learn $$ P(s_t | s_{t>}) $$ where $$ s_t \in S_v, S_e $$ where $$S_v$$, $$S_v$$ is the vad and
Edlund classes respectively.


## Vad States 

1. Learn by predicting the next state given previous -> a Turn-Taking "Language" Model.
2. Learn bypredicting the next $$N_h$$ frames, where $$h$$ is the horizon, the number of frames to predict in the future
  - Assumption: IID for all the outputs
  - Otherwise combine bits -> n frames -> 2^n =

<center>
<b>Loss and Accuracy</b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/exp1_loss_acc.png" alt="ALL" style='width: 80%'>
  </div>
</div>
</center>

<center>
<b>Fscore</b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/exp1_all_fscore.png" alt="ALL" style='width: 80%'>
  </div>
</div>
</center>

<center>
<b> Weighted Fscore</b>
<div class='row'>
  <div class='columns' style='flex: 100%'>
    <img src="/images/turntaking/experiment1/exp1_fscore_weight.png" alt="ALL" style='width: 50%'>
  </div>
</div>
</center>


### Inference

Generation of turntaking from the red line. All samples are from the same audio. 
<center>
<b> 1 Prediction Frames </b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/pred_frames_1/0_Vad_T1_greedyFalse_pred_frames1_a.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_1/0_Vad_T1_greedyFalse_pred_frames1_b.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_1/0_Vad_T1_greedyFalse_pred_frames1_c.png" alt="ALL" style='flex: 50%; width: 80%'>
  </div>
</div>
</center>

<center>
<b> 5 Prediction Frames </b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/pred_frames_5/0_Vad_T1_greedyFalse_pred_frames5_a.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_5/0_Vad_T1_greedyFalse_pred_frames5_b.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_5/0_Vad_T1_greedyFalse_pred_frames5_c.png" alt="ALL" style='flex: 50%; width: 80%'>
  </div>
</div>
</center>

<center>
<b> 10 Prediction Frames </b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/pred_frames_10/0_Vad_T1_greedyFalse_pred_frames10_a.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_10/0_Vad_T1_greedyFalse_pred_frames10_b.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_10/0_Vad_T1_greedyFalse_pred_frames10_c.png" alt="ALL" style='flex: 50%; width: 80%'>
  </div>
</div>
</center>


### Contextual Vad States (Edlund)

* None -> Pauses, Gaps
* Both -> overlap (within, between)
* 5 states 2.32 bits, 6 states 2.58 bits (correct?)


<center>
<b>Loss and Accuracy</b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/" alt="ALL" style='width: 80%'>
  </div>
</div>
</center>

<center>
<b>Fscore</b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/" alt="ALL" style='width: 80%'>
  </div>
</div>
</center>

<center>
<b> Weighted Fscore</b>
<div class='row'>
  <div class='columns' style='flex: 100%'>
    <img src="/images/turntaking/experiment1/" alt="ALL" style='width: 50%'>
  </div>
</div>
</center>


### Inference

Generation of turntaking from the red line. All samples are from the same audio. 
<center>
<b> 1 Prediction Frames </b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/pred_frames_1/0_Vad_T1_greedyFalse_pred_frames1_a.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_1/0_Vad_T1_greedyFalse_pred_frames1_b.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_1/0_Vad_T1_greedyFalse_pred_frames1_c.png" alt="ALL" style='flex: 50%; width: 80%'>
  </div>
</div>
</center>

<center>
<b> 5 Prediction Frames </b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/pred_frames_5/0_Vad_T1_greedyFalse_pred_frames5_a.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_5/0_Vad_T1_greedyFalse_pred_frames5_b.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_5/0_Vad_T1_greedyFalse_pred_frames5_c.png" alt="ALL" style='flex: 50%; width: 80%'>
  </div>
</div>
</center>

<center>
<b> 10 Prediction Frames </b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/pred_frames_10/0_Vad_T1_greedyFalse_pred_frames10_a.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_10/0_Vad_T1_greedyFalse_pred_frames10_b.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_10/0_Vad_T1_greedyFalse_pred_frames10_c.png" alt="ALL" style='flex: 50%; width: 80%'>
  </div>
</div>
</center>



## Experiment 4


Generate higher level representation of Conversation.


#### Causal VQVAE2

* Listen and compare the reconstructed audio with gt
* Low pass filter -> listen and compare
* Evaluate performance on generated melspec vs ground truth on Experiment 2
* Learn the prior over latent codes (algorithm 2)
  * How does the generated differ from the ground truth. Nonsense?

* Improvements
    - Learn a joint higher level code for a dialog
        - Condition the top+bottom codes on this conversational code
        - Look at generated dialogs
    - Condition the model on text tokens

<center>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/tt_cnn/vqvae2/vqvae_codebook_30swb.png" alt="ALL" style='width: 80%'>
    <img src="/images/turntaking/tt_cnn/vqvae2/vqvae_loss_30swb.png" alt="ALL" style='width: 80%'>
  </div>
</div>
</center>

<center>
<b> Causal VQVAE2 Algorithm 1 </b>

<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/tt_cnn/vqvae2/codebook_1.png" alt="ALL" style='width: 80%'>
    <img src="/images/turntaking/tt_cnn/vqvae2/codebook_2.png" alt="ALL" style='width: 80%'>
    <img src="/images/turntaking/tt_cnn/vqvae2/codebook_3.png" alt="ALL" style='width: 80%'>
    <img src="/images/turntaking/tt_cnn/vqvae2/codebook_4.png" alt="ALL" style='width: 80%'>
    <img src="/images/turntaking/tt_cnn/vqvae2/codebook_5.png" alt="ALL" style='width: 80%'>
    <img src="/images/turntaking/tt_cnn/vqvae2/codebook_6.png" alt="ALL" style='width: 80%'>
    <img src="/images/turntaking/tt_cnn/vqvae2/codebook_7.png" alt="ALL" style='width: 80%'>
    <img src="/images/turntaking/tt_cnn/vqvae2/codebook_8.png" alt="ALL" style='width: 80%'>
    <img src="/images/turntaking/tt_cnn/vqvae2/codebook_9.png" alt="ALL" style='width: 80%'>
    <img src="/images/turntaking/tt_cnn/vqvae2/codebook_10.png" alt="ALL" style='width: 80%'>
  </div>
</div>
</center>




