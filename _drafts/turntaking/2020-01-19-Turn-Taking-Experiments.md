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

### Experiment 1


We are interested in states so lets try and learn those directly


Learn $$ P(s_t | s_{t>}) $$ where $$ s_t \in S_v, S_e $$ where $$S_v$$, $$S_v$$ is the vad and
Edlund classes respectively.


## Training 

1. Learn by predicting the next state given previous -> a Turn-Taking "Language" Model.
2. Learn bypredicting the next $$N_h$$ frames, where $$h$$ is the horizon, the number of frames to predict in the future

<center>
<b>Loss and Accuracy</b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/exp1_loss_acc.png" alt="ALL" style='width: 80%'>
  </div>
</div>
</center>

<center>
<b> Weighted Fscore</b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/exp1_fscore_weight.png" alt="ALL" style='width: 80%'>
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

