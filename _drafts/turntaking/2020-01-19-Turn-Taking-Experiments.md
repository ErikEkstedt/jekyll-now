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
    <center><h3> Dialog State Chromogram </h3></center>
    <img src="/images/turntaking/chromogram/chromogram_dialog.png" alt="ALL" style='width: 100%'>
  </div>
</div>
</center>

-------------------------------

<center>
<div class="row">
  <div class="column" style='flex: 60%'>
    <center><h3> Dialog Segment Chromogram </h3></center>
    <img src="/images/turntaking/chromogram/chromogram_segment.png" alt="ALL" style='width: 90%'>
  </div>
  <div class="column" style='flex: 40%'>
    <center><h3> Dialog State Histogram </h3></center>
    <img src="/images/turntaking/chromogram/chromogram_hist.png" alt="ALL" style='width: 100%'>
  </div>
</div>
</center>


-------------------------------

<center>
<b> State distribution in datasets </b>
<div class='row'>
  <div class='columns'>
    <div class='row'>
      <img src="/images/turntaking/chromogram/labels_swb.png" alt="ALL" style='flex: 50%; width: 100%'>
    </div>
  </div>
  <div class='columns'>
    <div class='row'>
      <img src="/images/turntaking/chromogram/labels_maptask.png" alt="ALL" style='flex: 50%; width: 100%'>
    </div>
  </div>
  <div class='columns'>
    <div class='row'>
      <img src="/images/turntaking/chromogram/labels_robot.png" alt="ALL" style='flex: 50%; width: 100%'>
    </div>
 </div>
</div>

</center>

-------------------------------

<center>

<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/chromogram/swb_dur_label_hist.png" alt="ALL" style='width: 100%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/chromogram/maptask_dur_label_hist.png" alt="ALL" style='width: 100%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/chromogram/robot_dur_label_hist.png" alt="ALL" style='width: 100%'>
 </div>
</div>
</center>


-------------------------------

<center>
<h1> Turns </h1> 
</center>


<center>
<b> Turn Pair Sample (good) from Switchboard</b>
<img src="/images/turntaking/turns/turn_pair_sample_swb_good.png" alt="ALL" style='width: 100%'>

<h2> Turn Pair Sample (bad?) from Switchboard</h2>

<b>Starting where both speakers are active </b>

<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/turns/turn_pair_sample_swb_bad.png" alt="ALL" style='width: 100%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/turns/turn_pair_sample_swb_bad2.png" alt="ALL" style='width: 100%'>
  </div>
</div>

<b>Inculde entire system turn as context?</b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/turns/turn_pair_sample_swb_bad_system_turn.png" alt="ALL" style='width: 100%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/turns/turn_pair_sample_swb_bad_system_turn2.png" alt="ALL" style='width: 100%'>
  </div>
</div>
</center>

<ul>
  <li>Start of turn pair</li>
    <ul>
      <li>Start from last system turn end? </li>
      <li>Start from beginning of current user turn? </li>
      <li>Should turn pairs beginning with both be omitted?</li>
    </ul>
  <li>End of turn pair</li>
    <ul>
      <li>Include entire system turn?</li>
      <li>Include first system ipu?</li>
    </ul>
</ul>

<center>
<h3>Histogram over Turn Offset</h3>
<div class='row'>
  <div class='columns'>
    <div class='row'>
      <img src="/images/turntaking/turns/turn_pair_offset_hist_line_swb.png" alt="ALL" style='flex: 50%; width: 100%'>
    </div>
  </div>
  <div class='columns'>
    <div class='row'>
      <img src="/images/turntaking/turns/turn_pair_offset_hist_line_mt.png" alt="ALL" style='flex: 50%; width: 100%'>
    </div>
  </div>
  <div class='columns'>
    <div class='row'>
      <img src="/images/turntaking/turns/turn_pair_offset_hist_line_robot.png" alt="ALL" style='flex: 50%; width: 100%'>
    </div>
  </div>
</div>

<div class='row'>
  <div class='columns'>
    <div class='row'>
      <img src="/images/turntaking/turns/turn_pair_offset_hist_line_swb_mt.png" alt="ALL" style='flex: 50%; width: 100%'>
    </div>
  </div>
  <div class='columns'>
    <div class='row'>
      <img src="/images/turntaking/turns/turn_pair_offset_hist_line_all.png" alt="ALL" style='flex: 50%; width: 100%'>
    </div>
  </div>
</div>

<b>Omit Both</b>
<div class='row'>
  <div class='columns'>
    <div class='row'>
      <img src="/images/turntaking/turns/turn_pair_offset_hist_line_swb_mt_omit_both.png" alt="ALL" style='flex: 50%; width: 100%'>
    </div>
  </div>
  <div class='columns'>
    <div class='row'>
      <img src="/images/turntaking/turns/turn_pair_offset_hist_line_all_omit_both.png" alt="ALL" style='flex: 50%; width: 100%'>
    </div>
  </div>
</div>
</center>


-------------------------------

Where are interesting places to evalutate/train the loss?

<ul>
  <li>End of IPUs</li>
  <li>Mutual Silences</li>
  <li>Turn Clean moments</li>
  <li>Turn Overlap moments</li>
</ul>

<center>
<img src="/images/turntaking/events/events.png" alt="ALL" style='flex: 50%; width: 80%'>
</center>

## Vad activity prediction

<img src="/images/turntaking/model_prediction/OUT_vad_IN_pitch_mfcc_vad.png" alt="ALL" style='width: 100%'>
<img src="/images/turntaking/model_prediction/OUT_vad_IN_pitch_mfcc_vad_2.png" alt="ALL" style='width: 100%'>



### Questions

How to decide if a turn prediction is true or false? 

The ground truth average is shown in black segmented lines and the model prediction in the colors of
corresponding speaker. At a turn event the average ground truth predictions do not match the actual
action in the conversation. In the image below we show correct guesses with green points and are
defined correct if they follow the ground truth (averagem ground truth vs average model prediction).


--------------------------

## Vad and Vad state prediction 3 seconds

<center>
<div class='row'>
  <div class='columns' style='width: 50%'>
    <h4>Vad Prediction: sw2264</h4>
    <video src="/images/turntaking/model_prediction/sw2264_vad_prediction.mp4" height="" width="100%" type='video/mp4' preload="auto" controls='loop' autoplay="autoplay"></video>
    <ul>
      <li>0:17 - short backchannel -> no tt -> breath -> tt </li>
      <li>0:17 - short backchannel -> no tt -> breath -> tt </li>
    </ul>
  </div>
  <div class='columns' style='width: 50%'>
    <h4>Vad Prediction: sw2379</h4>
    <video src="/images/turntaking/model_prediction/sw2379_vad_prediction.mp4" height="" width="100%" type='video/mp4' preload="auto" controls='loop' autoplay="autoplay"></video>
  </div>
</div>

<div class='row'>
  <div class='columns' style='width: 50%'>
    <h4>Vad Classes Prediction: sw2264</h4>
    <video src="/images/turntaking/model_prediction/sw2264_vad_state_prediction.mp4" height="" width="100%" type='video/mp4' preload="auto" controls='loop' autoplay="autoplay"></video>
  </div>
  <div class='columns' style='width: 50%'>
    <h4>Vad Classes Prediction: sw2379</h4>
    <video src="/images/turntaking/model_prediction/sw2379_vad_state_prediction.mp4" height="" width="100%" type='video/mp4' preload="auto" controls='loop' autoplay="autoplay"></video>
  </div>
</div>
</center>


----------------------


<h2>Sample Training</h2>

<center>
<div class='row'>
  <div class='columns' style='width: 50%'>
    <img src="/images/turntaking/model_prediction/training/vad_acc.png" alt="ALL" style='width: 100%'>
  </div>
  <div class='columns' style='width: 50%'>
    <img src="/images/turntaking/model_prediction/training/vad_states_confusion.png"  alt="ALL" style='width: 100%'>
  </div>
</div>

<img src="/images/turntaking/model_prediction/training/vad_states_acc.png" alt="ALL" style='width: 100%'>
<img src="/images/turntaking/model_prediction/training/vad_acc_loss.png" alt="ALL" style='width: 100%'>
<img src="/images/turntaking/model_prediction/training/vad_states_extra_loss.png"  alt="ALL" style='width: 100%'>
<img src="/images/turntaking/model_prediction/training/vad_vad_states_loss.png"  alt="ALL" style='width: 100%'>

</center>


--------------------------------------------



<img src="/images/turntaking/model_prediction/TT_decision_trouble_vad_prediction.png"  alt="ALL" style='width: 100%'>


## Vad classes prediction

<img src="/images/turntaking/model_prediction/OUT_vad-classes_IN_pitch_mfcc_vad-states.png" alt="ALL" style='width: 100%'>
<img src="/images/turntaking/model_prediction/OUT_vad-classes_IN_pitch_mfcc_vad-states_2.png" alt="ALL" style='width: 100%'>




-------------------------------

### Global Labels

<video src="/images/turntaking/chromogram/global/video_labels_test.mp4" height="" width="100%" type='video/mp4' preload="none" controls='loop' autoplay="autoplay"></video>


-------------------------------
### Experiment 1
We are interested in states so lets try and learn those directly


---------------
**Learn the single frame prediction:**

$$ P(s_t | s_{t>}) $$ 

where $$ s_t \in S_v, S_e $$ where $$S_v$$, $$S_v$$ is the vad and Edlund classes respectively. Well defined conditional probability distribution.

--------------

**Learn mulitple frames prediction:**

$$ P(s'_t | s_{t>}), s'_t = \{s_t, s_{t+1}, ..., s_{t+h} \} $$ 
where $$h$$ is the number of future frames to predict. Assumption: IID for all the outputs, The different output frames are not dependent on the previous output frame in each given future prediction.

--------------

**Learn single label multiple frame prediction:**

$$ P(y_t | s_{t>}), y_t \in Y $$
where Y is constructed by combining $$\{s_0, s_1,...,s_h\}$$ consecutive frames into $$ 2^h$$ distinct classes.

--------------

## Vad States 
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

The plots above at least looks like they generate the same distribution as the data. However, when
predicting the next frame the current frame contains alot of information about the next. Thus it is
"easy" for a model to exploit the local smoothness of the conversational state and (almost) alwasy
guess the current value as the prediction value. Most of the states continue for a "longer" range
and so it is only at the state shifts that the distribution is "hard" to learn.

What kind of distribution do we get if we start from all the different starting states (4) and
generate for 100 steps? Lets do this for the temperature sampling approach (as above) and with a
greedy approach. What kinds of distributions do we get?

<center>
<div class='row'>
  <div class='columns' style='flex:33%'> <center> <b>Greedy (1000 steps)</b> </center> </div>
  <div class='columns' style='flex:33%'> <center> <b>Temp 1 sampling (100 steps)</b> </center> </div>
  <div class='columns' style='flex:33%'> <center> <b>labels (tmp swb root)</b> </center> </div>
</div>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/pred_frames_1/initial_states_greedy_100frames.png" alt="ALL" style='flex: 50%; width: 97%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/pred_frames_1/initial_states_Temp1_100frames.png" alt="ALL" style='flex: 50%; width: 97%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/pred_frames_1/train_label_distribution_temp.png" alt="ALL" style='flex: 50%; width: 97%'>
  </div>
</div>
</center>

Here we see that with temperature sampling we get a similar distribution but while doing a greedy
sampling we don't. The greedy approach kept on predicting the current state for each of the initial
states.


Lets pass a dialog segment through the network and see how the probabilities and likelihood (the
probability given for the actual answer) vary across the dialog.

<center>
<b> 1 Prediction Frames. Probs </b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/experiment1/pred_frames_1/dialog_probs_a.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_1/dialog_probs_b.png" alt="ALL" style='flex: 50%; width: 80%'>
    <img src="/images/turntaking/experiment1/pred_frames_1/dialog_probs_c.png" alt="ALL" style='flex: 50%; width: 80%'>
  </div>
</div>
</center>

Here we see the behavior we were worried about. The network is really biased to predict the same as
the current frame and only occasionally, e.g during silences, do we see any other behavior. At the
state shift in the dialog we see that the likelihood almost drop to zero and then instantly jump
right back up (we are back inside a stable state interval)


In the paper [An Unsupervised Autoregressive Model for Speech Representation
Learning](https://arxiv.org/pdf/1904.03240.pdf), which is based on [Contrastive Predictive
Coding](https://arxiv.org/pdf/1807.03748.pdf), they mitigate this problem by instead prediction the
frame $$n$$ steps into the future. Albeit they do this on the spectrograms frames (spectrums) directly.

Lets train a network to output the prediction for the frame n steps from now, that is 

$$ P(s_t+n | s_{t>})$$



-----------------------------------------

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




# Training


## Vad global activity prediction

### Experiment 3: CPC
Learning the global encoding might be a useful way partly to circumvent defining the future state of
the conversation by hand but also that the model might learn something useful which we did not think
about, or maybe just something useful in the space of what is possible to learn given the problem
formulation (input data, model parameters, ..., etc)


<center>
<b> Epoch 69 Training on 4 MFCC/channel to cpc_dim: 32</b>
<div class='row'>
  <div class='columns'>
    <img src="/images/turntaking/cpc/cpc_loss_ep69.png" alt="ALL" style='flex: 30%; width: 80%'>
  </div>
  <div class='columns'>
    <img src="/images/turntaking/cpc/mfcc4_to_cpc_120-32.png" alt="ALL" style='flex: 80%; width: 80%'>
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




