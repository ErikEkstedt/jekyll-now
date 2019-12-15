---
layout: post
---


Given that we have a sufficiently capable model for turn-taking in open ended, casual
conversation, can we train that model on particular setups to improve the specific
capability on that task?

One important aspect of this work is to (make better) robots and their
turntaking behavior. A common approach to turn-taking in robots has been to
listen to audio, determine if the user is speaking, after the user stops
speaking wait for a certain duration of time and if the user does not speak
during that time initiate an utterance. This is a very simple approach and often
lead to feelings of uncertainties about if the robot actually heard what the
user said. On top of this it could lead to both robot and users starts talking
at about the same time, a bad conversational organization moment.

What if we could train our turn-taking model on human-human dialog and then
transfer it to robot-human dialogs. Could we get a policy that would be more
fluid and take the turns mode appropriately?


<!--more-->


## Finetuning




What is transfer learning? How can we achieve transfer learning? 

#### Basics

* Learn a model $$ M(\theta) $$, with data from dataset a, $$p_a$$, evaluate model on dataset b,
    $$p_b$$
* Test on data from dataset b
    - Similar results are good

If dataset a and dataset b is represented by number sets constructed of data samples, they are
discrete, we may approximate their distribution through some definition. Given those defined
approximator functions we can compare their likeness. 

* What is a good way to approximate $$ p_n $$
* $$P_A = $$



A popular metric for defining distance
relationships between probability densities is the Kullback-Leibler Distance, $$ D_{KL}(p_{data, a} || p_{data, b}) $$


Given such a metric, lets refer to it as D, we may say that if D, the distance between a and b is
small, they are more likely from similar underlying distributions. Learning a teaches us a lot
about b. 

The types of distances in a similarity metric define the categories we label them as. A small D
is used during the training process, the validation split. The train and validation datasets
should have very small D.

If we make the distance longer there is somewhere a shift in the concepts we talk about and we
use words as classes or tasks. You learn data distributions that depends on similar underlying
smaller concepts or patterns. 


* Finetuning is the concept of training on dataset ab, then for a dataset with sufficiently large
    D with respect to dataset ab


Well the most straightforward
implementation would be to train on a corpora and then test on another after
which we train on a subset of the new corpora and see if we can better our
performance.


### H-H -> H-H , Swb -> Maptask


One thing to test would be to train on a corpora of casual spoken dialog
(switchboard) and see how well it performs on a corpora with a specific
task/setup (maptask)


### H-H -> H-R , Swb -> Robot Maptask


<video width="100%" controls>
  <source src="/images/turntaking/tt_finetune/4_grid_vids.mp4" type="video/mp4">
  Your browser does not support HTML5 video.
</video>

* Use sample dialogs to train on and just finetune on this new dataset
  - How much is forgotten? Loss on original training data?
  - Freeze the weights of the network and retrain the output layer
* Train specifically to easily learn new tasks
  - MAML
