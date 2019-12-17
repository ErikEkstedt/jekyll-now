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


## Transfer Learning


What is transfer learning? How can we achieve transfer learning? 

#### Basics

* Learn a model $$ M(\theta) $$
* With data from dataset A, $$p_A$$
* Evaluate model on dataset B, $$p_B$$

If dataset A and dataset B is represented by a set of vectors, discrete data samples, we may
approximate their distribution, a general (single? i.e not as the set of the specific data samples
but something more specific) descriptor for the set, through some definition. 

Given a defined approximator functions we may define a similarity metric to compare their likeness.
Given an approximator yielding a probability density function we may use a probability density
distance metric like the Kullback-Leibler Distance, $$ D_{KL}(p_{data, a} || p_{data, b}) $$.

* What is a good way to approximate $$ p_n, \forall n $$ ?
* $$ P_A = $$ ?
* $$ P_B = $$ ?
* What similarity metric to use?
  - KLD ?
  - $$ D = SimilarityMetric(P_A, P_B) $$.


#### Train, Val & Test

Given a similarity metric, D, we may say that if D, the distance between a and b is small, they are
more likely from similar underlying distributions. Meaning that learning A teaches us a lot about B. 

The types of distances in a similarity metric define the categories we label them as. A small D
is used during the training process, the validation split. The train and validation datasets
should have very small D.

$$ P_{train}, P_{validation}, P_{test} \in P_{training}  $$

thus the Distance D between any of $$ P_{train}, P_{validation}, P_{test} $$ is small. Given a
sufficiently small distance we refer to this paradigm as the training setup. This is how we show
that our models learn useful/general features.

#### Finetuning and Transfer Learning

If we make the distance, D, larger there is somewhere a shift in the concepts we talk about and we
use words as classes or tasks. For example you can say that learning task A, running forward, is
similar to task B, running backwards, but there is some distinction between the two. The same
underlying knowledge of the world is probably required but is used in slightly different ways.
Another example would be to learn to recognize, lets say 1000 different classes in images, Task A
but then we would like to classify 20 other classes Task B, we could use the model that does task A
well and give it some time to train on Task B. This is finetuning. Learn a model on some task, then
give it newer data and say "Ms model, you already know alot about seeing but please learn some
subtle ways to utilize your previous knowledge on these new datasets".



* Training set $$ P_{training} = \{P_{train}, P_{val}, P_{test}\} $$
* Finetune set $$ P_{finetune} = \{P_{fine-train}, P_{fine-val}, P_{fine-test}\} $$
* Learn a model $$ M_{0}(\theta;  P_{training} ) $$
* Train on a small sample from $$ P_{finetune} $$
* Evaluate on a larger portion of $$ P_{finetune} $$
  * Keep track on metrics (loss/acc) on $$ P_{training} $$ to see how much the model "forgets"
  * How well does the transfer setup compared to training a Model $$ M_{1}(\theta;  P_{finetune} ) $$ directly.
  * Do the models yield similar results?
  * Do their behaviour differ in some non-arbitrary ways?


------------------------------

### TODO
1. Evaluataion method for transfer learning.
  - Annotated robot labels. Exactly where the segment ends (previous work)
  - If robot speaks during the label frames -> zero the robot voice
  - Annotated Robot Dataset contains 10 speakers. Each with 4-5 dialogs.
    - Train on 1 dialog, validate on 1 dialog, get test scores for the rest.
    - Average over all speakers
2. Learn a model from scratch on the Annotated Robot Dataset 
3. Investigate the features the model learns from.
  - Does any features work better for transfer learning than others?
4. Improve finetuning
  - Update only on gradients based on user behavior
  - Only update the word model
  - Only last layer vs the entire model

##### Annotated Robot 

|                    Model Type | Dataset (Score) | Eval Dataset (Score) | Finetune Speaker Score Avg | Finetune Speaker Score Std |
|                    :--------- | :-------         | :------- | ------ | :------                    |
|                     Intensity | SWB (--)              | MapTask (--) |     -- | --                         |
|           Intensity + Prosody | SWB (--)              | MapTask (--) |     -- | --                         |
|                          MFCC | SWB (--)              | MapTask (--) |     -- | --                         |
|                       MelSpec | SWB (--)              | MapTask (--) |     -- | --                         |
|             Melspec + Prosody | SWB (--)              | MapTask (--) |     -- | --                         |
|     Melspec + Prosody + Words | SWB (--)              | MapTask (--) |     -- | --                         |
|               Melspec + Words | SWB (--)              | MapTask (--) |     -- | --                         |
|           Melspec + StopWords | SWB (--)              | MapTask (--) |     -- | --                         |
| Melspec + StopWords + prosody | SWB (--)              | MapTask (--) |     -- | --                         |
|                    :--------- | :-------         | :---------------- | ------ | :------                    |
|                     Intensity | MapTask (--)          | SWB (--) |     -- | --                         |
|           Intensity + Prosody | MapTask (--)          | SWB (--) |     -- | --                         |
|                          MFCC | MapTask (--)          | SWB (--) |     -- | --                         |
|                       MelSpec | MapTask (--)          | SWB (--) |     -- | --                         |
|             Melspec + Prosody | MapTask (--)          | SWB (--) |     -- | --                         |
|     Melspec + Prosody + Words | MapTask (--)          | SWB (--) |     -- | --                         |
|               Melspec + Words | MapTask (--)          | SWB (--) |     -- | --                         |
|           Melspec + StopWords | MapTask (--)          | SWB (--) |     -- | --                         |
| Melspec + StopWords + prosody | MapTask (--)          | SWB (--) |     -- | --                         |


---------------------------------

## Implementation
What datasets do we have access to? see [Turn-Taking Datasets](/Turn-Taking-Datasets/index.html)
#### H-H -> H-H , Swb -> Maptask

One thing to test would be to train on a corpora of casual spoken dialog
(switchboard) and see how well it performs on a corpora with a specific
task/setup (maptask)


#### H-H (SWB) -> H-R (Annotated (situated) Robot Maptask)


<video width="100%" controls>
  <source src="/images/turntaking/tt_finetune/4_grid_vids.mp4" type="video/mp4">
  Your browser does not support HTML5 video.
</video>

* Use sample dialogs to train on and just finetune on this new dataset
  - How much is forgotten? Loss on original training data?
  - Freeze the weights of the network and retrain the output layer
* Train specifically to easily learn new tasks
  - MAML



