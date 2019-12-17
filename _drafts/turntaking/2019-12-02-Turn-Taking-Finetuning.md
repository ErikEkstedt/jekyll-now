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
* Probabistic Models $$ P_{\theta}(x) $$
    - Evaluate density of a trained model?
    - Is a probabilistic model $$ P_{\theta}(x) $$ trained on maximum likelihood a reflection of the
        dataset?
    - Given the perfect probabilistic model we know the underlying ditribution of the data (up to
        some constant?)


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

The name "finetuning" implies a constraint of how much "tuning" should take place. This brings an
obvious question about what "much" tuning is and how one might measure the amount of tuning. In deep
learning changing the parameter weights of a network is synonymous to learning, tuning and training.
Performance, loss and ability to generalize is types of knowledge of the task/data. There are two
approaches one could take for analyzing tuning and they are to define the amount of tuning through
relative change of the weights or as a measure of the change in the performance on some defined
metric. That is relative-weight-change or relative-knowledge-change analysis. The word fine implies
that the change is relatively small. For the relative-weight-change analysis a norm metric may be
used like the L2-norm of the weight changes, for finetuning do not change the weights over some
certain ratio. For the knowledge change the word fine implies that we should not "forget" about our
previous knowledge, our performance on the original task, but still change such that we also do well
on another task. This implies that the knowledge-change is dependent on both the performance on the
old metric and the new.

If one is only interested on the later dataset/task there is no need to keep track of how much the
model "forgets". But this seems inconsistent with referring to the learning as finetuning. We do not
really care to change the model "finely" but we could change the entire model for all we care. The
name for this concept then, instead of finetuning, becomes transfer learning. In other words for
transfer learning we do not care about our old performance as long as we perform as well as we can
on our current task.

Let $$ P_A, P_B $$ define two datasets A and B. Let the Model $$M_{\theta_0}(P_A)$$ be a model
trained on dataset A, with a total parameter volume $$V_{\theta_0}$$ and a knowledge score $$
S_{\theta_0}$$ for any set of datasets containing A, $$P_n = \{P_A, ... \}$$.  $$ S_{\theta_0}(P_n)
$$ could be the loss or accuracy, on some metric, defining the knowledge of the model.  Let $$
D_V(\theta_i, \theta_j), D_K(S_A, A_x)$$ be the distance metric functions that calculates $$
\frac{\partial}{\partial \theta}, \frac{\partial}{\partial K} \in R $$, the relative-weight-change
and relative-knowledge-change respectively. A definition of finetuning is to learn the dataset/task
B such that the total change $$TC = | \frac{\partial}{\partial \theta} | + | \frac{\partial}{\partial K}
| \leq C_{finetune}$$, for some scalar $$C_{finetune} \in R > 0$$. If $$ TC = 0 $$ no tuning has been done.

> Sidenote association to Schmidhuber "ultimate" learner:
>
> Only improve when the knowledge score
> $$S_K > 0 $$ for all tasks. This definition, correctly, does not directly seem to
> incorporate any constraint on the relative change of parameters

Given $$TC > C_{finetune}$$ we may say that the learning was transfer learning.



> ### Close but no Cigar -> [Transfer Learning, wiki](https://en.wikipedia.org/wiki/Transfer_learning)



------------------------------

### TODO
1. Evaluation method for transfer learning.
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


#### Small update

Finetune a model by training on 1 dialog, validate on another and test on a third. Below are values
for the 10 speakers in the dataset and the finetuning (alpha) results for 20 epochs.

In the top graph the training loss and the validation loss are shown and in the bottom is the
average accuracy on specific labeled moments in the conversation.

<img src="/images/turntaking/tt_finetune/finetune_10speakers.png" alt=""/>


| Speaker | TestScore | ValScore | Epoch |
| :------ | --------- | -------- | ----- |
| 0 | 0.7586206999318353 | 0.7500000428408384 | 16 |
| 1 | 0.541666696468989 | 0.8148148192299737 | 18 |
| 2 | 0.5945945945945946 | 0.7222222089767456 | 17 |
| 3 | 0.5319149024943088 | 0.7384615604694073 | 13 |
| 4 | 0.8181818398562345 | 0.9 | 6 |
| 5 | 0.6428571513720921 | 0.7777778219293665 | 4 |
| 6 | 0.6538461492611811 | 0.800000011920929 | 0 |
| 7 | 0.4545454536987977 | 0.5151515169577165 | 2 |
| 8 | 0.690476173446292 | 0.5789473801851273 | 10 |
| 9 | 0.6333333333333333 | 0.6545454491268504 | 15 |
| **Avg** | **0.63** | **0.725** | -  |







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



