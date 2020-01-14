---
layout: post
---


Given that we have a "sufficiently" capable model for turn-taking in open ended, casual
phone conversations (Switchboard), can we train that model on particular setups to improve the specific
capability on that task?


<!--more-->

* class prediction model
    - overfit on swb_root takes 2 layers 80 hidden on supermodel with all features

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

|                    Model Type | hidden/layer | Dataset (Score) | Eval Dataset (Score) | Finetune Speaker Score Avg | Finetune Speaker Score Std |
|                    :--------- |:-- | :-------         | :------- | ------ | :------                    |
|                       MelSpec |7/3 | SWB (--)              | MapTask (--) |     -- | --                         |
|             Melspec + Prosody |7/3 | SWB (--)              | MapTask (--) |     -- | --                         |
|             Melspec + Prosody |10/3 | SWB (--)              | MapTask (--) |     -- | --                         |
|             Melspec + Prosody |16/3 | SWB (--)              | MapTask (--) |     -- | --                         |
|             Melspec + Prosody |10/3 | MT (--)              | swb_test_100 (--) |     -- | --                         |
|     Melspec + Prosody + Words |5/3 | SWB (--)              | MapTask (--) |     -- | --                         |

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



