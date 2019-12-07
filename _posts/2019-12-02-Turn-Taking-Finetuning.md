---
layout: post
author: Erik
excerpt_separator: <!--more-->
---


Given that we have a sufficiently capable model for turn-taking in open ended, casual
conversation, can we train that model on particaular setups to improve the specific
capability on that task?

<!--more-->

* Use sample dialogs to train on and just finetune on this new dataset
  - How much is forgotten? Loss on original training data?
  - Freeze the weights of the network and retrain the output layer
* Train specifically to easily learn new tasks
  - MAML
