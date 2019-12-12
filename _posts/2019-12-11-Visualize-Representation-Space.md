---
layout: post
author: Erik
---



Sources:
* [Memorization in RNNs](https://distill.pub/2019/memorization-in-rnns/)
* [Visualizing and Understanding RNNs, (Karpathy, Johnson, 2016)](https://arxiv.org/pdf/1506.02078.pdf)



<!--more-->

TODO:
* [ ] TSNE projection of hidden space
    - [x] sample video
    - [ ] Gaps vs Pauses
    - [ ] speaker vs listeners
* [ ] Plain activation
* [ ] Saturation areas, [(Karpathy, Johnson, 2016)](https://arxiv.org/pdf/1506.02078.pdf)


## TSNE Projection of the latent space


Lets investigate our model by looking at the final RNN representation before the last Conv1D/Linear
layer. This state arguably should include information about the current and past state of the
conversation as well as some information about the future (these representations are used to
construct the prediction output). What ways are there to do this?

One way is to simply visualize the activation over a conversation and manually look for values we
think are interesting. However, this is not a very robust method and it could be very difficult to
actually see any patterns here (especially if the latent space is big). Another way is to project
this space on to a 2D plane using algorithms which try to cluster the points in some way. A useful
algorithm for this end is the [TSNE, van der Maaten, 2008](http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) algorithm.

In the following video snippet 2 TSNE embedded representations are shown. The left is for the first rnn
layer closest to the melspectrogram and the right one is the final representation before the ouput
layer.


<div class='centerImg'>
<video width="90%" controls>
  <source src="/images/turntaking/tt_representation/video_z_states2.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>
</div>

## Getting More Specific

The TSNE algorithm requires all the data for the clustering and for this specific sample only the
current dialog was used. This means that the representations, from both speakers, over all frames
were used as input to TSNE. In other words the dimensions of the 2D plane in the visualization does
not hold any specific meaning. Being in the lower left has no meaning other than that points in that
region are more similar, according to the TSNE training, than compared to other areas.

To get a better understanding we should do the TSNE calculations over as many datapoints as we have.
We could do it on all our data (training, validation and test set) or perhaps on the validation/test
set.

Posit that we do this, then we have the capability to compare multiple videos with the knowledge
that the projected Z space is the same. A better approach could be to ask certain questions
regarding this space. We could, for example, visualize the entire dataset, focus on some cluster (if
present), then look at what moments in the conversations they correspond to. Do we se cluster for
silence/speaking? Questions? hesitation markers? backchannels?

Or we could do it the other way around. From our training setup we care more about certain parts of
the conversation than others, explicitly HOLDS and SHIFTS, do we see any structure if we compare
these? Does the representations for the person speaking before the mutual silence differ from the
listener? How does correctly classified shifts/holds compare to the incorrectly classified? And so
on.

Instead of looking at a large cloud of tsne-points corresponding to various parts of the
conversations we may simplify it and look at the average encoding over these moments. Using the
average points we can show it as images and can abstract away from the continuous dialogs.


### Shifts vs Holds


### Speaker vs listener






