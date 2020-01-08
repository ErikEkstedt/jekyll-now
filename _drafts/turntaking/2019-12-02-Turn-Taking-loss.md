---
layout: post
author: Erik
excerpt_separator: <!--more-->
---

A semi-supervised approach to learn conversational organization, or how to manage
turn-taking, when to speak and when to listen, is to learn to predict the
future voice activity for each speaker during a conversation.

A perfect model, which during any time in a conversation, could predict the
future voice activity for both speaker would be able to have a fluent
conversation (given that it is also capable of knowing what to say). It would
know when a person is speaking and the likelihood of when that utterance is
done.

<!--more-->

## Learn future voice activation

* Choose prediction window of interest
* Guess whether a frame in the prediction window contains speech or not
* Lets pick the next 3 second window and guess the probability that each of
  those frames are speech/no-speech
* Optimize the prediction


$$ \frac{1}{100} = 4 $$

Regression
* We have the ground truth of speech in frames
* We optimize the MSELoss, $$ \frac{1}{N} \sum_{N} (y_{prediction} - y)^2 $$
Classification
- Two classes speech/no-speech


**Problems:**
- The prediction frames are not independent
  - Not between frames for a single speaker
  - Not between the predictions for each channel
- The frames closer to the current frame will have less variance than the later ones
  - It is easier to guess the next frame than 50 frames into the future


**Not Problems but...**
- The resulting prediction curve may be difficult to actually use
  - How are the prediction curves translated into actual turntaking?
  - Choose a interval that you care about, say the next second, compare the
    output probabilities between channels, then pick the one with the largest
    probability mass/volume.
- Where are other models (previous) work that focus on output such like this?
  - What loss-function and models have been tried? any?


## Learn future Turn-Taking states

Discrete states of a dialog. Edlund paper [Pauses, gaps and overlaps in Conversations, Heldner, Edlund 2010](http://www.speech.kth.se/prod/publications/files/3418.pdf)

<img class='centerImg' src="/images/turntaking/tt_discrete/edlund_states.png" alt="ALL" style='width: 90%'>


This yields 6 general classes for defining a conversation with regards to turn taking.
1. Only the first speaker is active
2. Only the second speaker is active
3. Silence in between utterance of the same speaker "Pauses"
4. Silence in between utterance of different speakers "Gaps"
5. Overlap between
6. Overlap within

Below we see the distribution from switchboard and maptask.

<div class='row'>
  <div class='column'>
    <img class='centerImg' src="/images/turntaking/tt_discrete/swb_hist_classes.png" alt="ALL" style='width: 90%'>
  </div>
  <div class='column'>
    <img class='centerImg' src="/images/turntaking/tt_discrete/mt_hist_classes.png" alt="ALL" style='width: 90%'>
  </div>
</div>

Given a prediction window (e.g 18 50ms frames) we can define a joint distribution combining
$$n_{bins}$$ bins. Given that we have 6 state for each frame this yields a distribution of $$
n_{frames}^{n_{bins}}$$ classes. If $$n_{bins} == n_{frames}$$ then we do not downsample the states
but for $$n_{bins} \leq n_{frames}$$ we loose some information but the number of total states
drastically change. As a starting point we model the 18 prediction frames as 3 bins which yields
$$r^6=216$$ classes.

<div class='row'>
  <div class='column'>
    <img class='centerImg' src="/images/turntaking/tt_discrete/swb_hist_labels_f18_b3.png" alt="ALL" style='width: 90%'>
  </div>
  <div class='column'>
    <img class='centerImg' src="/images/turntaking/tt_discrete/mt_hist_labels_f18_b3.png" alt="ALL" style='width: 90%'>
  </div>
</div>

As we see in the images above the total space of classes is sparse. This makes sense because we may
imagine some combination of states that will not be ordered consecutively e.g 'only_speaker0, pause,
only_speaker1' because that combine state is impossible given our definitions.

The most commom distributions are


Switchboard labels

1. class: 0 (25.4%): ['only_speaker_0', 'only_speaker_0', 'only_speaker_0']
2. class: 43 (24.6%): ['only_speaker_1', 'only_speaker_1', 'only_speaker_1']
3. class: 72 (3.16%): ['pause', 'only_speaker_0', 'only_speaker_0']
4. class: 79 (3.13%): ['pause', 'only_speaker_1', 'only_speaker_1']
5. class: 86 (3.07%): ['pause', 'pause', 'pause']
6. class: 44 (3.02%): ['only_speaker_1', 'only_speaker_1', 'pause']
7. class: 2 (2.97%): ['only_speaker_0', 'only_speaker_0', 'pause']
8. class: 50 (2.0%): ['only_speaker_1', 'pause', 'pause']
9. class: 85 (1.97%): ['pause', 'pause', 'only_speaker_1']
10. class: 84 (1.97%): ['pause', 'pause', 'only_speaker_0']
11. class: 14 (1.95%): ['only_speaker_0', 'pause', 'pause']
12. class: 12 (1.51%): ['only_speaker_0', 'pause', 'only_speaker_0']
13. class: 49 (1.47%): ['only_speaker_1', 'pause', 'only_speaker_1']
14. class: 3 (1.07%): ['only_speaker_0', 'only_speaker_0', 'gap']
15. class: 129 (0.991%): ['gap', 'gap', 'gap']
16. class: 45 (0.941%): ['only_speaker_1', 'only_speaker_1', 'gap']
17. class: 115 (0.861%): ['gap', 'only_speaker_1', 'only_speaker_1']
18. class: 108 (0.859%): ['gap', 'only_speaker_0', 'only_speaker_0']
19. class: 42 (0.69%): ['only_speaker_1', 'only_speaker_1', 'only_speaker_0']
20. class: 1 (0.669%): ['only_speaker_0', 'only_speaker_0', 'only_speaker_1']

Maptask labels

1. class: 0 (22.1%): ['only_speaker_0', 'only_speaker_0', 'only_speaker_0']
2. class: 129 (7.81%): ['gap', 'gap', 'gap']
3. class: 86 (7.42%): ['pause', 'pause', 'pause']
4. class: 43 (6.73%): ['only_speaker_1', 'only_speaker_1', 'only_speaker_1']
5. class: 72 (4.94%): ['pause', 'only_speaker_0', 'only_speaker_0']
6. class: 2 (4.19%): ['only_speaker_0', 'only_speaker_0', 'pause']
7. class: 84 (3.72%): ['pause', 'pause', 'only_speaker_0']
8. class: 14 (3.6%): ['only_speaker_0', 'pause', 'pause']
9. class: 3 (2.9%): ['only_speaker_0', 'only_speaker_0', 'gap']
10. class: 21 (2.28%): ['only_speaker_0', 'gap', 'gap']
11. class: 12 (2.18%): ['only_speaker_0', 'pause', 'only_speaker_0']
12. class: 108 (2.05%): ['gap', 'only_speaker_0', 'only_speaker_0']
13. class: 126 (1.9%): ['gap', 'gap', 'only_speaker_0']
14. class: 127 (1.86%): ['gap', 'gap', 'only_speaker_1']
15. class: 57 (1.72%): ['only_speaker_1', 'gap', 'gap']
16. class: 115 (1.57%): ['gap', 'only_speaker_1', 'only_speaker_1']
17. class: 45 (1.55%): ['only_speaker_1', 'only_speaker_1', 'gap']
18. class: 79 (1.0%): ['pause', 'only_speaker_1', 'only_speaker_1']
19. class: 19 (0.993%): ['only_speaker_0', 'gap', 'only_speaker_1']
20. class: 54 (0.97%): ['only_speaker_1', 'gap', 'only_speaker_0']


## Distill voice activation to context classes
* Multiclass classification
* Instead of having the voice activity as is we transform it, based on context, to classes
  - Beginning of utterance  (Onset - after gap)
  - Speaking                (Continuation - Speech and holds)
  - End of utterance        (Closure - before gap)
  - Listening               (Other)
  - Acknowledgment          (Overlap within)


