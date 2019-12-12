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

Lets see if we may divide the binary state space of each channel (four states)
into states with information more absolute. The first distinction we can make is
to determine, for each channel, if it is currently speaking or not. This yields
$$  n_{channels}^2 $$  possible states. For our purposes we have 2 channels
resulting in 4 states $$\{(0,0),(0,1),(1,0),(1,1)\} $$ . Now we are interested
in information regarding the future of the conversation. Lets discretize the
future in to specific states. Lets pick 0.5 seconds as our reference time and
say that we are interested in the next 2 seconds, that is 4 discrete times (0,
.5], (.5 1], (1 1.5], (1.5 2]. Lets also add a bin for "later" ($$ \to \infty $$). This
gives us 5 possible time states. So far so good.

Given the first 4 states, speaking/no-speaking, we want to get the correct time
bin for when the speaker stops speaking and when the listener starts speaking.
Lets say that channel 0 is currently speaking and channel 1 is currently
listening. Now, the future state we construct should tell us when speaker 0 stops
speaking and speaker 1 stops being quiet. For our purposes lets say that speaker
0 stops talking 1.3 seconds and speaker 1 starts talking 1.7 seconds from the
current frame respectively. We have four rows (combination of speakers
speaking/no-speaking), five columns (the time slots) and the resulting state
would then be:

![State of conversation](../../images/tt_loss/state.png)

Where the rows correspond to speaker0 speaking, speaker1 speaking, speaker0
listens, speaker1 listens. The columns correspond to the time slots and the 1
values indicate where the feature of the current line stops. Speaker 0 stops
speaking 1.3 seconds from now (row 1, column 3) and speaker 1 stops being quiet 1.7
seconds from now (row 4, column 4). The last column means that the feature row
does not stop in "any forseeable future".

Now, how many combination of states do we get? We have 4 possible starting
states, $$n_{channels}^2$$, where each of these contain exactly 2 rows with
one value of one each. Each row has 5 columns$$n_{time}$$  and therefore we
can have $$n_{time} * n_{time}$$ or $$n_{time}^2$$  possible
permutations, and for this example this becomes $$ 25 (5^2) $$. The total states is
therefore 4*25=100. Or in more general terms $$N_{states} = n_{channels}^2 * n_{time}^2$$ 
possible future states.

Now we do not need to care about this specific states as rows and columns
containing ones and zeros but just the fact that we have $$N_{states})$$ 
possible output classes. In a way what we have constructed is a discretized
version of the conversation, with regard to turn-taking, who and when someone
speaks and listens. We could view these independent states as the next "words"
or "labels" in our turn-taking text and may train a network to output the
probabilities associated with either of these future "words", a language model
for turn-taking!

Now we have a 100 output classes and can optimize the model the maximize the
log-likelihood asssociated with each label/word/state.

















## Distill voice activation to context classes
* Multiclass classification
* Instead of having the voice activity as is we transform it, based on context, to classes
  - Beginning of utterance  (Onset - after gap)
  - Speaking                (Continuation - Speech and holds)
  - End of utterance        (Closure - before gap)
  - Listening               (Other)
  - Acknowledgment          (Overlap within)


