---
layout: post
author: Erik
excerpt_separator: <!--more-->
---



How can we evaluate our performance on the given task? After we decided upon a specific loss
function our objective is to get that as low as possible. Thus we already now a metric to measure
our performance, however, this may not give enough insight. Were does the model predict accurately
and were does it suck? Does the model pick up on any particular patterns? why? Would this model be
valuable for enhancing turn organization for any system?


<!--more-->

There are many things that we would like to know about the model for this particular problem setup.
Turn organization concerns speech between two or more interlocutors and therefore many of the things
we wonder if the model picks up on are associated what we know about this field, our prior
knowledge.



Closely related to our output space (voice activity) we may infer particular events that we could
use to see whether the model matches our expectations. We may talk about [pauses, gaps and
overlaps](http://www.speech.kth.se/prod/publications/files/3418.pdf). A pause is defined as a moment
of mutual silence where the current speaker pauses and then continues speaking. The definition for a
gap is again located at a mutual silence but where the speaker following the silence is different
from the speaker before. Overlaps may be divided further into two categories, following the paper by
Heldner and Edlund, to overlaps between turns and within turns. An overlap within turns is when a
speaker is speaking, having the turn, and the other speaker utters an utterance that is finished
before the first speaker turn is up. Backchannels for example would fall into this category. The
overlaps between turns is when a speaker is holding the turn, the other speaker starts speaking
before the former is finished, and then the former speakers utterance is done and the second speaker
continues the conversation (holding the turn). From a turn-taking perspective the pauses would precede
Holds (holding the turn) and gaps would precede Shifts. Overlaps between would be associated with
Shifts while overlaps within is associated with Holds.


Given the voice activation from a conversation these states may be extracted and we may look at the
prediction during these events and see if the model fails/succeeds on any in particular.


--------------------

- Loss at Gaps?
- Loss at Pauses?
- Loss at overlaps?

--------------------


Linguists may define a conversation on the level of [dialog
acts](https://en.wikipedia.org/wiki/Dialog_act) which are acts like statements, questions and
requests. Dialog acts is a subset of [speech acts](https://en.wikipedia.org/wiki/Speech_act) which
includes acts such as apologizing, promising, ordering, answering etc. During a conversation between
two speakers we would like the model to output higher probability for the other speaker to speak
following a question of the former. How can we evaluate this?

--------------------

- Loss at Questions?
  - Loss at Yes/No-questions?
  - Loss at WH-questions?
- Loss at statements?
- Acknowledge-answers ("oh okay")?
- Backchannel-questions ('oh really')?

--------------------

On a "lower" level of inquiry we might ask if the prosody (intensity, pitch and duration)
of certain utterances or words follow our prior intuitions. In prosody we may talk about
[intonation](https://en.wikipedia.org/wiki/Intonation_(linguistics)) or the rising and
falling of pitch in an utterence (or in words).

- Normal conversations: middle, high pitch. falling pitch at the end of an utterance.
- Yes/No questions: rising pitch at the end. "do you think so?"
- Wh-question: 2-3-1 pattern -> start middle/high -> rise above -> fall down at end "Who(2) did(3) it(1)?"

--------------------

- Loss at end of utterance words with 
  - falling pitch?
  - rising pitch?

--------------------













## Additional Miscellaneous Scribblings
* Speech acts: 
  - **J. L. Austin** [How to do things with words]()
  - Two quotes about Austins work:
  "In linguistics and the philosophy of mind, a locutionary act is the performance of an utterance, and hence of a speech act. The term equally refers to the surface meaning of an utterance because, according to J. L. Austin's posthumous How To Do Things With Words, a speech act should be analysed as a locutionary act (i.e. the actual utterance and its ostensible meaning, comprising phonetic, phatic, and rhetic acts corresponding to the verbal, syntactic, and semantic aspects of any meaningful utterance), as well as an illocutionary act (the semantic 'illocutionary force' of the utterance, thus its real, intended meaning), and in certain cases a further perlocutionary act (i.e. its actual effect, whether intended or not)."
  "In Austin's framework, locution is what was said and meant, illocution is what was done, and perlocution is what happened as a result."
  - [How to do things with words: A summary (blog)](https://disabilitywrites.commons.gc.cuny.edu/2013/09/06/i-say-therefore-i-do-j-l-austins-how-to-do-things-with-words/)

