---
layout: post
title: Information
---




<!--more-->


* Claude Shannon
  - 1948 [A Mathematical Theory of Communication](http://www.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
  - Information source
  - Transmitter
  - Signal
  - Recieved signal
  - Reciever

<img class="centerImg" src="/assets/images/Information/Shannon_schematic.png " alt="Shannon schematic"/>

Disclaimer: Much of the text is paraphrased from other sources and is mostly a way for
myself to learn.


Information theory is based on probability theory and statistics and often involves of
information of distributions associated with random variables. The basic unit of
information is the **bit** which has a binary base but could be defined by other bases
where a common one is the **nat** where the base is **e**.

Useful equations

## [Entropy](http://www.scholarpedia.org/article/Mutual_information)

Entropy is the measure of information in a single random variable. Higher entropy -> more
more uncertain, less entropy -> more certain.

$$ H(X) = - \sum_x P_X(x) log( p_X(x) = - \mathbb{E}_{p_X} log(p_X) $$

> Although not particularly obvious from this equation, H(X) has a very concrete
> interpretation: Suppose x is chosen randomly from the distribution PX(x) , and someone who
> knows the distribution PX(x) is asked to guess which x was chosen by asking only yes/no
> questions. If the guesser uses the optimal question-asking strategy, which is to divide
> the probability in half on each guess by asking questions like "is x greater than x0 ?",
> then the average number of yes/no questions it takes to guess x lies between H(X) and
> H(X)+1 (Cover and Thomas, 1991). Scholarpedia


### Conditional Entropy

The conditional entropy is the average uncertainty about X after observing a second random variable Y , and is given by

$$ H( X | Y ) = \sum_y \left[ - P_{X | Y} (x | y) log \frac{P_{XY}(x,y)}{P_Y(y)}  \right] = - \mathbb{E}_{p_Y} \left[ log(p_{X | Y}) \right] $$ 

is the conditional probability of x given y. 


## Mutual Information

Mutual information is the measure of information in common between two random variables.
For two discrete variables X and Y whose joint probability distribution is PXY(x,y) , the
mutual information between them, denoted I(X;Y) , is given by (Shannon and Weaver, 1949;
Cover and Thomas, 1991)

$$ I(X;Y)= \sum_{x,y} P_{XY}(x,y) log \frac{P_{XY}(x,y)}{P_X(x) P_Y(y)} =  \mathbb{E}_{P_{XY}} log \frac{P_{XY}}{P_X P_Y} $$

With the definitions of H(X) and H(X|Y) this
becomes:

$$ I(X;Y)=H(X)−H(X|Y) $$

Mutual information is therefore the reduction in uncertainty about variable X , or the expected reduction in the number of yes/no questions needed to guess X after observing Y 

"How much uncertainty in one variable is reduced by knowing another variable" [The Information Bottleneck Theory, Naftali Tishby](https://www.youtube.com/watch?v=EQTtBRM0sIs)


## Kullback-Leibler divergence (information gain)

The Kullback-Leibler divergence is a "distance" metric (however it is not symmetric and
thus not a true distance) between two distributions.

$$ D_{KL}(p(X) || q(x)) = \sum_{x \in X} p(x) log \frac{p(x)}{q(x)} $$

The mutual information can then also be expresssed as

$$ I(X,Y) = D_{KL} \left( P_{XY}(X, Y) || P(X) P(Y) \right) $$


# Deep Learning and Information






## Informational Bottleneck

Let the input to a neural network be X, a layer Y and the output Z. Then the mutual
information between the input and the output will follow:
![data_processing_inequality_DPI_and_invariance.png](/assets/images/Information/data_processing_inequality_DPI_and_invariance.png)




## Sources

* [wikipedia](https://en.wikipedia.org/wiki/Information_theory)
* [Scholarpedia: Mutual Information](http://www.scholarpedia.org/article/Mutual_information)
* [Scholarpedia: Entropy](http://www.scholarpedia.org/article/Entropy)
