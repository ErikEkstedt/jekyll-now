---
layout: post
title: Generative Models
---



## What are Generative Models?


* What are Generative Models?
  - A probability distribution.
  - A generator
  - A way to sample data as similar to real data as possible
  - *p(x; theta)*

<!--more-->

* What should they know?
  - They approximate the probability space over all possible X.
  - A structure "separating" probable x's and UNprobable x's

* How do we train them to know this?
  - We want the model *p(x; theta)* to be as true as possible, i.e as close to *p\**(x)* as
    possible
  * A distance metric between distributions is the **KLD** function
  * KLD is non-symmetric >= 0
  * KLD becomes the difference between the negative average log-likelihood, 
    `p(X; theta)` and the Entropy of the unknown distribution, `H(p*(x))`

![Equation 1]( {{ site.baseurl }}/images/generative-models/gen_mod_know_ekv1.png)
* Change of Variable
  - Flow
![Equation 2]( {{ site.baseurl }}/images/generative-models/gen_mod_know_ekv2.png)
* Maximizing LogLikehood Estimation
  * We cant do anything about the unknown distribution so we are forced to maximize the LogLikelhood
![Equation 3]( {{ site.baseurl }}/images/generative-models/gen_mod_know_ekv3.png)

