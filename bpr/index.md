---
layout: layout
title:  Bayesian Personalized Ranking from Implicit Feedback
---

# Introduction

We will show how state-of-the-art models like MF or adaptive kNN can be optimized with respect to this criterion to provide better ranking quality than with usual learning methods. 

Let $U$ be the set of all users and $I$ the set of all items. In our scenario implicit feedback $S \subseteq U \times I$ is available. For convenience we also define:

$$
\begin{align*}
    I^+_u &:= \{i \in I : (u, i) \in S\} \\
    U^+_i &:= \{u \in U : (u, i) \in S\}
\end{align*}
$$

The usual approach for item recommenders is to predict a personalized score $$\hat{x}_{ui}$$ for an item that reflects the preference of the user for the item. Then the items are ranked by sorting them according to that score. Machine learning approaches for item recommenders typically create the training data from $S$ by giving pairs $(u, i) \in S$ a positive class label and all other combinations in $(U \times I) \backslash S$ a negative one. Then a model is fitted to this data. That means the model is optimized to predict the value 1 for elements in $S$ and 0 for the rest. The problem with this approach is that all elements the model should rank in the future ($(U \times I) \backslash S$) are presented to the learning algorithm as negative feedback during training. That means a model with enough expressiveness (that can fit the training data exactly) cannot rank at all as it predicts only 0s. The only reason why such machine learning methods can predict rankings are strategies to prevent overfitting, like regularization.
We use a different approach by using item pairs as training data and optimize for correctly ranking item pairs instead of scoring single items as this better represents the problem than just replacing missing values with negative ones. From $S$ we try to reconstruct for each user parts of $\succ_u$. If an item $i$ has been viewed by user $u$ – i.e. $(u, i) \in S$ – then we assume that the user prefers this item over all other non-observed items. For items that have both been seen by a user, we cannot infer any preference. The same is true for two items that a user has not seen yet. To formalize this we create training data $D_S : U \times I \times I$ by:

$$
D_S := \{(u, i, j) \mid i \in I^+_u \land j \in I \backslash I^+_u \}
$$

The semantics of $(u, i, j) \in D_S$ is that user $u$ is assumed to prefer $i$ over $j$. As $\succ_u$ is antisymmetric, the negative cases are regarded implicitly.


The Bayesian formulation of finding the correct personalized ranking for all items $i \in I$ is to maximize the following posterior probability where $\Theta$ represents the parameter vector of an arbitrary model class (e.g. matrix factorization):

$$ p(\Theta| \succ_u) \propto p(\succ_u | \Theta) p(\Theta) $$

Here, $\succ_u$ is the desired but latent preference structure for user $u$. 

# Assumptions

1. All users act independently of each other.
2. The ordering of each pair of items $\(i, j\)$ for a specific user is independent of the ordering of every other pair.
3. The total ranking $\succ_u$ follows the properties of a total order (totality, antisymmetry and transitivity).
4. The prior density $p(\Theta)$ is a normal distribution with zero mean and variance-covariance matrix $\Sigma_{\Theta}$ and to reduce the number of unknown hyperparameters, we set $\Sigma_{\Theta} = \lambda_{\Theta}I$.

Under these assumptions, we can formulate the maximum posterior estimator to derive our generic optimization criterion for personalized ranking BPR-Opt:

$$
\begin{eqnarray}
\text{BPR-Opt} : &=& \ln p(\Theta| \succ_u) \\
&=& \ln p(\succ_u \vert \Theta) + \ln p(\Theta) \\
&=& \ln \prod_{u \in U} p(\succ_u | \Theta) + \ln p(\Theta) \tag{Due to assumption 1.} \\
&=& \ln \prod_{(u,i,j) \in D_S} p(i \succ_u j| \Theta) + \ln p(\Theta) \tag{Due to assumptions 2 and 3.}\\
&=& \ln \prod_{(u,i,j) \in D_S} \sigma(\hat{x}_{uij}) + \ln p(\Theta) \\
&=& \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{uij}) + \ln p(\Theta) \\
&=& \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{uij}) - \lambda_{\Theta} ||\Theta||^2
\end{eqnarray}
$$

where $\lambda_{\Theta}$ are model-specific regularization parameters. We define the individual probability that a user really prefers item $i$ to item $j$ as:

$$
p(i \succ_u j| \Theta) := \sigma(\hat{x}_{uij})
$$

where $\sigma$ is the logistic sigmoid function:
$$ \sigma(x) := \frac{1}{1 + e^{-x}} $$
Here $$\hat{x}_{uij}$$ is an arbitrary real-valued function of the model parameter vector $\Theta$, which captures the special relationship between user $u$, item $i$, and item $j$. In other words, our generic framework delegates the task of modeling the relationship between $u$, $i$, and $j$ to an underlying model class like matrix factorization or adaptive kNN, which are in charge of estimating $$\hat{x}_{uij}$$. 

# BPR Learning Algorithm

As the BPR criterion is differentiable, gradient descent-based algorithms are an obvious choice for maximization. But as we will see, standard gradient descent is not the right choice for our problem. To solve this issue, we propose LearnBPR, a stochastic gradient-descent algorithm based on bootstrap sampling of training triples. 

The two most common algorithms for gradient descent are either full or stochastic gradient descent. In the first case, in each step, the full gradient over all training data is computed, and then the model parameters are updated with the learning rate $\alpha$.


In general, this approach leads to a descent in the 'correct' direction, but convergence is slow. As we have $O(\vert S \vert \vert I \vert)$ training triples in $D_S$, computing the full gradient in each update step is not feasible. Furthermore, for optimizing BPR-Opt with full gradient descent, the skewness in the training pairs leads to poor convergence. Imagine an item $i$ that is often positive. Then we have many terms of the form $\hat{x}_{uij}$ in the loss because for many users $u$, the item $i$ is compared against all negative items $j$ (the dominating class). Thus the gradient for model parameters depending on $i$ would dominate largely the gradient. That means very small learning rates would have to be chosen. Secondly, regularization is difficult as the gradients differ much.

The other popular approach is stochastic gradient descent. In this case, for each triple $(u, i, j) \in D_S$, an update is performed.

In general, this is a good approach for our skew problem but the order in which the training pairs are traversed is crucial. A typical approach that traverses the data item-wise or user-wise will lead to poor convergence as there are so many consecutive updates on the same user-item pair – i.e. for one user-item pair $(u, i)$ there are many $j$ with $(u, i, j) \in D_S$.

To solve this issue, we suggest using a stochastic gradient descent algorithm that chooses the triples randomly (uniformly distributed). With this approach, the chances to pick the same user-item combination in consecutive update steps are small. We suggest using a bootstrap sampling approach with replacement because stopping can be performed at any step. Abandoning the idea of full cycles through the data is especially useful in our case as the number of examples is very large and for convergence often a fraction of a full cycle is sufficient. We choose the number of single steps in our evaluation linearly depending on the number of observed positive feedback $S$.

#  Evaluation Methodology

We use the leave-one-out evaluation scheme, where we remove for each user randomly one action (one user-item pair) from his history, i.e., we remove one entry from $$I^+_u$$ per user $u$. This results in a disjoint train set $S_{\text{train}}$ and test set $S_{\text{test}}$. The models are then learned on $S_{\text{train}}$, and their predicted personalized ranking is evaluated on the test set $S_{\text{test}}$ by the average AUC statistic:

$$ 
\text{AUC} = \frac{1}{|U|} \sum_u \frac{1}{|E(u)|} \sum_{(i,j) \in E(u)} \delta(\hat{x}_{ui} > \hat{x}_{uj})
$$

where the evaluation pairs per user $u$ are:

$$ 
E(u) := \{(i, j) \mid (u, i) \in S_{\text{test}} \land (u, j) \notin (S_{\text{test}} \cup S_{\text{train}})\}
$$

A higher value of the AUC indicates better quality. The trivial AUC of a random guess method is 0.5, and the best achievable quality is 1.

