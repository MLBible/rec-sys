---
layout: layout
title: Matrix Factorisation
---

# Introduction

Matrix factorization is a class of collaborative filtering models. Specifically, the model factorizes the user-item interaction matrix (e.g., rating matrix) into the product of two lower-rank matrices, capturing the low-rank structure of the user-item interactions.

Let $$\mathbf{R} \in \mathbb{R}^{m \times n}$$ denote the interaction matrix with $$m$$ users and $$n$$ items, and the values of $$\mathbf{R}$$ represent explicit ratings. The user-item interaction will be factorized into a user latent matrix $$\mathbf{P} \in \mathbb{R}^{m \times k}$$ and an item latent matrix $$\mathbf{Q} \in \mathbb{R}^{n \times k}$$, where $$k \ll m,n$$, is the latent factor size. Let $$\mathbf{p}_{u}$$ denote the $$u^{\text{th}}$$ row of $$\mathbf{P}$$ and $$\mathbf{q}_{i}$$ denote the  $$i^{\text{th}}$$row of $$\mathbf{Q}$$. For a given item $i$, the elements of $\mathbf{q}_{i}$ measure the extent to which the item possesses those characteristics such as the genres and language of a movie. For a given user $u$, the elements of $$\mathbf{p}_{u}$$ measure the extent of interest the user has in items’ corresponding characteristics. These latent factors might measure obvious dimensions as mentioned in those examples or are completely uninterpretable. The predicted ratings can be estimated by

$$
\hat{\mathbf{R}} = \mathbf{P}\mathbf{Q}^T \tag{1}
$$

where $$\hat{\mathbf{R}} \in \mathbb{R}^{m \times n}$$ is the predicted rating matrix which has the same shape as $\mathbf{R}$. One major problem of this prediction rule is that users/items biases can not be modeled. For example, some users tend to give higher ratings or some items always get lower ratings due to poorer quality. To capture these biases, user specific and item specific bias terms are introduced. Specifically, the predicted rating user $u$ gives to item $i$ is calculated by

$$
\hat{\mathbf{R}}_{ui} = \mathbf{p}_u \mathbf{q}_i^T + b_u + b_i \tag{2}
$$

Then, we train the matrix factorization model by minimizing the mean squared error between predicted rating scores and real rating scores. The objective function is defined as follows:

$$
\underset{\mathbf{P}, \mathbf{Q}, b}{\mathrm{argmin}}
\sum_{(u,i) \in \kappa} \Vert \mathbf{R}_{ui} - \hat{\mathbf{R}}_{ui} \Vert^2 + 
\lambda(\Vert\mathbf{P}\Vert^2_F + \Vert\mathbf{Q}\Vert^2_F + b_u^2 + b_i^2) \tag{3}
$$
 
where $\lambda$ denotes the regularization rate. The regularizing term  is used to avoid over-fitting by penalizing the magnitude of the parameters. The $(u, i)$ pairs for which $\mathbf{R}_{ui}$ is known are stored in the set $\kappa$. The model parameters can be learned with an optimization algorithm, such as Stochastic Gradient Descent and Adam. We then implement the RMSE (root-mean-square error) measure, which is commonly used to measure the differences between rating scores predicted by the model and the actually observed ratings (ground truth).

# Limitations

Matrix factorization is essentially a linear model and as is not capable of capturing complex nonlinear and intricate relationships that may be predictive of users’ preferences.

