---
layout: layout
title: AutoRec
---

# Introduction to Autoencoders

Autoencoders were first introduced as a neural network that is trained to reconstruct its input. Their main purpose is learning in an unsupervised manner an “informative” representation of the data that can be used for various implications such as clustering. The problem is to learn the functions $A : \mathbb{R}^n \to \mathbb{R}^p$ (encoder) and $B : \mathbb{R}^p \to \mathbb{R}^n$ (decoder) that satisfy

$$
\arg\min_{A,B} \mathbb{E}[\Delta \left(\mathbf{x}, B \circ A(\mathbf{x}) \right)]^2 \tag{1}
$$

where $\mathbb{E}$ is the expectation over the distribution of $x$, and $\Delta$ is the reconstruction loss function, which measures the distance between the output of the decoder and the input. The latter is usually set to be the $l_2$-norm. In the most popular form of autoencoders, $A$ and $B$ are neural networks. In the special case that $A$ and $B$ are linear operations, we get a linear autoencoder. In the case of a linear autoencoder where we also drop the non-linear operations, the autoencoder would achieve the same latent representation as Principal Component Analysis (PCA). Therefore, an autoencoder is, in fact, a generalization of PCA, where instead of finding a low-dimensional hyperplane in which the data lies, it is able to learn a non-linear manifold. Autoencoders may be trained end-to-end or gradually layer by layer. In the latter case, they are "stacked" together, which leads to a deeper encoder.

# AutoRec: Autoencoders Meet Collaborative Filtering

Assume a dataset consisting of M users and N items. Let $r_m \in \mathbb{R}^N$ be a partially observed vector for the user $m$ consisting of its preference score to each of the $N$ items and let $r_n$ be item $n$'s partially observed vector for each user. The AutoRec model has two variants: user-based AutoRec (U-AutoRec) and item-based AutoRec (I-AutoRec). In U-AutoRec, the autoencoder takes as input each partially observed $r_m$, projects it into a low-dimensional latent (hidden) space, and then reconstruct $r_m$ in the output space to predict missing ratings for purposes of recommendation for specific users. Similarly, for I-AutoRec, the autoencoder learns a lower-dimensional representation of user preferences for specific items.

The U-AutoRec objective is defined as

$$
\min_{\theta}\sum _{m=1}^{M} \Vert r_m -h(r_m; \theta ) \Vert ^{2} _{O} + \frac{\lambda}{2}\left( \Vert W \Vert ^{2} _{F} + \Vert V \Vert ^{2} _{F}\right)
$$

where 

- $h(r_m; \theta)$ is the reconstruction of input $r_m$ defined as-,

$$
h(r_m; \theta) = f(W \cdot g(Vr_m + \mu) + b)
$$

- $$\Vert \cdot \Vert^2_{O}$$ means that the loss is defined only on the observed preferences of the user.
- $f(\cdot)$ and $g(\cdot)$ are activation functions.
- $$\theta = \{W, V, \mu, b\}$$ is the set of parameters for transformations $W \in \mathbb{R}^{N \times k}$, $V \in \mathbb{R}^{k \times N}$, and biases $\mu \in \mathbb{R}^{k}$, $b \in \mathbb{R}^{N}$
- $\lambda > 0$ is the regularisation strength. 

This objective corresponds to an auto-associative neural network with a single, $k$-dimensional hidden layer where $k \ll N$. The parameters are learned using backpropagation. In total, U-AutoRec requires the estimation of $2Nk\ +\ N\ +\ k$ parameters. At prediction time, we can investigate the reconstruction vector and find items that the user is likely to prefer. The I-AutoRec can be defined in a similar manner. 

The basic AutoRec model was extended by including denoising techniques and incorporating users' and items' side information such as user demographics or item description. The denoising serves as another type of regularization that prevents the auto-encoder from overfitting rare patterns that do not concur with general user preferences. The side information was shown to improve accuracy and speed up the training process.

Similar to the original AutoRec, two symmetrical models have been proposed, one that works with user preference vectors $r_m$ and the other with item preference vectors $r_n$. In the general case, these vectors may consist of explicit ratings. The Collaborative Denoising Auto-Encoder (CDAE) model essentially applies the same approach to vectors of implicit ratings rather than explicit ratings. Finally, a variational approach has been attempted by applying VAE in a similar fashion.

# Collaborative Filtering with Stacked Denoising AutoEncoders and Sparse Inputs

# Hybrid recommender system based on autoencoders

# Collaborative denoising auto-encoders for top-n recommender systems


