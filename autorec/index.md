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

Assume a dataset consisting of M users and N items. Let $r_m \in \mathbb{R}^N$ be a partially observed vector for the user $m$ consisting of its preference score to each of the $N$ items where unknown ratings are set to zeros by default. Similarly, let $r_n$ be item $n$'s partially observed vector for each user. The AutoRec model has two variants: user-based AutoRec (U-AutoRec) and item-based AutoRec (I-AutoRec). In U-AutoRec, the autoencoder takes as input each partially observed $r_m$, projects it into a low-dimensional latent (hidden) space, and then reconstruct $r_m$ in the output space to predict missing ratings for purposes of recommendation for specific users. Similarly, for I-AutoRec, the autoencoder learns a lower-dimensional representation of user preferences for specific items.

The U-AutoRec objective is defined as

$$
\min_{\theta}\sum _{m=1}^{M} \Vert r_m -h(r_m; \theta ) \Vert ^{2} _{O} + \frac{\lambda}{2}\left( \Vert W \Vert ^{2} _{F} + \Vert V \Vert ^{2} _{F}\right)
$$

where 

- $h(r_m; \theta)$ is the reconstruction of input $r_m$ defined as-,

$$
h(r_m; \theta) = f(W \cdot g(Vr_m + \mu) + b)
$$

- $$\Vert \cdot \Vert^2_{O}$$ means that the loss is defined only on the observed preferences of the user, that is, only weights that are associated with observed inputs are updated during back-propagation.
- $f(\cdot)$ and $g(\cdot)$ are activation functions.
- $$\theta = \{W, V, \mu, b\}$$ is the set of parameters for transformations $W \in \mathbb{R}^{N \times k}$, $V \in \mathbb{R}^{k \times N}$, and biases $\mu \in \mathbb{R}^{k}$, $b \in \mathbb{R}^{N}$. The weight matrix $W$ of the reverse mapping may optionally be constrained by $W = V^T$, in which case the autoencoder is said to have tied weights.
- $\lambda > 0$ is the regularisation strength. 

This objective corresponds to an auto-associative neural network with a single, $k$-dimensional hidden layer where $k \ll N$. The parameters are learned using backpropagation. In total, U-AutoRec requires the estimation of $2Nk\ +\ N\ +\ k$ parameters. At prediction time, we can investigate the reconstruction vector and find items that the user is likely to prefer. The I-AutoRec can be defined in a similar manner. 

The basic AutoRec model was extended by including denoising techniques and incorporating users' and items' side information such as user demographics or item description. The denoising serves as another type of regularization that prevents the auto-encoder from overfitting rare patterns that do not concur with general user preferences. The side information was shown to improve accuracy and speed up the training process.

Similar to the original AutoRec, two symmetrical models have been proposed, one that works with user preference vectors $r_m$ and the other with item preference vectors $r_n$. In the general case, these vectors may consist of explicit ratings. The Collaborative Denoising Auto-Encoder (CDAE) model essentially applies the same approach to vectors of implicit ratings rather than explicit ratings. Finally, a variational approach has been attempted by applying VAE in a similar fashion.

# Denoising Auto-encoder

The Denoising Auto-encoder (DAE) extends the classical auto-encoder by training to reconstruct each data point $\mathbf{x}$ from its (partially) corrupted version $\tilde{\mathbf{x}}$. The goal of DAE is to force the hidden layer to discover more robust features and to prevent it from simply learning the identity function. The corrupted input $$\tilde{\mathbf{x}}$$ is typically drawn from a conditional distribution $$p(\tilde{\mathbf{x}} \vert {\mathbf{x}})$$. Common corruption choices are the additive Gaussian noise and the multiplicative mask-out/drop-out noise. Under mask-out/drop-out corruption, one randomly overwrites each of the dimensions of $\mathbf{x}$ with $0$ with a probability of $q$:

$$
\begin{eqnarray}
P(\tilde{x_d} = \delta x_d) & = & 1 - q \\
P(\tilde{x_d} = 0) & = & q \tag{2}
\end{eqnarray}
$$

All information about the chosen components is thus removed from that particular input pattern, and the autoencoder will be trained to "fill-in" these artificially introduced "blanks." To make the corruption unbiased, one sets the uncorrupted values to $\delta = \frac{1}{1-q}$ times their original value. The  corruption + denoising procedure is applied not only on the input, but also recursively to intermediate representations.

The denoising autoencoder can thus be seen as a way to define and learn a manifold. The intermediate representation $Y = f(X)$ can be interpreted as a coordinate system for points on the manifold (this is most clear if we force the dimension of $Y$ to be smaller than the dimension of $X$). More generally, one can think of $Y = f(X)$ as a representation of $X$ which is well suited to capture the main variations in the data, i.e., on the manifold. When additional criteria (such as sparsity) are introduced in the learning model, one can no longer directly view $Y = f(X)$ as an explicit low-dimensional coordinate system for points on the manifold, but it retains the property of capturing the main factors of variation in the data.

# Collaborative Denoising Auto-Encoder

Similar to the standard Denoising Auto-Encoder, CDAE is also represented as a one-hidden-layer neural network. The key difference is that the input also encodes a latent vector for the user, which allows CDAE to be a much better recommender model.

In the input layer, there are in total $I+1$ nodes, where each of the first $I$ nodes corresponds to an item, and the last node is a user-specific node, which means the node and its associated weights are unique for each user $u \in U$ in the data. We refer to the first $I$ nodes as _item input nodes_, and the last node as the _user input node_. Given the historical feedback $O$ by users on the item set $I$, we can transform $O$ into the training set containing $U$ instances $$\{y_1, y_2, \ldots, y_U\}$$, where $$y_u = \{y_{u1}, y_{u2}, \ldots, y_{uI}\}$$ is the $I$-dimensional feedback vector of user $u$ on all the items in $I$. $y_u$ is a sparse binary vector that only has $\vert O_u \vert $ non-zero values: $y_{ui} = 1$ if $i$ is in the set $O_u$, otherwise $y_{ui} = 0$. There are $K$ nodes in the hidden layer, and these nodes are fully connected to the nodes of the input layer. Here, $K$ is a predefined constant, which is usually much smaller than the size of the input vectors. The hidden layer also has an additional node to model the bias effects. We use $W \in \mathbb{R}^{I \times K}$ to denote the weight matrix between the item input nodes and the nodes in the hidden layer, and $V_u \in \mathbb{R}^K$ to denote the weight vector for the user input node. Note that $V_u$ is a user-specific vector, i.e., for each of the users, we have one unique vector. From another point of view, $W_i$ and $V_u$ can be seen as the distributed representations of item $i$ and user $u$ respectively.

In the output layer, there are $I$ nodes representing reconstructions of the input vector $y_u$. The nodes in the output layer are fully connected with nodes in the hidden layer. The weight matrix is denoted by $$W^{'} \in \mathbb{R}^{I \times K}$$, and we denote the weight vector for the bias node in the hidden layer by $b \in \mathbb{R}^I$. Formally, the inputs of CDAE are the corrupted feedback vector $\tilde{y}_u$, which is generated from $p(\tilde{y}_u \vert y_u)$ as stated in Equation 2. Intuitively, the non-zero values in $y_u$ are randomly dropped out independently with probability $q$. The resulting vector $y_u$ is still a sparse vector, where the indexes of the non-zero values are a subset of those of the original vector.


CDAE first maps the input to a latent representation $z_u$, which is computed as follows:

$$
z_u = h(W^{T}\tilde{y}_u + V_u + b) \tag{3}
$$

where $h(\cdot)$ is an element-wise mapping function (e.g., identity function $h(x) = x$ or sigmoid function $h(x) = \sigma(x) = \frac{1}{1 + e^{-x}}$), and $b \in \mathbb{R}^K$ is the offset vector.

At the output layer, the latent representation is then mapped back to the original input space to reconstruct the input vector. The output value $\hat{y}_{ui}$ for node $i$ is computed as follows:

$$
\hat{y}_{ui} = f \left(W^{'}_{i}z_u + b^{'}_i \right) \tag{4}
$$

where $$W^{'} \in \mathbb{R}^{I \times K}$$ and $$b^{'}$$ are the weight matrix and the offset vector for the output layer, respectively, and $f(\cdot)$ is also a mapping function.

We learn the parameters of CDAE by minimizing the average reconstruction error:

$$
arg \ min \ \frac{1}{U} \sum_{u=1}^{U} \mathbb{E}_{p(\tilde{y}_u \vert y_u)} \left[ l(\tilde{y}_u, \hat{y}_u) \right] + R(W, W^{i}, V, b, b^{'}) \tag{5}
$$

where $R$ is the regularization term to control the model complexity.

$$
R(\cdot) = \frac{\lambda}{2} \left(\Vert W \Vert^2_2 + \Vert W^{'} \Vert ^2_2 + \Vert V \Vert^2_2 + \Vert b \Vert^2_2 + \Vert b^{'} \Vert^2_2 \right)
$$

We apply Stochastic Gradient Descent (SGD) to learn the parameters. Because the number of output nodes equals the number of items, the time complexity of one iteration over all users is $O(U \cdot I \cdot K)$, which is impractical when the number of users and the number of items are large. Instead of computing the gradients on all the outputs, we only sample a subset of the negative items $S_u$ from $\bar{O}_u$ and compute the gradients on the items in $O_u \cup S_u$. The size of $S_u$ is proportional to the size of $O_u$. So the overall complexity of the learning algorithm is linear in the size of $O$ and the number of latent dimensions $K$. An alternative solution is to build a Hierarchical Softmax tree on the output layer, but it requires the loss function on the output layer to be softmax loss.

## Learning Algorithm for CDAE

To be written from the research paper "Collaborative Denoising Auto-Encoders for Top-N Recommender Systems".

# Collaborative Filtering with Stacked Denoising AutoEncoders and Sparse Inputs

# Hybrid recommender system based on autoencoders

# Collaborative denoising auto-encoders for top-n recommender systems


