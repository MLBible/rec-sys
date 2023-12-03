---
layout: layout
title: Convolutional Sequence Embedding Recommendation - CASER
---

# Introduction

Let $$U = \{u_1, u_2, \dots, u_{\vert U \vert}\}$$ be the set of users and $$I = \{i_1, i_2, \dots, i_{\vert I \vert}\}$$ be the set of items. Each user $u$ is associated with a sequence of some items from $I$, 

$$
S_u = \{S_{1}^u, S_{2}^u, \dots, S_{|S_u|}^u\} \tag{1}
$$ 

where $$S_{t}^u \in I$$. The index $t$ for $$S_{t}^u$$ denotes the order in which an action occurs in the sequence $S^u$, not the absolute timestamp as in temporal recommendation models. Given all users' sequences $S_u$, the goal is to recommend each user a list of items that maximize their future needs, by considering both general preferences and sequential patterns. Unlike conventional top-N recommendation, top-N sequential recommendation models user behavior as a sequence of items, instead of a set of items.

The proposed model, Convolutional Sequence Embedding Recommendation (Caser), incorporates the Convolutional Neural Network (CNN) to learn sequential features and the Latent Factor Model (LFM) to learn user-specific features. The goal of Caser’s network design is multi-fold: capture both users' general preferences and sequential patterns at both union-level and point-level, and capture skip behaviors, all in unobserved spaces. Caser consists of three components: Embedding Look-up, Convolutional Layers, and Fully-connected Layers. To train the CNN, for each user $u$, we extract every $L$ successive items as input and their next $T$ items as the targets from the user’s sequence $S^u$. This is done by sliding a window of size $L + T$ over the user’s sequence, and each window generates a training instance for $u$, denoted by a triplet $(u, \text{previous } L \text{ items}, \text{next } T \text{ items})$.

# Embedding Look-up

Caser captures sequence features in the latent space by feeding the embeddings of the previous $L$ items into the neural network. The embedding $Q_i \in \mathbb{R}^d$ for item $i$ is a similar concept to its latent factors. Here, $d$ is the number of latent dimensions. The embedding look-up operation retrieves the embeddings of the previous $L$ items and stacks them together, resulting in a matrix $E^{(u, t)} \in \mathbb{R}^{L \times d}$ for the user $u$ at time step $t$, we have: 

$$
E^{(u, t)} = \begin{bmatrix}
Q_{S_{t-L}^u} \\
\vdots \\
Q_{S_{t-2}^u} \\
Q_{S_{t-1}^u}
\end{bmatrix} \tag{2}
$$

Along with the item embeddings, we also have an embedding $P_u \in \mathbb{R}^d$ for a user $u$, representing user features in latent space. 

# Convolutional Layers

Borrowing the idea of using CNNs in text classification, our approach regards the $L \times d$ matrix $E$ as the "image" of the previous $L$ items in the latent space and regards sequential patterns as local features of this "image". This approach enables the use of convolution filters to search for sequential patterns. Unlike image recognition, the "image" $E$ is not given because the embedding $Q_i$ for all items must be learned simultaneously with all filters.

## Horizontal Convolutional Layer

This layer has $n$ horizontal filters $F_k \in \mathbb{R}^{h \times d}$, $1 \leq k \leq n$. Here, $$h \in \{1, \dots, L\}$$ is the height of a filter. For example, if $L=4$, one may choose to have $n=8$ filters, two for each $h$ in $$\{1, 2, 3, 4\}$$. Each filter $F_k$ will slide from top to bottom on $E$ and interact with all horizontal dimensions of $E$ of the items $i$, $1 \leq i \leq L-h+1$. The result of the interaction is the $i$-th convolution value given by:

$$
c_k^i = \phi_c \left(E_{i:i+h-1} \odot F_k\right) \tag{3}
$$

Here, the symbol $\odot$ denotes the inner product operator and $\phi_c$ is the activation function for convolutional layers. This value is the inner product between $F_k$ and the sub-matrix formed by the row $i$ to row $i-h+1$ of $E$, denoted by $E_{i:i+h-1}$. The final convolution result of $F_k$ is the vector:

$$
c_k = \left(c_k^1, c_k^2, \dots, c_k^{L-h+1}\right) \tag{4}
$$

We then apply a max-pooling operation to $c_k$ to extract the maximum value from all values produced by this particular filter. The maximum value captures the most significant feature extracted by the filter. Therefore, for the $n$ filters in this layer, the output value $o \in \mathbb{R}^n$ is:

$$
o = \{\max(\mathbf{c}_1), \max(\mathbf{c}_2), \dots, \max(\mathbf{c}_n)\} \tag{5}
$$

Horizontal filters interact with every successive h items through their embeddings $E$. Both the embeddings and the filters are learned to minimize an objective function that encodes the prediction error of target items. By sliding filters of various heights, a significant signal will be picked up regardless of location. Therefore, horizontal filters can be trained to capture **union-level patterns** with multiple union sizes.

## Vertical Convolutional Layer

We use tilde ($\sim$) for the symbols of this layer. Suppose that there are $\tilde{n}$ vertical filters $\tilde{F}_k \in \mathbb{R}^{L \times 1}$, $1 \leq k \leq \tilde{n}$. Each filter $\tilde{F}_k$ interacts with the columns of $E$ by sliding $d$ times from left to right on $E$, yielding the vertical convolution result $\tilde{c}_k$:

$$
\tilde{c}_k = \left(\tilde{c}_k^1, \tilde{c}_k^2, \dots, \tilde{c}_k^d\right) \tag{6}
$$

For the inner product interaction, it is easy to verify that this result is equal to the weighted sum over the $L$ rows of $E$ with $\tilde{F}_k$ as the weights:

$$
\tilde{c}_k = \sum_{l=1}^{L} \tilde{F}_{k, l} E^{(l)}
$$

where $E^{(l)}$ is the $l$-th row of $E$. Therefore, with vertical filters, we can learn to aggregate the embeddings of the $L$ previous items, similar to Fossil’s weighted sum to aggregate the $L$ previous items’ latent representations. The difference is that each filter $\tilde{F}_k$ is acting like a different aggregator. Thus, similar to Fossil, these vertical filters are capturing point-level sequential patterns through weighted sums over previous items’ latent representations. While Fossil uses a single weighted sum for each user, we can use $\tilde{n}$ global vertical filters to produce $\tilde{n}$ weighted sums $\tilde{o} \in \mathbb{R}^{d \tilde{n}}$ for all users:

$$
\tilde{o} = \left[\tilde{c}_1, \tilde{c}_2, \dots, \tilde{c}_{\tilde{n}}\right]
$$

Since their usage is aggregation, vertical filters have some differences from horizontal ones: 

1. The size of each vertical filter is fixed to be $L \times 1$. This is because each column of $E$ is latent for us; it is meaningless to interact with multiple successive columns at one time.

2. There is no need to apply a max-pooling operation over the vertical convolution results, as we want to keep the aggregation for every latent dimension. Thus, the output of this layer is $\tilde{o}$.

# Fully-connected Layers

We concatenate the outputs of the two convolutional layers and feed them into a fully-connected neural network layer to get more high-level and abstract features:

$$
z = \phi_a \left( W \begin{bmatrix} \mathbf{o} \\ \tilde{\mathbf{o}} \end{bmatrix} + b \right)
$$

where $$W \in \mathbb{R}^{d \times (n + d\tilde{n})}$$ is the weight matrix that projects the concatenation layer to an $d$-dimensional hidden layer, $b \in \mathbb{R}^d$ is the corresponding bias term, and $\phi_a$ is the activation function for the fully-connected layer. $z \in \mathbb{R}^d$ is what we call convolutional sequence embedding, which encodes all kinds of sequential features of the $L$ previous items.

To capture the user’s general preferences, we also look up the user embedding $P_u$ and concatenate the two $d$-dimensional vectors, $z$ and $P_u$, together and project them to an output layer with $\vert I \vert$ nodes, written as:

$$
y^{(u, t)} = W' \begin{bmatrix} z \\ P_u \end{bmatrix} + b'
$$

where $$b' \in \mathbb{R}^{\vert I \vert}$$ and $$W' \in \mathbb{R}^{\vert I \vert \times 2d}$$ are the bias term and weight matrix for the output layer, respectively. The value $y^{(u, t)}_i$ in the output layer is associated with the probability of how likely user $u$ will interact with item $i$ at time step $t$. $z$ tends to capture short-term sequential patterns, whereas the user embedding $P_u$ captures the user’s long-term general preferences. Here, we put the user embedding $P_u$ in the last hidden layer for several reasons: 

1. It can have the ability to generalize other models. 

2. We can pre-train our model’s parameters with other generalized models’ parameters. Such pre-training is critical to model performance.

# Network Training

To train the network, we transform the values of the output layer, $y^{(u, t)}$, to probabilities by:

$$
p(S_{t}^u \mid S_{t-1}^u, S_{t-2}^u, \dots, S_{t-L}^u) = \sigma(y^{(u, t)}_{S_{t}^u})
$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function. Let $$C_u = \{L+1, L+2, \dots, \vert S^u \vert\}$$ be the collection of time steps for which we would like to make predictions for user $u$. The likelihood of all sequences in the dataset is:

$$
p(S \mid \Theta) = \prod_{u} \prod_{t \in C_u} \sigma(y^{(u, t)}_{S_{t}^u}) \cdot \prod_{j \neq S_{t}^u} (1 - \sigma(y_{j}^{(u, t)}))
$$

To further capture skip behaviors, we could consider the next $T$ target items, $$D_{t}^u = \{S_{t}^u S_{t+1}^u, \dots S_{t+T}^u\}$$, at once by replacing the immediate next item $S_{t}^u$ in the above equation with $D_{t}^u$. Taking the negative logarithm of likelihood, we get the objective function, also known as binary cross-entropy loss:

$$
\mathcal{L} = \sum_{u} \sum_{t \in C_u} \sum_{i \in D_{t}^u} - \log ( \sigma(y_{i}^{(u, t)}) ) + \sum_{j \neq i} - \log (1 - \sigma(y_{j}^{(u, t)})) \tag{7}
$$

Following previous works, for each target item $i$, we randomly sample several (3 in our experiments) negative instances $j$ in the second term.
The model parameters $$\Theta = \{P, Q, F, \tilde{F}, W, W', b, b' \}$$ are learned by minimizing the objective function in Eqn(7) on the training set, whereas the hyperparameters (e.g., $d, n, \tilde{n}, L, T$) are tuned on the validation set via grid search.

To control model complexity and avoid overfitting, we use two kinds of regularization methods: the L2 Norm is applied for all model parameters, and the Dropout technique with 50% drop ratio is used on fully-connected layers.

The complexity for making recommendations to all users is $O(\vert U \vert \vert I \vert d)$, where the complexity of convolution operations is ignored.

# Caser vs. MF
In reseach paper.

# Caser vs. FPMC
In reseach paper.

# Caser vs. Fossil
In reseach paper.

---

Figure 4 shows two "horizontal filters" that capture two union-level sequential patterns. These filters, represented as $h \times d$ matrices, have a height $h=2$ and a full width equal to $d$. They pick up signals for sequential patterns by sliding over the rows of $E$. For example, the first filter picks up the sequential pattern "(Airport, Hotel) GreatWall" by having larger values in the latent dimensions where Airport and Hotel have large values. Similarly, a "vertical filter" is a $L \times 1$ matrix and will slide over the columns of $E$. More details are explained below. 
