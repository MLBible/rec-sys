---
layout: layout
title: AppJoy - Personalized Mobile Application Discovery
---
# Introduction

AppJoy applies an **RFD (Recency, Frequency, and Duration) model**, which is a variation of the widely-used **RFM (Recency, Frequency, and Monetary)** model in marketing that measures a customers behavior and loyalty. Recency measures how recently a customer has purchased; Frequency measures how often the customer purchased in a given time period; and Monetary measures how much the customer spent in that period. In the context of using mobile applications, AppJoy focuses on the user interacting with the application through the applications user interface. Thus recency means how recently a user has interacted with the application. Frequency means how frequently the user interacted with the application in a given time period. Instead of using Monetary value, AppJoy uses Duration to measure how long the user actually interacted with the application. By combining these three values, RFD can provide a good estimate of how much a user likes to use an application.

 We define recency ($$v_R$$) as the time elapsed since the last use of the application $$p$$ by the user $$u$$, frequency ($$v_F$$) as the number of times $$u$$ interacted with the application within a certain period, and duration ($$v_D$$) as the total duration time that $$u$$ interacted with an application during that period. The usage score is thus represented as:

$$v_{up} = w_R v_R + w_F v_F + w_D v_D$$

where $$w_R$$, $$w_F$$, and $$w_D$$ are the weights based on their relative importance. The combination of these three measurements reflects the user's application taste. The applications that have been used more recently, more frequently, and for more time are likely to be favored more by the user.

# Slope One Prediction

Consider two users, $$A$$ and $$B$$, two items, $$I$$ and $$J$$, and the table below. 

|       |   Item I   |   Item J   |
| :---: | :---: | :---: |
|   User A   |  1   |  1.5   |
|   User B   |  2   |  ?   |

User $$A$$ gave item $$I$$ a rating of 1, whereas user $$B$$ gave it a rating of 2, while user $$A$$ gave item $$J$$ a rating of 15. We observe that item $$J$$ is rated more than item $$I$$ by $$1.5 - 1 = 0.5$$ points. Thus, we could predict that user $$B$$ will give item $$J$$ a rating of $$2 + 0.5 = 2.5$$. We call user $$B$$ the predictee user and item $$J$$ the predictee item. Many such differentials exist in a training set for each unknown rating, and we take an average of these differentials.

Given two evaluation arrays $$v_i$$ and $$w_i$$ with $$i = 1$$ to $$n$$, we search for the best predictor of the form $$f(x) = x + b$$ (and hence the name Slope One) to predict $$w$$ from $$v$$ by minimizing $$\sum_{i=1}^{n}(v_i + b - w_i)^2$$. 

Deriving with respect to $$b$$ and setting the derivative to zero, we get:



$$
b = \frac{\sum_{i=1}^{n}w_i - v_i}{n}
$$ 



In other words, the constant $$b$$ must be chosen to be the average difference between the two arrays. 

This result motivates the following scheme. Given a training set $$\chi$$, and any two items $$j$$ and $$i$$ with ratings $$u_j$$ and $$u_i$$ respectively in some user evaluation $$u$$, (annotated as $$u \in {S_{ji}}(\chi)$$), we consider the average deviation of item $$i$$ with respect to item $$j$$ as:

$$
\text{dev}_{ji} = \sum_{u \in S_{j,i}(\chi)} \frac{u_j - u_i}{\text{card}(S_{ji}(\chi))}
$$

Note that any user evaluation $$u$$ not containing both $$u_j$$ and $$u_i$$ is not included in the summation. The symmetric matrix defined by $$\text{dev}_{ji}$$ can be computed once and updated quickly when new data is entered.

Given that $$\text{dev}_{ji} + u_i$$ is a prediction for $$u_j$$ given $$u_i$$, a reasonable predictor might be the average of all such predictions:

$$
P(u)_j = \frac{1}{\text{card}(R_j)} \sum_{i \in R_j} \left(\text{dev}_{ji} + u_i\right)
$$

where 

$$
R_j = \{i \,|\, i \in S(u), i \neq  j,  \text{card}(S_{ji}(\chi)) > 0\}
$$ 

is the set of all relevant items.

Basically, $$\text{dev}_{ji}$$ is the average score difference between $$j$$ and $$i$$ from all the users who have used both of them. Then, AppJoy predicts $$u_j$$ based on $$u_i$$ by adding $$\text{dev}_{ji}$$ to $$u_i$$ and taking an average for all relevant applications $$i$$. The simplicity of this approach also makes it easy to implement, and its prediction accuracy is comparable to more sophisticated and computationally expensive algorithms.

One of the drawbacks of simple Slope One is that the number of scores observed is not taken into consideration. Assuming that we are given the scores of user $$u$$ on applications $$i$$ and $$k$$ to predict the score of user $$u$$ on application $$j$$. If 1000 users have used $$i$$ and $$j$$ whereas only 10 users have used $$k$$ and $$j$$, the score of user $$u$$ on $$i$$ is likely to be a much better predictor for $$j$$ than the score of user $$u$$ on $$k$$ is. Taking this into account, the prediction can be changed to:

$$
P_w(u_j) = \frac{\sum_{i \in R_{uj}} (\text{dev}_{ji} + u_i) \text{card}(S_{ji})}{\sum_{i \in R_{uj}} \text{card}(S_{ji})} \tag{1}
$$

# Algorithm: AppJoy Similarity Matrix

for each ordered vector of app ids $$T_u$$ by user $$u \in U$$ do

$$\quad$$ while size($$T_u$$) > 1 do

$$\quad$$ $$\quad$$ find an application $$p \in T_u$$ with score $$v_{up}$$

$$\quad$$ $$\quad$$ set $$T_u = T_u - p$$

$$\quad$$ $$\quad$$ for each application $$q \in T_u$$ with score $$v_{uq}$$ do

$$\quad$$ $$\quad$$ $$\quad$$ find $$\text{Diff}(p, q) = v_{up} - v_{uq}$$

$$\quad$$ $$\quad$$ $$\quad$$ set $$\text{Count}(p, q) = \text{Count}(p, q) + 1$$

$$\quad$$ $$\quad$$ end for

$$\quad$$ end while

end for

for each pair application $$p$$ and $$q$$ do 

$$\quad$$ find $$\text{Diff}(p, q) = \text{Diff}(p, q) / \text{Count}(p, q)$$

end for

return Diff

This approach is called the Weighted Slope One and the Algorithm shows how to compute the similarity matrix of the applications using this approach. 

Once the application similarity matrix is computed, AppJoy makes recommendations for the user $u$ as follows. For an application $j$ not used by the user $u$, AppJoy can calculate its weighted slope one score $Pw(u_j)$ using Equation 1, while the $dev_{ji}$ can be looked up in the similarity matrix. AppJoy computes the scores for all applications $j$ not in $S(u)$, if there is a $Diff(ji)$ entry for any $i$ in $S(u)$, and returns the top $N$ applications with the highest scores.
