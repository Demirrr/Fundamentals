# Fundamentals

With this collection of notebooks, I would like to share my experience of learning and researching machine learning techniques. To this end, on each notebook, the reader is firstly provided with a brief mathematical background knowledge. Therafter, I provided a naive Python implementation of the respective technique. During the implementation, I focused on the readability.

Recreating techniques via implementing them from scratch necessitates understanding atomic details of the respective techniques. Listen 
+ [Bottou](https://youtu.be/adXwym8Lakg?t=5307)
+ [Yann LeCun](https://youtu.be/Svb1c6AkRzE?t=693)
+ [Yoshua Bengio](https://youtu.be/pnTLZQhFpaE?t=1269), and 
+ [Andrej Karpathy](https://youtu.be/_au3yw46lcg?t=782). 
They can not be all wrong, can they ? :)

Feel free to use/modify any code provided in this repository. However, I have a request from you.
Please do not forget that **“No one has ever become poor by giving.” – Anne Frank**. 
Please make a small donation to [World Food Programme](https://donatenow.wfp.org/).

conda create -n fun python=3.10 --no-default-packages && conda activate fun
pip install numpy
pip install matplotlib
pip install seaborn
# Content
Prerequisite: Linear Algebra
1. Machine Learning
   1. Naive Bayes
   2. Regression
   3. Maximum Likelihood vs Maximum A Posteriori Estimation
   4. Support Vector Machines
   5. Loss Funciton Langscape
   6. Kernalization
   7. Generative vs Discriminator
   8. Gaussian Process
   9. Bayesian Optimization
   10. Bagging
   11. From Decision Tree to Random Forest
   12. Boosting

2. Deep Learning
   1. Linear Classification
   2. Vanilla Nets
   3. Convolutions
   4. Forward and Backward Passes in Nets
   5. Optimization as API
   6. Vanishing Gradient
   7. Batch Normalization
   8. Dropout
   9. Recurrent Nets
   10. LSTM
   11. Generative Adversarial Network
   12. Graph Convolutional Networks
   13. Laplace Redux

3. Reinforcement Learning
   1. Search
   2. MDP
   3. RL
   4. Deep Q-Network

4. Numerical Optimization
   1. Descent them all:
      1. Gradient Descent
      2. Stochastic Gradient Descent
      3. Momentum
      4. Nesterov's Momentum
5. NLP
6. Visualization
7. Programming
8. ?
9. ?
10. Algorithms
    1. Data Structures
       1. Hash Table
       2. Linked List
       3. Queue
       4. Stack
    2. Search
       1. Breath First Search
       2. Depth First Search
       3. Bellman Ford
       4. Dijkstra