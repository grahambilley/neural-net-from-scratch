# Batch gradient descent has the following problems:
# 1. It requires a lot of memory to calculate costs for all training datapoints.
# 2. It can get stuck in local minimums
# 
# Alternatives are:
# 1. Stochastic gradient descent. 
# Often works well, and is fast in general. Weights and biases are adjusted after each 
# training sample.
# 
# Our handwritten digits dataset has 20k training samples, each are 28x28 pixels.
# This means to calculate the cost for one neuron, 20k x 28 x 28 = 15.68 M calculations 
# have to be made, and everything needs to be stored in memory. Batch gradient descent 
# isn't suitable to this dataset. And stochastic gradient descent is fast. but is more 
# vulnerable to anomolies in the data. In this case, we will use mini batch gradient
# descent.
# 
# How?
# 1. Shuffle the dataset
# 2. Split the dataset into smaller baches
# 3. Process a batch and update the weights and biases
# 4. Repeat until all batches are processed.




