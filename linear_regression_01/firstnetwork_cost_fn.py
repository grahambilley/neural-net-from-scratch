# Define training data
inputs  = [1,2,3,4]
targets = [2,4,6,8]

# Define model parameters
w = 0.1     # Weight
learning_rate = 0.1
epochs = 25

def predict(i):
    return w*i

# Define cost functions
# Mean Squared Errors
def msqe(errors):
    sq_err = [i**2 for i in errors]
    return sum(sq_err)/len(sq_err)

# Train the network
for _ in range(epochs):
    pred   = [predict(i) for i in inputs]
    errors = [t-p for p, t in zip(pred, targets)]
    sq_err = [i**2 for i in errors]
    cost   = msqe(errors)
    print(f'Weight: {w:.2f}, Cost: {cost:.2f}')
    # Back propagation
    w += learning_rate*cost

# Test Data
test_inputs  = [5,6]
test_targets = [10,12]
test_preds = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_targets, test_preds):
    print(f'Inputs: {i}, Targets: {t}, Preds: {p:.4f}')

'''
For linear regression, the preferred cost function is mean squared error.

If we use the mean squared error cost function directly, and don't have our 
parameters adjusted appropriately, the errors can oscillate and grow very 
quickly, and lead to an overflow error. (For example, w=0.1 learning_rate=0.1)

Instead of coding the cost function explicitly, calculate the derivative of
the cost function. Minimize the cost function by moving toward where the 
derivative is zero. The sign of the derivative tells us the direction we need
to adjust the weights.

Sum the derivatives across all training samples. This is batch gradient descent.
'''