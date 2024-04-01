print('\n\n--------------------------------------------------\n\n')

# Define training data
inputs  = [1,2,3,4]
targets = [2,4,6,8]

# Define model parameters
w = 0.1     # Weight
learning_rate = 0.1
epochs = 10

def predict(i):
    return w*i

# Train the network
for _ in range(epochs):
    print(f'epoch: {_}')
    # Feed forward: Calculate predictions for all training data.
    pred   = [predict(i) for i in inputs]
    print(f'pred: {pred}')

    # Calculate the cost over all training data
    errors_sq = [(p-t)**2 for p, t in zip(pred, targets)]
    cost = sum(errors_sq)/len(errors_sq)
    print(f'errors: {[p-t for p,t in zip(pred, targets)]}')
    print(f'errors_sq: {errors_sq}')
    print(f'Weight: {w:.2f}, Cost: {cost:.2f}')

    # Back propagation: Calculate the error derivatives and weight deltas.
    errors_d = [2*(p-t) for p, t in zip(pred, targets)]
    weight_d = [e*i for e, i in zip(errors_d, inputs)]
    w -= learning_rate* sum(weight_d)/len(weight_d)
    print(f'errors_d: {errors_d}, weight_d: {weight_d}')
    print(f'weight adjustment: {sum(weight_d)/len(weight_d)}')
    print('\n------------------\n')

# Test Data
test_inputs  = [5,6]
test_targets = [10,12]
test_preds = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_targets, test_preds):
    print(f'Inputs: {i}, Targets: {t}, Preds: {p:.4f}')

'''
For linear regression, the preferred cost function is meas squared error.

If we use the mean squared error cost function directly, and don't have our 
parameters adjusted appropriately, the errors can oscillate and grow very 
quickly, and lead to an overflow error. (For example, w=0.1 learning_rate=0.1)
So instead of 

We minimize MSE by finding the point where the derivative is zero. 
The sign of the derivative of the cost function tells us the direction that 
the weights should be adjusted.

We scale the error derivatives by the input value to get the weight delta.
'''