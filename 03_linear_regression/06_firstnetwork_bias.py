print('\n\n--------------------------------------------------\n\n')

# Define training data
inputs  = [1,2,3,4]
targets = [12,14,16,18]

# Define model parameters
w = 0.1     # Weight
b = 0.3     # Bias
learning_rate = 0.1
epochs = 100

def predict(i):
    return w*i + b

# Train the network
for _ in range(epochs):
    # Feed forward: Calculate predictions for all training data.
    pred   = [predict(i) for i in inputs]
    # Calculate the cost over all training data
    errors_sq = [(p-t)**2 for p, t in zip(pred, targets)]
    cost = sum(errors_sq)/len(errors_sq)
    print(f'Weight: {w:.2f}, Bias:{b:.2f}, Cost: {cost:.2f}')

    # Back propagation: Calculate the error derivatives and weight deltas.
    errors_d = [2*(p-t) for p, t in zip(pred, targets)]
    weight_d = [e*i for e, i in zip(errors_d, inputs)]
    bias_d   = [e*1 for e in errors_d]
    w -= learning_rate* sum(weight_d)/len(weight_d)
    b -= learning_rate* sum(bias_d)/len(bias_d)

# Test Data
print('-----------------------------')
test_inputs  = [5,6]
test_targets = [20,22]
test_preds = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_targets, test_preds):
    print(f'Inputs: {i}, Targets: {t}, Preds: {p:.4f}')
print('-----------------------------')

'''
The previous examples forced the y-intercept to be zero. 
Here we will adjust the training data by moving the targets all up by 10, and 
introduce a bias to the neural network.
'''