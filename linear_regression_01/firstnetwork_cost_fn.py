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
# Mean Sum of Errors
def mse(errors):
    return sum(errors)/len(errors)

# Train the network
for _ in range(epochs):
    pred   = [predict(i) for i in inputs]
    errors = [t-p for p, t in zip(pred, targets)]
    cost   = mse(errors)
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

'''