# Define training data
inputs  = [1,2,3,4]
targets = [2,4,6,8]

# Define model parameters
w = 0.1     # Weight

def predict(i):
    return w*i

# Train the network
pred   = [predict(i) for i in inputs]
errors = [t-p for p, t in zip(pred, targets)]
cost   = sum(errors)/len(errors)
print(f'Weight: {w:.2f}, Cost: {cost:.2f}')
