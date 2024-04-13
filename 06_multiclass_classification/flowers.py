import flowersdata as data
import math

weights = [[0.1, 0.2], [0.15, 0.25], [0.18, 0.1]]
biases = [0.3, 0.4, 0.35]
epochs = 1
learning_rate = 0.1

def softmax(predictions):
    m = max(predictions)
    temp = [math.exp(p-m) for p in predictions]
    total = sum(temp)
    return [t/total for t in temp]

def log_loss(activations, targets):
    losses = [-t * math.log(a) - (1-t)*math.log(1-a) for a, t in zip(activations, targets)]
    return sum(losses)

# train the network
for epoch in range(epochs):
    pred = [[sum([w*i for w, i in zip(we, inp)]) + 
             bi for we, bi in zip(weights, biases)] for inp in data.inputs]
    act = [softmax(p) for p in pred]
    # print(act)
    # print(sum(act[0]))  # Should be 1.0

    # Use log_loss to calculate the cost over all training samples
    cost = sum([log_loss(ac, ta) for ac, ta in zip(act, data.targets)]) / len(act)
    print(f'ep:{epoch}, c:{cost:.4f}')

    # print(pred)
    # print(f'row count: {len(pred)}')
    # print(f'column count: {len(pred[0])}')

