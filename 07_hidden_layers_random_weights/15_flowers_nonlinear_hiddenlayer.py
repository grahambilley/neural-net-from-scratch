import flowersdata_nonlinear as data
import math
import random

def softmax(predictions):
    m = max(predictions)
    temp = [math.exp(p-m) for p in predictions]
    total = sum(temp)
    return [t/total for t in temp]

def log_loss(activations, targets):
    losses = [-t * math.log(a) - (1-t)*math.log(1-a) for a, t in zip(activations, targets)]
    return sum(losses)

epochs = 5000
learning_rate = 0.3
input_count, hidden_count, output_count = 2, 8, 3

# Create the weights matrices
# wih = Weights for Input to Hidden Layer
# who = Weights for Hidden to Output Layer
# bih = Biases for Input to Hidden Layer
# bho = Biases for Hidden to Output Layer

# wih = [[0.1, -0.2], [-0.3, 0.25], [0.12, 0.23], [-0.11, -0.22]] # 4 Hidden Neurons
# who = [[0.2, 0.17, 0.3, -0.11], [0.3, -0.4, 0.5, -0.22], [0.12, 0.23, 0.15, 0.33]]
# bih = [0.2, 0.34, 0.21, 0.44]     # 4 hidden neurons
# bho = [0.3, 0.29, 0.37]     # 3 Output Neurons

# Initialize weights randomly, on a uniform [-0.5, 0.5]
wih = [[random.random() - 0.5 for _ in range(input_count)] for _ in range(hidden_count)]
who = [[random.random() - 0.5 for _ in range(hidden_count)] for _ in range(output_count)]
# Initialize biases as 0
bih = [0 for _ in range(hidden_count)]
bho = [0 for _ in range(output_count)]

for epoch in range(epochs):
    pred_h = [[sum([w*a for w,a in zip(weights, inp)]) + 
               bias for weights, bias in zip(wih, bih)] for inp in data.inputs]
    
    # print(len(pred_h))  # should be 60 training samples
    # print(len(pred_h[0]))  # should be 4 (hidden layer) neurons

    act_h  = [[max(0,p) for p in pred] for pred in pred_h] # applly ReLU
    pred_o = [[sum([w*a for w,a in zip(weights, inp)]) + 
               bias for weights, bias in zip(who, bho)] for inp in act_h]
    act_o  = [softmax(predictions) for predictions in pred_o]

    cost = sum([log_loss(a,t) for a,t in zip(act_o, data.targets)]) / len(act_o)
    print(f'epoch: {epoch} cost:{cost:.4f}')

    # Error derivatives
    errors_d_o = [[a-t for a,t in zip(ac,ta)] for ac,ta in zip(act_o, data.targets)]

    # Calculate error derivatives for the hidden layer
    who_T = list(zip(*who))
    errors_d_h = [[sum([d*w for d,w in zip(deltas, weights)]) * (0 if p <= 0 else 1)
                   for weights, p in zip(who_T, pred)] for deltas, pred in zip(errors_d_o, pred_h)]
    
    # Gradient Hidden -> Output
    act_h_T = list(zip(*act_h))
    errors_d_o_T = list(zip(*errors_d_o))
    who_d = [[sum([d*a for d,a in zip(deltas, act)]) for deltas in errors_d_o_T]
             for act in act_h_T]
    bho_d = [sum([d for d in deltas]) for deltas in errors_d_o_T]

    # Gradient for Input -> Hidden
    inputs_T = list(zip(*data.inputs))
    errors_d_h_T = list(zip(*errors_d_h))
    wih_d = [[sum([d*a for d,a in zip(deltas, act)]) for deltas in errors_d_h_T]
             for act in inputs_T]
    bih_d = [sum([d for d in deltas]) for deltas in errors_d_h_T]

    # Update weights and biases for all layers
    who_d_T = list(zip(*who_d))
    for y in range(output_count):
        for x in range(hidden_count):
            who[y][x] -= learning_rate * who_d_T[y][x] / len(data.inputs)
        bho[y] -= learning_rate * bho_d[y] / len(data.inputs)

    wih_d_T = list(zip(*wih_d))
    for y in range(hidden_count):
        for x in range(input_count):
            wih[y][x] -= learning_rate * wih_d_T[y][x] / len(data.inputs)
        bih[y] -= learning_rate * bih_d[y] / len(data.inputs)

# Test the network
pred_h = [[sum([w*a for w,a in zip(weights, inp)]) + 
           bias for weights, bias in zip(wih, bih)] for inp in data.test_inputs]
act_h = [[max(0,p) for p in pre] for pre in pred_h]
pred_o = [[sum([w*a for w,a in zip(weights, inp)]) + bias
           for weights, bias in zip(who, bho)] for inp in act_h]
act_o = [softmax(predictions) for predictions in pred_o]

correct = 0
for a,t in zip(act_o, data.test_targets):
    if a.index(max(a)) == t.index(max(t)):
        correct += 1

print(f'Correct: {correct}/{len(act_o)} ({correct / len(act_o):%})')














