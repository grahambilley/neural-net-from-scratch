import mnistreader as reader
import random
import math

def softmax(predictions):
    m = max(predictions)
    temp = [math.exp(p-m) for p in predictions]
    total = sum(temp)
    return [t/total for t in temp]

def log_loss(activations, targets):
    losses = [-t * math.log(a) - (1-t)*math.log(1-a) for a, t in zip(activations, targets)]
    return sum(losses)

epochs = 3
batch_size = 400
learning_rate = 0.4
input_count, hidden_count, output_count = 784, 20, 10

w_i_h = [[random.random() - 0.5 for _ in range(input_count)] for _ in range(hidden_count)]
w_h_o = [[random.random() - 0.5 for _ in range(hidden_count)] for _ in range(output_count)]
b_i_h = [0 for _ in range(hidden_count)]
b_h_o = [0 for _ in range(output_count)]

# Train the network
for epoch in range(epochs):
    for labels, targets, inputs in reader.get_training_samples(batch_size):
        pred_h = [[sum([w*a for w,a in zip(weights, inp)]) + 
                   bias for weights, bias in zip(w_i_h, b_i_h)] for inp in inputs]
        act_h = [[max(0,p) for p in pred] for pred in pred_h]
        pred_o = [[sum([w*a for w,a in zip(weights, inp)]) + 
                   bias for weights, bias in zip(w_h_o, b_h_o)] for inp in act_h]
        act_o = [softmax(predictions) for predictions in pred_o]
        
        cost = sum([log_loss(a, t) for a,t in zip(act_o, targets)]) / len(act_o)
        print(f'epoch:{epoch} cost:{cost:.4}')

        # Error derivatives
        errors_d_o = [[a-t for a,t in zip(ac,ta)] for ac,ta in zip(act_o, targets)]

        # Calculate error derivatives for the hidden layer
        who_T = list(zip(*w_h_o))
        errors_d_h = [[sum([d*w for d,w in zip(deltas, weights)]) * (0 if p <= 0 else 1)
                       for weights, p in zip(who_T, pred)] for deltas, pred in zip(errors_d_o, pred_h)]
        
        # Gradient Hidden -> Output
        act_h_T = list(zip(*act_h))
        errors_d_o_T = list(zip(*errors_d_o))
        who_d = [[sum([d*a for d,a in zip(deltas, act)]) for deltas in errors_d_o_T]
                 for act in act_h_T]
        bho_d = [sum([d for d in deltas]) for deltas in errors_d_o_T]

        # Gradient for Input -> Hidden
        inputs_T = list(zip(*inputs))
        errors_d_h_T = list(zip(*errors_d_h))
        wih_d = [[sum([d*a for d,a in zip(deltas, act)]) for deltas in errors_d_h_T]
                 for act in inputs_T]
        bih_d = [sum([d for d in deltas]) for deltas in errors_d_h_T]

        # Update weights and biases for all layers
        who_d_T = list(zip(*who_d))
        for y in range(output_count):
            for x in range(hidden_count):
                w_h_o[y][x] -= learning_rate * who_d_T[y][x] / len(inputs)
            b_h_o[y] -= learning_rate * bho_d[y] / len(inputs)

        wih_d_T = list(zip(*wih_d))
        for y in range(hidden_count):
            for x in range(input_count):
                w_i_h[y][x] -= learning_rate * wih_d_T[y][x] / len(inputs)
            b_i_h[y] -= learning_rate * bih_d[y] / len(inputs)

# Test the network
pred_h = [[sum([w*a for w,a in zip(weights, inp)]) + 
           bias for weights, bias in zip(w_i_h, b_i_h)] for inp in inputs]
act_h = [[max(0,p) for p in pre] for pre in pred_h]
pred_o = [[sum([w*a for w,a in zip(weights, inp)]) + bias
           for weights, bias in zip(w_h_o, b_h_o)] for inp in act_h]
act_o = [softmax(predictions) for predictions in pred_o]

correct = 0
for a,t,i in zip(act_o, targets, inputs):
    ma_neuron = a.index(max(a))
    ma_target = t.index(max(t))
    if ma_neuron == ma_target:
        correct += 1
        
print(f'Correct: {correct}/{len(inputs)} : ({correct / len(inputs):.2%})')
