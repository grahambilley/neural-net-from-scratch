import random

def get_training_samples(batch_size):
    with open('train.csv') as file:
        text = file.read()
        
    # The dataset is separated by newlines.
    # Put each datapoint into a list.
    textlines = text.strip().split('\n')
    # Shuffle the list.
    random.shuffle(textlines)
    # Pull out samples until the batch size is reached.
    start = 0
    while start < len(textlines):
        labels = []
        targets = []
        inputs = []
        end = start + batch_size
        for textline in textlines[start:end]:
            cells = textline.split(',')
            # The label is stored in the first position
            labels.append(int(cells[0]))
            # The label is one-hot encoded using the next 10 positions
            targets.append([float(c) for c in cells[1:11]])
            # The pixels of the image are stored in the rest of the list
            inputs.append([float(c) for c in cells[11:]])
        yield labels, targets, inputs
        start += batch_size
        

def get_test_samples():
    with open('test.csv', 'r') as file:
        text = file.read()
    textlines = text.strip().split('\n')
    # NOTE: There is no need to shuffle or batch the test data.
    labels = []
    targets = []
    inputs = []
    for textline in textlines:
        cells = textline.split(',')
        # The label is stored in the first position
        labels.append(int(cells[0]))
        # The label is one-hot encoded using the next 10 positions
        targets.append([float(c) for c in cells[1:11]])
        # The pixels of the image are stored in the rest of the list
        inputs.append([float(c) for c in cells[11:]])
    return labels, targets, inputs
