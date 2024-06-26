# If there are 3 categories, instead of using the output from
# one neuron, a better way is to create 3 output neurons. 
# The neuron that is activated most determines the category.

inputs = [(0.0000, 0.0000), (0.2778, 0.2500), (0.2778, 0.9375), (0.9167, 0.6563),
    (0.4167, 0.2500), (0.3611, 0.3438), (0.3333, 0.4063), (0.9722, 0.3750),
    (0.0833, 0.3438), (0.6389, 0.3438), (0.4167, 0.6875), (0.7500, 0.6875),
    (0.0833, 0.1875), (0.9167, 0.5313), (0.1389, 0.2500), (0.8333, 0.6250),
    (0.8056, 0.6250), (0.1944, 1.0000), (0.8333, 0.5625), (0.4167, 1.0000),
    (1.0000, 0.6875), (0.4722, 0.6563), (0.3611, 0.5625), (0.4722, 0.8438),
    (0.1667, 0.3125), (0.4167, 0.9375), (0.3611, 0.9688), (0.9167, 0.3438),
    (0.0833, 0.0313), (0.3333, 0.8750)]

red   = (1, 0, 0)
green = (0, 1, 0)
blue  = (0, 0, 1)

targets = [red, red, blue, green, red, red, red, green, red, green, blue, green, red,
    green, red, green, green, blue, green, blue, green, blue, blue, blue,
    red, blue, blue, green, red, blue]

test_inputs = [(0.0278, 0.0313), (0.0556, 0.0625), (0.1111, 0.1563), (0.3611, 0.3750),
    (0.2778, 0.3438), (0.8333, 0.3750), (0.5556, 0.4375), (0.8333, 0.5313),
    (0.8611, 0.6563), (0.8056, 0.5625), (0.4722, 0.6563), (0.3611, 0.5625),
    (0.4722, 0.8438), (0.3611, 0.9688), (0.4167, 0.9375)]

test_targets = [red, red, red, red, red, green, green, green,
    green, green, blue, blue, blue, blue, blue]