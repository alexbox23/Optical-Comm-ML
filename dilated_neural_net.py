import tensorflow as tf

training_epochs = 500
num_layers = 4
learning_rate = 0.01
input_samples = 2 ** num_layers
batch_size = 16

x = tf.placeholder(tf.float32, [input_samples, 1])
y = tf.placeholder(tf.float32, [1, 1])

weights = []
biases = []
for i in range(num_layers):
    tensor = tf.matrix_band_part(tf.random_normal([2**i, 2**(i+1)], stddev=0.03), 0, 1)
    W = tf.Variable(tensor, name='W'+str(i))
    weights.insert(0, W)
    b = tf.Variable(tf.random_normal([2**i, 1]), name='b'+str(i))
    biases.insert(0, b)

hidden_outputs = []
input_vector = x
for i in range(len(weights)):
    prod = tf.matmul(weights[i],input_vector)
    hidden_out = tf.add(prod, biases[i])
    # hidden_out = tf.nn.relu(hidden_out)
    hidden_outputs.append(hidden_out)
    input_vector = hidden_out

cost = tf.losses.mean_squared_error(y, hidden_outputs[-1])

weight_gradients = []
bias_gradients = []
weight_updates = []
bias_updates = []
for i in range(num_layers):
    W = weights[i]
    b = biases[i]
    grad_W, grad_b = tf.gradients(xs=[W, b], ys=cost)
    grad_W = tf.matrix_band_part(grad_W, 0, 1)
    weight_gradients.append(grad_W)
    bias_gradients.append(grad_b)

    new_W = W.assign(W - learning_rate * grad_W)
    new_b = b.assign(b - learning_rate * grad_b)
    weigt_updates.append(new_W)
    bias_updates.append(new_b)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0

        for i in range(batch_size):
            batch_
            "TODO: write csv import, train data"