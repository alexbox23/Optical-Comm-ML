import tensorflow as tf
import csv

# parses one column from the csv file
# header is the number of rows at the beginning that don't contain data
def csv_parser(file, column, header):
    ret = []
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if header > 0:
                header -= 1
            else:
                ret.append(float(row[column]))
    return ret

data = csv_parser('data/data_Binary_NRZ_RX(small).csv', 1, 7)
labels = csv_parser('data/labels_Binary_NRZ_TX.csv', 0, 0)
samples_per_label = 16

training_portion = 1/2
training_size = int(len(data)*training_portion)
training_set = data[:training_size]
test_set = data[training_size:]
training_labels = labels[:int(training_size/samples_per_label)]
test_labels = labels[int(training_size/samples_per_label):]

training_epochs = 100
num_layers = 4
learning_rate = 0.01
input_samples = 2 ** num_layers
batch_size = samples_per_label * 4

x = tf.placeholder(tf.float32, [input_samples, None])
y = tf.placeholder(tf.float32, [1, None])

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
    prod = tf.matmul(weights[i], input_vector)
    hidden_out = tf.add(prod, biases[i])
    hidden_out = tf.nn.relu(hidden_out)
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
    # grad_W = tf.matrix_band_part(grad_W, 0, 1)
    weight_gradients.append(grad_W)
    bias_gradients.append(grad_b)

    new_W = W.assign(W - learning_rate * grad_W)
    new_b = b.assign(b - learning_rate * grad_b)
    weight_updates.append(new_W)
    bias_updates.append(new_b)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("training...")
    for epoch in range(training_epochs):
        avg_cost = 0
        batch_count = 0
        for i in range(0, len(training_set), batch_size):
            if i + batch_size <= len(training_set) - input_samples:
                batch_y = [[]]
                batch_x = [[] for j in range(input_samples)]
                j = i
                while j < i + batch_size:
                    batch_y[0].append(training_labels[(j + input_samples - 1)//samples_per_label])
                    for k in range(input_samples):
                        batch_x[k].append(training_set[j + k])
                    j += 1
                fetches = {'W': weight_updates, 'b': bias_updates, 'c': cost}
                result_dict = sess.run(fetches, feed_dict={x: batch_x, y: batch_y})
                avg_cost += result_dict['c']
                batch_count += 1

        avg_cost /= batch_count
        if epoch % 10 == 0 or epoch == training_epochs - 1:
            print("epoch: " + str(epoch) + "\t avg cost: " + str(avg_cost))

    print("done training.")

    test_input = [[] for i in range(samples_per_label)]
    for i in range(len(test_labels)):
        single_input = test_set[i*samples_per_label:(i+1)*samples_per_label]
        for j in range(samples_per_label):
            test_input[j].append(single_input[j])

    # maximum distance between prediction and label before prediction is incorrect
    threshold = tf.constant(0.5)

    wrong_predictions = tf.greater(tf.abs(tf.subtract(hidden_outputs[-1], y)), threshold)
    misses = tf.reduce_sum(tf.cast(wrong_predictions, tf.uint8))
    missed_bits = misses.eval({x: test_input, y: [test_labels]})
    print("test cost: " + str(cost.eval({x: test_input, y: [test_labels]})))
    print("missed bits: " + str(missed_bits))

    print(hidden_outputs[-1].eval({x: test_input, y: [test_labels]})[:8])
    print(test_labels[:8])

    "TODO:
        change to classifier
        add pooling layers 
        cost function: cross entropy

        start at 2nd bit to do dilated convolution, window lag = 16, sweep over all 16 samples. 
        dilated convolution layer: k filters = number of bits squared = 4 for binary, 16 for pam4
        max pooling layer: down sample to 8 samples per bit
        dilated convolution layer: k filters
        max pooling: down sample to 4
        concatenate k x k into single waveform
        matrix of weights to classes 



    "
