#coding: utf-8
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
import os
mnist = input_data.read_data_sets(".", one_hot=True)

import tensorflow as tf

a = datetime.now()
# Parameters
learning_rate = 0.001
training_epochs = 2
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

modelPathIndex = 'mynet/save_net.ckpt.index'
modelPath = 'mynet/save_net.ckpt'

# tf Graph input
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])

def savedModelExists(path):
    if os.path.exists(path):
        return True
    else:
        return False

def compute_accuracy(v_xs, v_ys):
    # global predition
    # y_pre = sess.run(predition, feed_dict={xs: v_xs, keep_prod: 1})
    # correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    # return result
    # Test model
    global pred
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    result = sess.run(accuracy, feed_dict={x: v_xs, y: v_ys})
    return result
    # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    #
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # global predition
    # # y_pre = sess.run(predition, feed_dict={xs: v_xs, keep_prod: 1})
    # correct_prediction = tf.equal(tf.argmax(predition, 1), tf.argmax(v_ys, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prod: 1})
    # return result

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    #we can add dropout layer
    # drop_out = tf.nn.dropout(layer_2, 0.75)


    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    saver = tf.train.Saver()
    if savedModelExists(modelPathIndex):
        saver.restore(sess, modelPath)
    else:
        sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print('Epoch:', '%04d' % (epoch+1), \
                'cost=', '{:.9f}'.format(avg_cost), \
                'accuracy:', '{:.9f}'.format(compute_accuracy(mnist.test.images, mnist.test.labels)))
    print('Optimization Finished!')
    saver.save(sess, modelPath)

    # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    # print("Accuracy:", compute_accuracy(mnist.test.images, mnist.test.labels))

b = datetime.now()
print((b - a).seconds)