#coding: utf-8
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
import os
mnist = input_data.read_data_sets(".", one_hot=True)

import tensorflow as tf

a = datetime.now()
# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

modelPathIndex = 'mynet_conv/save_net.ckpt.index'
modelPath = 'mynet_conv/save_net.ckpt'

# tf Graph input
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])
x_image = tf.reshape(x, shape = [-1, 28, 28, 1])

def savedModelExists(path):
    if os.path.exists(path):
        return True
    else:
        return False

def compute_accuracy(v_xs, v_ys):
    global pred
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    result = sess.run(accuracy, feed_dict={x: v_xs, y: v_ys, keep_prob: 0.5})
    return result

def weightVariable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1), name = name)

def biaseVariable(shape, name):
    return tf.Variable(tf.constant(0.1, shape = shape), name = name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

## conv1 layer ##
W_conv1 = weightVariable([5, 5, 1, 32], 'W_conv1')
b_conv1 = biaseVariable([32], 'b_conv1')
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
p_conv1 = max_pool_2x2(h_conv1)

## conv2 layer ##
W_conv2 = weightVariable([5, 5, 32, 64], 'W_conv2')
b_conv2 = biaseVariable([64], 'b_conv2')
h_conv2 = tf.nn.relu(conv2d(p_conv1, W_conv2) + b_conv2)
p_conv2 = max_pool_2x2(h_conv2)

## func1 layer ##
W_func1= weightVariable([7*7*64, 128], 'W_func1')
b_func1 =biaseVariable([128], 'b_func1')
p_conv2_flat = tf.reshape(p_conv2, [-1, 7*7*64])
h_func1 = tf.nn.relu(tf.matmul(p_conv2_flat, W_func1) + b_func1)
h_func1_drop = tf.nn.dropout(h_func1, keep_prob)

## func2 layer ##
W_func2= weightVariable([128, 10], 'W_func2')
b_func2 =biaseVariable([10], 'b_func2')
pred = tf.nn.softmax(tf.matmul(h_func1_drop, W_func2) + b_func2)

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
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob:0.5})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print('Epoch:', '%04d' % (epoch+1), \
                'cost=', '{:.9f}'.format(avg_cost), \
                'accuracy:', '{:.9f}'.format(compute_accuracy(mnist.test.images, mnist.test.labels)))
    print('Optimization Finished!')
    saver.save(sess, modelPath)
    l = []
    map(l)
