#coding: utf-8
from datetime import datetime
import os
import pickle
import numpy as np
import tensorflow as tf

a = datetime.now()
# Parameters
learning_rate = 0.0001
training_epochs = 2
batch_size = 16
display_step = 1

# Network Parameters
n_classes = 2 # MNIST total classes (0-9 digits)

modelPathIndex = 'mynet_3dconv/save_net.ckpt.index'
modelPath = 'mynet_3dconv/save_net.ckpt'

with open('val_data_2.pkl') as f:
	data = pickle.load(f)
	labels = []
	for item in data[:, 1]:
		a = np.zeros((1, 2))[0]
		if item == 0:
			a[0] = 1
		else:
			a[1] = 1
		labels.append(a)
	labels = np.array(labels)
	data = data[:, 0]

	x_test = data[0:300]
	arr = []
	for item in x_test:
		arr.append(item.reshape((1, -1))[0])
	x_test = np.array(arr)
	x_train = data[301:]
	arr = []
	for item in x_train:
		arr.append(item.reshape((1, -1))[0])
	x_train = np.array(arr)

	y_test = labels[0:300]
	y_train = labels[301:]
	
	del data

def batches(batch_size, data_size):
    arr = []
    n_batches = np.ceil(float(data_size) / batch_size)
    for i in range(int(n_batches)):
        tail = (i + 1) * batch_size - 1
        head = (i + 1) * batch_size - 1 - (batch_size - 1)
        if tail >= data_size:
            tail = data_size - 1
        arr.append((head, tail))
    return arr

train_batches = batches(batch_size,  x_train.shape[0])
test_batches = batches(batch_size, x_test.shape[0])


# tf Graph input
x = tf.placeholder('float', [None, 60*60*60])
x_image = tf.reshape(x, shape = [-1, 60, 60, 60, 1])
y = tf.placeholder('float', [None, n_classes])

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
	result = sess.run(accuracy, feed_dict={x: v_xs, y: v_ys})
	return result

def weightVariable(shape, name):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1), name = name)

def biaseVariable(shape, name):
	return tf.Variable(tf.constant(0.1, shape = shape), name = name)

def conv3d(x, W):
	return tf.nn.conv3d(x, W, strides = [1, 1, 1, 1, 1], padding = 'SAME')

def max_pool(x):
	return tf.nn.max_pool3d(x, ksize = [1,2,2,2,1], strides = [1,2,2,2,1], padding = 'VALID')

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

## conv1 layer ##
W_conv1 = weightVariable([3, 3, 3, 1, 32], 'W_conv1')
b_conv1 = biaseVariable([32], 'b_conv1')
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
p_conv1 = max_pool(h_conv1)

## conv2 layer ##
W_conv2 = weightVariable([3, 3, 3, 32, 64], 'W_conv2')
b_conv2 = biaseVariable([64], 'b_conv2')
h_conv2 = tf.nn.relu(conv3d(p_conv1, W_conv2) + b_conv2)
p_conv2 = max_pool(h_conv2)

## conv3 layer ##
W_conv3 = weightVariable([3, 3, 3, 64, 128], 'W_conv3')
b_conv3 = biaseVariable([128], 'b_conv3')
h_conv3 = tf.nn.relu(conv3d(p_conv2, W_conv3) + b_conv3)
#p_conv3 = max_pool(h_conv3)

## conv4 layer ##
W_conv4 = weightVariable([4, 4, 4, 128, 128], 'W_conv4')
b_conv4 = biaseVariable([128], 'b_conv4')
h_conv4 = tf.nn.relu(tf.nn.conv3d(h_conv3, W_conv4, strides = [1, 1, 1, 1, 1], padding = 'VALID') + b_conv4)
#p_conv4 = max_pool(h_conv4)

## conv5 layer ##
W_conv5 = weightVariable([3, 3, 3, 128, 256], 'W_conv5')
b_conv5 = biaseVariable([256], 'b_conv5')
h_conv5 = tf.nn.relu(conv3d(p_conv1, W_conv2) + b_conv2)
p_conv5 = max_pool(h_conv2)

## func1 layer ##
W_func1= weightVariable([15*15*15*64, 32], 'W_func1')
b_func1 =biaseVariable([32], 'b_func1')
p_conv2_flat = tf.reshape(p_conv2, [-1, 15*15*15*64])
h_func1 = tf.nn.relu(tf.matmul(p_conv2_flat, W_func1) + b_func1)
# h_func1_drop = tf.nn.dropout(h_func1)

## func2 layer ##
W_func2= weightVariable([32, 2], 'W_func2')
b_func2 =biaseVariable([2], 'b_func2')
pred = tf.nn.softmax(tf.matmul(h_func1, W_func2) + b_func2)

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
		for item in train_batches:
			head = item[0]
			tail = item[1]
			x_train_min = x_train[head:tail]
			y_train_min = y_train[head:tail]
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([optimizer, cost], feed_dict={x: x_train_min, y: y_train_min})
			avg_cost += c
		
		# Compute average loss
		n_batches = np.ceil(float(x_train.shape[0]) / batch_size)
		avg_cost /= n_batches
		
		# Display logs per epoch step
		if epoch % display_step == 0:
			print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))
		    #print('Epoch:', '%04d' % (epoch+1), \
		        #'cost=', '{:.9f}'.format(avg_cost), \
		        #'accuracy:', '{:.9f}'.format(compute_accuracy(x_test, y_test)))
	print('Optimization Finished!')
	saver.save(sess, modelPath)
