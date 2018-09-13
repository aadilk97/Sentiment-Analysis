import pickle
import os
import numpy as np
import tensorflow as tf
import math


from sklearn.preprocessing import normalize

def dotproduct(v1, v2):
  	return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  	return math.sqrt(dotproduct(v, v))

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

# X_loaded = pickle.load(open('data_X_new.pkl', 'rb'))



X_loaded = np.array(pickle.load(open('twitter_X.pkl', 'rb')))
y_loaded = np.array(pickle.load(open('twitter_y.pkl', 'rb')))


# y_loaded = np.array(y_loaded)
# y_loaded = y_loaded.reshape((y_loaded.shape[0], 1))

X_train = X_loaded[0:11500]
X_test = X_loaded[11500:14639]

y_train = y_loaded[0:11500].reshape((11500, 1))
y_test = y_loaded[11500:14639]

print (X_train.shape)

# j = 10000
# for i in range(10000, 20000):
# 	if i % 1000 == 0:
# 		print (i)
# 	if y_loaded[i] == 0:
# 		X_train = np.vstack((X_train, X_loaded[i].reshape(1, 300, 20)))
# 		y_train = np.vstack((y_train, y_loaded[i].reshape(1, 1)))


# print (X_train.shape, y_train.shape)
# print (y_train[-1])




# ros = RandomOverSampler()
# X_train, y_train = ros.fit_sample(X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2])), y_train)

# X_train = X_train.reshape((X_train.shape[0], 300, 20))

# print (y_train.shape)
# class_freq = np.array(np.bincount(y_train[:, 0]))
# print (class_freq)
# print (X_train.shape, y_train.shape)




y_train = get_one_hot(y_train, 3)
y_test = get_one_hot(y_test, 3)



# weights = tf.constant([class_freq[1]/class_freq[0], class_freq[0]/class_freq[1]])
# print (weights)


X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

## Convolution 1
W1 = tf.Variable(tf.random_normal([3, 3, 1, 50]), dtype=tf.float32, name='W1')
b1 = tf.Variable(tf.random_normal([50]), dtype=tf.float32, name='b1')

x1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='VALID')
x1 = tf.add(x1, b1)
x1 = tf.nn.relu(x1)

## Max pooling 1
x1 = tf.nn.max_pool(x1, [1,2,2,1], strides=[1,2,2,1], padding='VALID')
x1 = tf.contrib.layers.layer_norm(x1)


## Convolution 2
W2 = tf.Variable(tf.random_normal([4, 4, 50, 100]), dtype=tf.float32, name='W2')
b2 = tf.Variable(tf.random_normal([100]), dtype=tf.float32, name='b2')

x2 = tf.nn.conv2d(x1, W2, strides=[1,1,1,1], padding='VALID')
x2 = tf.add(x2, b2)
x2 = tf.nn.relu(x2)
x2 = tf.contrib.layers.layer_norm(x2)



## Convolution 3
W3 = tf.Variable(tf.random_normal([4, 4, 100, 100]), dtype=tf.float32, name='W3')
b3 = tf.Variable(tf.random_normal([100]), dtype=tf.float32, name='b3')

x3 = tf.nn.conv2d(x2, W3, strides=[1,1,1,1], padding='VALID')
x3 = tf.add(x3, b3)
x3 = tf.nn.relu(x3)

## Max pooling 2
x3 = tf.nn.max_pool(x3, [1,2,2,1], strides=[1,1,1,1], padding='VALID')
x3 = tf.contrib.layers.layer_norm(x3)



## Fully connected 1
W4 = tf.Variable(tf.random_normal([512, 28400]), dtype=tf.float32, name='W4')
b4 = tf.Variable(tf.random_normal([512]), dtype=tf.float32, name='b4')

x3 = tf.reshape(x3, [-1, 28400])
fully_connected1 = tf.matmul(x3, tf.transpose(W4))
fully_connected1 = tf.add(fully_connected1, b4)
fully_connected1 = tf.nn.relu(fully_connected1)
fully_connected1 = tf.contrib.layers.layer_norm(fully_connected1)


##Drop out
fully_connected1 = tf.nn.dropout(fully_connected1, 0.5)


## Output 
W5 = tf.Variable(tf.random_normal([3, 512]), dtype=tf.float32, name='W5')
b5 = tf.Variable(tf.random_normal([3]), dtype=tf.float32, name='b5')

y_ = tf.matmul(fully_connected1, tf.transpose(W5))
y_ = tf.add(y_, b5)


output = tf.nn.softmax(y_)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)



init_g = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_g)

	# batch_x = X_train[0:10].reshape((10, 31, 300, 1))
	# batch_y = y_train[0:10]

	# # print (batch_x.shape)
	# xt = sess.run(y_, feed_dict={X: batch_x, y:batch_y})
	# print (xt.shape)



	# x3 = sess.run([x3], feed_dict={X: X_train[0:10].reshape(10, 21, 300, 1), y: y_train[0:10]})
	# print (np.array(x3).shape)


	batch_size = 50
	for epoch in range(20000):
		curr_pointer = 0
		epoch_cost = 0
		score = 0
		while curr_pointer < (X_train.shape[0]):
			batch_x = X_train[curr_pointer: curr_pointer + batch_size].reshape((batch_size, 21, 300, 1))
			batch_y = y_train[curr_pointer: curr_pointer + batch_size]
			curr_pointer += batch_size

			_, c, pred = sess.run([optimizer, cost, y_], feed_dict={X: batch_x, y:batch_y})
			epoch_cost += c

			#print ("Batch cost = ", c, "curr_pointer = ", curr_pointer)
			j = 0
			for i in range(batch_size):
				if np.argmax(pred[i]) == np.argmax(batch_y[i]):
					score += 1

			

		print ("Cost for epoch ", epoch, "= ", epoch_cost)
		print ("Training accuracy = ", score / X_train.shape[0])

		X_test = X_test.reshape((X_test.shape[0], 21, 300, 1))
		pred = sess.run([y_], feed_dict={X:X_test, y:y_test})
		pred = np.array(pred).reshape(X_test.shape[0], 3)

		
		score = 0
		for i in range(X_test.shape[0]):
			if np.argmax(pred[i]) == np.argmax(y_test[i]):
				score += 1

		print ("Acc = ", score / X_test.shape[0])
		print ()