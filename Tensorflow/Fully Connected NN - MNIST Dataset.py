"""
A 3 layer fully connected neural network
  L1 - 300 Neurons
  L2 - 10 Neurons
"""

## IMPORT LIBRARIES
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

## DEFINE THE COMPUTATIONAL MAP

    # optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 50

    # placeholders for the training data
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

    # weights and biases
W1 = tf.Variable(tf.random_normal([784, 300], stddev = 0.03))
b1 = tf.Variable(tf.random_normal([300]))

W2 = tf.Variable(tf.random_normal([300, 10], stddev = 0.03))
b2 = tf.Variable(tf.random_normal([10]))

    # neural network equations
A1 = tf.nn.relu(tf.matmul(x, W1) + b1)
A2 = tf.nn.softmax(tf.matmul(A1, W2) + b2)

    # cost function
        # set bounds for A2 so that A2 doesnot have 0 as a value or a value greater than 1
A2_bounded = tf.clip_by_value(A2, 1e-10, 0.99999)
        # cross entropy
cost = -tf.reduce_mean(tf.reduce_sum(y * tf.log(A2_bounded)+ (1 - y) * tf.log(1 - A2_bounded)), axis=1)
    # optimiser
    
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

    # evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(A2, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## RUN THE COMPUTATIONAL MAP

    # initialise variables
init = tf.global_variables_initializer()

    # Session 
with tf.Session() as sess:
    sess.run(init)
    num_batches = int(len(mnist.train.labels)/batch_size)
    
    for epoch in range(epochs):
        avg_cost = 0        
        for i in range(num_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
            _, c = sess.run([optimiser, cost], feed_dict = {x:batch_x, y:batch_y})
            avg_cost += c /num_batches 
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    # evalute the model
    print(sess.run(accuracy, feed_dict = {x:mnist.test.images, y:mnist.test.labels} ))
