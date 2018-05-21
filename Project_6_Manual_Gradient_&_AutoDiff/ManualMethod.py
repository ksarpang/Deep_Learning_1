# Manual method for gradient descent

import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

housing = fetch_california_housing()
m,n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)),housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")

XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)

with tf.Session() as sess:
    print(theta.eval())

theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name="theta")
y_pred = tf.matmul(X,theta,name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name = "mse")
gradients = 2/m*tf.matmul(XT, error)
learning_rate = 0.01
training_op = tf.assign(theta, theta - learning_rate* gradients)
init = tf.global_variables_initializer()
graph = tf.get_default_graph()
operations = graph.get_operations()



with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs', graph)
    writer.close()


graph = tf.get_default_graph()
operations = graph.get_operations()
print(len(operations))
