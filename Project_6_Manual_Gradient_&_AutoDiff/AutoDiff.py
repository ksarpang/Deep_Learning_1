# Automatic differentiating
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
housing = fetch_california_housing()
m,n = housing.data.shape


scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X,theta,name="predictions")
error = y_pred - y

mse = tf.reduce_mean(tf.square(error),name = "mse")
gradients = 2/m*tf.matmul(tf.transpose(X), error)
learning_rate = 0.01
training_op = tf.assign(theta, theta - learning_rate* gradients)
init = tf.global_variables_initializer()
gradients = tf.gradients(mse, [theta])[0]

with tf.Session() as sess:
    sess.run(init)
    # writer = tf.summary.FileWriter('logs', graph)
    print(mse.eval())
    sess.run(training_op)
    print(mse.eval())
    sess.run(training_op)
    graph = tf.get_default_graph()
    writer = tf.summary.FileWriter('logs', graph)
    writer.close()

# graph = tf.get_default_graph()
operations = graph.get_operations()
print(len(operations))
