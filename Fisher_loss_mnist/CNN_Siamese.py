# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np


# I googled: tensorflow cnn example
# https://www.datacamp.com/community/tutorials/cnn-tensorflow-python

class CNN_Siamese:

    # Create model
    def __init__(self, loss_type, feature_space_dimension, margin_in_loss=0.25):
        # self.x1 = tf.placeholder(tf.float32, [None, 49152])
        # self.x1Image = tf.reshape(self.x1, [-1, 128, 128, 3])
        # self.x2 = tf.placeholder(tf.float32, [None, 49152])
        # self.x2Image = tf.reshape(self.x2, [-1, 128, 128, 3])
        # self.x3 = tf.placeholder(tf.float32, [None, 49152])
        # self.x3Image = tf.reshape(self.x3, [-1, 128, 128, 3])

        self.x1 = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.x1Image = self.x1
        self.x2 = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.x2Image = self.x2
        self.x3 = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.x3Image = self.x3

        self.margin_in_loss = margin_in_loss

        # self.loss_type = tf.placeholder(tf.float32, [1, 1])

        # self.weights = {
        #     'wc1': tf.get_variable('W0', shape=(3,3,3,32), initializer=tf.contrib.layers.xavier_initializer()),
        #     'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
        #     'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
        #     'wd1': tf.get_variable('W3', shape=(4*4*128,1024), initializer=tf.contrib.layers.xavier_initializer()),
        #     }

        self.weights = {
            'wc1': tf.get_variable('W0', shape=(3, 3, 3, 32), initializer=tf.glorot_uniform_initializer()),
            'wc2': tf.get_variable('W1', shape=(3, 3, 32, 64), initializer=tf.glorot_uniform_initializer()),
            'wc3': tf.get_variable('W2', shape=(3, 3, 64, 128), initializer=tf.glorot_uniform_initializer()),
            'wd1': tf.get_variable('W3', shape=(16 * 16 * 128, 500), initializer=tf.glorot_uniform_initializer()),
            'out': tf.get_variable('W6', shape=(500, feature_space_dimension), initializer=tf.glorot_uniform_initializer()),
        }
        self.biases = {
            'bc1': tf.get_variable('B0', shape=(32), initializer=tf.glorot_uniform_initializer()),
            'bc2': tf.get_variable('B1', shape=(64), initializer=tf.glorot_uniform_initializer()),
            'bc3': tf.get_variable('B2', shape=(128), initializer=tf.glorot_uniform_initializer()),
            'bd1': tf.get_variable('B3', shape=(500), initializer=tf.glorot_uniform_initializer()),
            'out': tf.get_variable('B4', shape=(feature_space_dimension), initializer=tf.glorot_uniform_initializer()),
        }

        self.loss_type = loss_type
        # Create loss
        if self.loss_type == "triplet":
            with tf.variable_scope("siamese") as scope:
                self.o1 = self.conv_net(self.x1Image, self.weights, self.biases)
                self.o2 = self.conv_net(self.x2Image, self.weights, self.biases)
                self.o3 = self.conv_net(self.x3Image, self.weights, self.biases)
            self.loss = self.loss_with_spring()
        elif self.loss_type == "FDA":
            with tf.variable_scope("siamese") as scope:
                self.o1 = self.conv_net_FDA(self.x1Image, self.weights, self.biases, o_index=1)
                self.o2 = self.conv_net_FDA(self.x2Image, self.weights, self.biases, o_index=2)
                self.o3 = self.conv_net_FDA(self.x3Image, self.weights, self.biases, o_index=3)
            self.loss = self.loss_FDA()
        # self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    def conv2d(self, x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv_net(self, x, weights, biases):
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        conv1 = tf.nn.relu(conv1)
        conv1 = self.maxpool2d(conv1, k=2)
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = tf.nn.relu(conv2)
        conv2 = self.maxpool2d(conv2, k=2)
        conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'])
        conv3 = tf.nn.relu(conv3)
        conv3 = self.maxpool2d(conv3, k=2)
        fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        # fc1 = tf.nn.relu(fc1)
        # fc1 = tf.nn.sigmoid(fc1)
        # finally we multiply the fully connected layer with the weights and add a bias term.
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    def conv_net_FDA(self, x, weights, biases, o_index):
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        conv1 = tf.nn.relu(conv1)
        conv1 = self.maxpool2d(conv1, k=2)
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = tf.nn.relu(conv2)
        conv2 = self.maxpool2d(conv2, k=2)
        conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'])
        conv3 = tf.nn.relu(conv3)
        conv3 = self.maxpool2d(conv3, k=2)
        fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        if o_index == 1:
            self.o1_output_oneToLastLayer = fc1
        elif o_index == 2:
            self.o2_output_oneToLastLayer = fc1
        elif o_index == 3:
            self.o3_output_oneToLastLayer = fc1
        # fc1 = tf.nn.relu(fc1)
        # fc1 = tf.nn.sigmoid(fc1)
        # finally we multiply the fully connected layer with the weights and add a bias term.
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        self.weights_lastLayer = weights['out']
        return out

    # def loss_with_spring(self):
    #     # margin = 0.2
    #     margin = 0.25
    #     eucdP = tf.pow(tf.subtract(self.o1, self.o2), 2)
    #     pos = tf.reduce_sum(eucdP, 1)
    #
    #     eucdN = tf.pow(tf.subtract(self.o1, self.o3), 2)
    #     neg = tf.reduce_sum(eucdN, 1)
    #
    #     C = tf.constant(margin, name="C")
    #     basic_loss = tf.subtract(pos, neg, name="loss")
    #     basic_loss = tf.add(basic_loss, C, name="loss")
    #     loss = tf.maximum(basic_loss, 0.0)
    #
    #     return loss

    def loss_with_spring(self):
        d_pos = tf.reduce_sum(tf.square(self.o1 - self.o2), 1)
        d_neg = tf.reduce_sum(tf.square(self.o1 - self.o3), 1)

        loss = tf.maximum(0., self.margin_in_loss + d_pos - d_neg)
        loss = tf.reduce_mean(loss)

        return loss

    def loss_FDA(self):
        # margin = 0.2
        # margin = 0.25
        margin = 10

        # calculation of within scatter:
        temp1 = self.o1_output_oneToLastLayer - self.o2_output_oneToLastLayer
        temp1 = tf.transpose(temp1)  # --> becomes: rows are features and columns are samples of batch
        S_within = tf.linalg.matmul(a=temp1, b=tf.transpose(temp1))

        # calculation of between scatter:
        temp2 = self.o1_output_oneToLastLayer - self.o3_output_oneToLastLayer
        temp2 = tf.transpose(temp2)  # --> becomes: rows are features and columns are samples of batch
        S_between = tf.linalg.matmul(a=temp2, b=tf.transpose(temp2))

        # calculation of variance of projection considering within scatter:
        temp3 = tf.linalg.matmul(a=tf.transpose(self.weights_lastLayer), b=S_within)
        temp3 = tf.linalg.matmul(a=temp3, b=self.weights_lastLayer)
        within_scatter_term = tf.linalg.trace(temp3)

        # calculation of variance of projection considering between scatter:
        temp4 = tf.linalg.matmul(a=tf.transpose(self.weights_lastLayer), b=S_between)
        temp4 = tf.linalg.matmul(a=temp4, b=self.weights_lastLayer)
        between_scatter_term = tf.linalg.trace(temp4)

        # calculation of loss:
        loss = tf.math.maximum(0., margin + within_scatter_term - between_scatter_term)
        # loss = within_scatter_term - between_scatter_term + margin
        # loss = (within_scatter_term) / (between_scatter_term)
        # loss = (within_scatter_term + margin) / (between_scatter_term)
        # loss = (within_scatter_term) / (between_scatter_term - margin)
        loss = tf.reduce_mean(loss)

        return loss

    # def network(self, x):
    #     weights = []
    #     fc1 = self.fc_layer(x, 1024, "fc1")
    #     ac1 = tf.nn.relu(fc1)
    #     fc2 = self.fc_layer(ac1, 1024, "fc2")
    #     ac2 = tf.nn.relu(fc2)
    #     fc3 = self.fc_layer(ac2, 3, "fc3")
    #     return fc3

    # def fc_layer(self, bottom, n_weight, name):
    #     n_prev_weight = bottom.get_shape()[1]
    #     initer = tf.truncated_normal_initializer(stddev=0.01)
    #     W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
    #     b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
    #     fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
    #     return fc
