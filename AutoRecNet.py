import tensorflow as tf
import numpy as np

# layer params
layers = [1024, 512, 128]
layers = {
    'w1': tf.get_variable("w1", [9000, 1024], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer),
    'w2': tf.get_variable("w2", [1024, 128], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer),
    'b1': tf.get_variable("b1", [1024], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer),
    'b2': tf.get_variable("b2", [128], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer),
    'b3': tf.get_variable("b3", [128], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer),
    'b4': tf.get_variable("b4", [1024], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer)
}

class AutoRecNet(object):
    """docstring for AutoRecNet."""
    def __init__(self, layer_params, num_movies, **args):
        self.layer_params = layer_params
        self.num_movies = num_movies

        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, shape=[None, 9000], name='input_x')
            self.y = tf.placeholder(tf.float32, shape=[None, 9000], name='input_y')
            self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[1], name='dropout_keep_prob')

        with tf.name_scope("Encoder"):
            e1 = tf.nn.selu(tf.nn.xw_plus_b(self.x, layer_params['w1'], layer_params['b1']))
            e2 = tf.nn.selu(tf.nn.xw_plus_b(e1, layer_params['w2'], layer_params['b2']))

        with tf.name_scope("Coding"):
            z = tf.nn.dropout(e2, keep_prob=self.dropout_keep_prob)

        with tf.name_scope("Decoder"):
            w2_t = tf.transpose(tf.get_variable("w3"))
            w1_t = tf.transpose(tf.get_variable("w2"))

            d1 = tf.nn.selu(tf.nn.xw_plus_b(z, w2_t, layer_params['b3']))
            d2 = tf.nn.xw_plus_b(d1, w1_t, layer_params['b4'])

        with tf.name_scope("Output"):
            self.yhat = d2

        with tf.name_scope("Loss"):
            self.loss = tf.square(tf.mean_squared_error(self.y, self.yhat))
