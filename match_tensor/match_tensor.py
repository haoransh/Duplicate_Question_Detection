import tensorflow as tf
from match_tensor.model_base import ModelBase


class MatchTensorClassifier(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def match_model(self, q1_embedded_words, q2_embedded_words):
        # embed question
        recurrent_layer = tf.keras.layers.Bidirectional(
            tf.contrib.keras.layers.GRU(128, return_sequences=True),
            merge_mode='concat')
        q1_embedding = recurrent_layer(q1_embedded_words)
        q2_embedding = recurrent_layer(q2_embedded_words)

        # similarity
        with tf.name_scope('match_tensor'):
            q1 = tf.transpose(q1_embedding, perm=[0, 2, 1])
            q2 = tf.transpose(q2_embedding, perm=[0, 2, 1])
            q1 = tf.expand_dims(q1, axis=3)
            print(q1.shape)
            q2 = tf.expand_dims(q2, axis=2)
            print(q2.shape)
            match_tensor = tf.matmul(q1, q2)
            match_tensor = tf.transpose(match_tensor, perm=[0, 2, 3, 1])
            tf.summary.image(
                'match_tensor', tf.reduce_mean(match_tensor, axis=3, keep_dims=True))
            print(match_tensor.shape)

        # cnn layers
        conv1 = tf.layers.conv2d(match_tensor, filters=128, kernel_size=(3, 3))
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.layers.max_pooling2d(
            conv1, pool_size=(2, 2), strides=(2, 2))
        conv1 = tf.layers.batch_normalization(conv1)

        conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=(3, 3))
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.max_pooling2d(
            conv2, pool_size=(2, 2), strides=(2, 2))
        conv2 = tf.layers.batch_normalization(conv2)

        conv3 = tf.layers.conv2d(conv2, filters=128, kernel_size=(3, 3))
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.reduce_max(conv3, axis=(1, 2))

        dense1 = tf.layers.dense(conv3, 64)
        dense1 = tf.layers.batch_normalization(dense1)
        dense1 = tf.tanh(dense1)
        dense2 = tf.layers.dense(dense1, 1)
        similarity = tf.squeeze(tf.sigmoid(dense2))

        return dense1, similarity
