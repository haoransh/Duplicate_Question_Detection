import tensorflow as tf

from match_tensor.model_base import ModelBase


class LSTMClassifier(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def match_model(self, q1_embedded_words, q2_embedded_words):
        # embed question
        recurrent_layer = tf.keras.layers.Bidirectional(
            tf.contrib.keras.layers.GRU(128, return_sequences=False),
            merge_mode='concat')
        q1_embedding = recurrent_layer(q1_embedded_words)
        q2_embedding = recurrent_layer(q2_embedded_words)
        print(q1_embedding.shape)
        concat = tf.concat([q1_embedding, q2_embedding], axis=1)
        dense1 = tf.layers.dense(concat, 64)
        dense1 = tf.layers.batch_normalization(dense1)
        dense1 = tf.tanh(dense1)
        dense2 = tf.layers.dense(dense1, 1)
        # dense2 = tf.layers.batch_normalization(dense2)
        similarity = tf.squeeze(tf.sigmoid(dense2))

        return dense1, similarity
