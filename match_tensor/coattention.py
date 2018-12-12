import tensorflow as tf

from match_tensor.model_base import ModelBase


class CoattentionClassifier(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def match_model(self, q1_embedded_words, q2_embedded_words):
        # embed question
        recurrent_layer = tf.keras.layers.Bidirectional(
            tf.contrib.keras.layers.GRU(128, return_sequences=True),
            merge_mode='concat')
        q1_embedding = recurrent_layer(q1_embedded_words)
        q2_embedding = recurrent_layer(q2_embedded_words)
        print(q1_embedding.shape)

        with tf.name_scope('affinity'):
            affinity_mat = tf.matmul(q1_embedding, tf.transpose(q2_embedding, perm=[0, 2, 1]))

        # scale by row
        with tf.name_scope('norm_by_row'):
            q1_attention = (affinity_mat - tf.reduce_min(affinity_mat, axis=2)) \
                           / (tf.reduce_max(affinity_mat, axis=2)
                              - tf.reduce_min(affinity_mat, axis=2)
                              + tf.keras.backend.epsilon())

        with tf.name_scope('q1_attended'):
            q1_attended = tf.matmul(q1_attention, q1_embedding)

        # scale by column
        with tf.name_scope('norm_by_col'):
            q2_attention = (affinity_mat - tf.reduce_min(affinity_mat, axis=1)
                            / (tf.reduce_max(affinity_mat, axis=1)
                               - tf.reduce_min(affinity_mat, axis=1)
                               + tf.keras.backend.epsilon()))

        with tf.name_scope('q2_attended'):
            q2_attended = tf.matmul(tf.transpose(q2_attention, perm=[0, 2, 1]), q2_embedding)

        concat = tf.concat([q1_attended, q2_attended], axis=1)
        
        recurrent2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, return_sequences=False),
            merge_mode='concat')(concat)

        dense1 = tf.layers.dense(recurrent2, 64)
        dense1 = tf.layers.batch_normalization(dense1)
        dense1 = tf.tanh(dense1)
        dense2 = tf.layers.dense(dense1, 1)
        similarity = tf.squeeze(tf.sigmoid(dense2))

        return dense1, similarity
