import numpy as np
import tensorflow as tf


class MatchTensorClassifier:
    def __init__(self, fixed_question_len, embedding_size, vocab_size, embedding_mat_path):
        self.fixed_question_len = fixed_question_len
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.embed_mat = np.load(embedding_mat_path)
        print(self.embed_mat.shape)
        print(self.vocab_size)

    def __call__(self, features, labels, mode):
        q1, q2 = features

        # embed words
        embedding_mat = tf.get_variable(
            'word_embeddings',
            shape=[self.vocab_size, self.embedding_size],
            initializer=tf.constant_initializer(self.embed_mat),
            trainable=False)

        q1_embedded_words = tf.nn.embedding_lookup(embedding_mat, q1)
        q2_embedded_words = tf.nn.embedding_lookup(embedding_mat, q2)

        # embed question
        recurrent_layer = tf.contrib.keras.layers.GRU(50, return_sequences=True)
        q1_embedding = recurrent_layer(q1_embedded_words)
        q2_embedding = recurrent_layer(q2_embedded_words)
        print(q1_embedding.shape)

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
            print(match_tensor.shape)

        # cnn layers
        conv1 = tf.layers.conv2d(match_tensor, filters=20, kernel_size=(3, 3))
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2))
        conv1 = tf.layers.batch_normalization(conv1)

        conv2 = tf.layers.conv2d(conv1, filters=20, kernel_size=(3, 3))
        conv2 = tf.nn.relu(conv2)
        print(conv2.shape)
        conv2 = tf.reduce_max(conv2, axis=(1, 2))
        print(conv2.shape)

        dense1 = tf.layers.dense(conv2, 64)
        dense1 = tf.layers.batch_normalization(dense1)
        dense1 = tf.tanh(dense1)
        dense2 = tf.layers.dense(dense1, 1)
        similarity = tf.squeeze(tf.sigmoid(dense2))

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=similarity)

        # loss = tf.losses.log_loss(labels=labels, predictions=similarity)
        # hinge loss
        hinged_loss = tf.multiply(tf.cast(labels, tf.float32),
                                  tf.minimum(similarity - 0.6, 0)) \
                      + tf.multiply(tf.cast((1 - labels), tf.float32),
                                    tf.maximum(similarity - 0.4, 0))
        loss = tf.losses.absolute_difference(hinged_loss, tf.zeros_like(hinged_loss))

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # threshold predictions
        bin_similarity = tf.cast(tf.round(similarity), tf.int32)
        metrics_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=bin_similarity),
            'mean_prediction': tf.metrics.mean(similarity),
            'var_prediction': tf.metrics.mean(tf.square(similarity - tf.reduce_mean(similarity))),
            'false_negative': tf.metrics.false_negatives(labels=labels, predictions=bin_similarity),
            'false_positive': tf.metrics.false_positives(labels=labels, predictions=bin_similarity)
        }

        return tf.estimator.EstimatorSpec(loss=loss, eval_metric_ops=metrics_ops, mode=mode)
