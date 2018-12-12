import numpy as np
import tensorflow as tf


class ModelBase:
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
        with tf.variable_scope('token_embeddings'):
            vocab = tf.get_variable(
                'vocab_embeddings',
                shape=[self.vocab_size, self.embedding_size],
                initializer=tf.constant_initializer(self.embed_mat),
                trainable=False)
            pad = tf.zeros((1, self.embedding_size))
            oov = tf.get_variable('oov', shape=[2, self.embedding_size], trainable=True)
            embedding_mat = tf.concat([pad, oov, vocab], axis=0)

        q1_embedded_words = tf.nn.embedding_lookup(embedding_mat, q1)
        q2_embedded_words = tf.nn.embedding_lookup(embedding_mat, q2)

        dense1, similarity = self.match_model(
            q1_embedded_words, q2_embedded_words)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions={
                'last_layer': dense1,
                'probability': similarity
            })

        loss = tf.losses.log_loss(labels=labels, predictions=similarity)

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

    def match_model(self, q1_embedded_words, q2_embedded_words):
        raise NotImplementedError
