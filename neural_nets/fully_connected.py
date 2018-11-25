import tensorflow as tf


class FullyConnectedClassifier:
    def __init__(self, fixed_question_len, embedding_size, vocab_size):
        self.fixed_question_len = fixed_question_len
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

    def __call__(self, features, labels, mode):
        q1, q2 = features

        # embed words
        embedding_mat = tf.get_variable('word_embeddings', shape=[self.vocab_size, self.embedding_size])
        q1_embedded_words = tf.nn.embedding_lookup(embedding_mat, q1)
        q2_embedded_words = tf.nn.embedding_lookup(embedding_mat, q2)

        # embed question
        q1_embedding = tf.reduce_mean(q1_embedded_words, axis=1, name='q1_embedding')
        q2_embedding = tf.reduce_mean(q2_embedded_words, axis=1, name='q2_embedding')

        # similarity
        with tf.name_scope('similarity'):
            similarity = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(q1_embedding, q2_embedding), axis=1))

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=similarity)

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
