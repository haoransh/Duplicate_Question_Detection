from datetime import datetime

import tensorflow as tf

from neural_nets.data_pipeline import QuestionPairsGenerator, \
    QuestionPairsDatasetInputFunc
from neural_nets.fully_connected import FullyConnectedClassifier

if __name__ == '__main__':
    # TODO add cmd argumenets

    dataset_input_func = QuestionPairsDatasetInputFunc(
        batch_size=32,
        shuffle_buffer_size=100,
        input_filepath='data/qpairs_dev_small.csv',
        fixed_question_len=10,
        vocab_filepath='data/glove6B_vocab_inverted_index.csv')

    estimator_func = FullyConnectedClassifier(
        embedding_size=50,
        vocab_size=dataset_input_func.gen.vocab_size,
        fixed_question_len=dataset_input_func.gen.question_len)

    duplicated_classifier = tf.estimator.Estimator(
        model_fn=estimator_func, model_dir="output/test_{}".format(datetime.now().strftime('%m-%d-%H%M%S')))

    duplicated_classifier.train(
        input_fn=dataset_input_func,
        steps=5)
