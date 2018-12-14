import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf

from match_tensor.coattention import CoattentionClassifier
from match_tensor.data_pipeline import QuestionPairsDatasetInputFn
from match_tensor.bi_gru_siamese import BiGRUSiameseClassifier
from match_tensor.match_tensor import MatchTensorClassifier

models = {
    'match_tensor': MatchTensorClassifier,
    'bi_gru': BiGRUSiameseClassifier,
    'coattention': CoattentionClassifier
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--word-embedding-size', type=int, default=50)
    parser.add_argument('--shuffle-buffer-size', type=int, default=1000)
    parser.add_argument('--fixed-question-len', type=int, default=32)

    parser.add_argument('--train-filepath', type=str, required=True)
    parser.add_argument('--embed-mat-filepath', type=str, default=None)
    parser.add_argument('--vocab-filepath', type=str, required=True)
    parser.add_argument('--cv-filepath', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='output')
    parser.add_argument('--model-name', type=str, default="")
    arg = parser.parse_args()

    dataset_input_fn_train = QuestionPairsDatasetInputFn(
        batch_size=arg.batch_size,
        shuffle_buffer_size=arg.shuffle_buffer_size,
        input_filepath=arg.train_filepath,
        fixed_question_len=arg.fixed_question_len,
        vocab_filepath=arg.vocab_filepath)

    if arg.cv_filepath is not None:
        dataset_input_fn_cv = QuestionPairsDatasetInputFn(
            batch_size=arg.batch_size,
            input_filepath=arg.cv_filepath,
            fixed_question_len=arg.fixed_question_len,
            vocab_filepath=arg.vocab_filepath)

    estimator_fn = models[arg.model](
        embedding_size=arg.word_embedding_size,
        vocab_size=dataset_input_fn_train.gen.vocab_size,
        fixed_question_len=dataset_input_fn_train.gen.question_len,
        embedding_mat_path=arg.embed_mat_filepath)

    duplicated_classifier = tf.estimator.Estimator(
        model_fn=estimator_fn, model_dir="{}/{}{}".format(
            arg.checkpoint_dir,
            arg.model_name,
            datetime.now().strftime('_%m-%d-%H%M%S')),
            config=tf.estimator.RunConfig(keep_checkpoint_max=0))

    # TODO implement early stop
    while True:  # train forever
        duplicated_classifier.train(input_fn=dataset_input_fn_train)
        if arg.cv_filepath is not None:
            metrics = duplicated_classifier.evaluate(input_fn=dataset_input_fn_cv)
            tf.logging.info(metrics)
