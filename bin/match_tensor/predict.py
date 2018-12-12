import argparse
from os import path

import pandas as pd

import tensorflow as tf
import numpy as np


from match_tensor.data_pipeline import QuestionPairsDatasetInputFn
from match_tensor.match_tensor import MatchTensorClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--word-embedding-size', type=int, default=50)
    parser.add_argument('--fixed-question-len', type=int, default=32)

    parser.add_argument('--input-filepath', type=str, required=True)
    parser.add_argument('--embed-mat-filepath', type=str, default=None)
    parser.add_argument('--vocab-filepath', type=str, required=True)
    parser.add_argument('--model-dir', type=str, default="")

    parser.add_argument('--output-filename', type=str, default='match_tensor_predictions.csv')
    arg = parser.parse_args()

    dataset_input_fn = QuestionPairsDatasetInputFn(
        batch_size=arg.batch_size,
        shuffle_buffer_size=-1,
        input_filepath=arg.input_filepath,
        fixed_question_len=arg.fixed_question_len,
        vocab_filepath=arg.vocab_filepath)

    estimator_fn = MatchTensorClassifier(
        embedding_size=arg.word_embedding_size,
        vocab_size=dataset_input_fn.gen.vocab_size,
        fixed_question_len=dataset_input_fn.gen.question_len,
        embedding_mat_path=arg.embed_mat_filepath)

    duplicated_classifier = tf.estimator.Estimator(
        model_fn=estimator_fn, model_dir=arg.model_dir)

    pred = duplicated_classifier.predict(input_fn=dataset_input_fn)
    prob = np.zeros(len(pred))
    activations = np.zeros((len(pred), len(pred[0]['last_layer'])))
    for i, r in enumerate(pred):
        prob[i] = r['probability']
        activations[i] = r['last_layer']

    feat_df = pd.DataFrame(
        data=activations,
        columns=['bi_gru_{}'.format(i) for i in range(0, activations.shape[1])])
    feat_df['id'] = pd.read_csv(
        arg.input_filepath, usecols=['id'])['id']
    feat_df['bi_gru_is_duplicate'] = prob

    feat_df.to_csv(path.join('features', arg.output_filename), index=None)
