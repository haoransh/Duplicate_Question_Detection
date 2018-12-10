import argparse
import pandas as pd
from datetime import datetime
import numpy as np

import tensorflow as tf

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

    pred = list(duplicated_classifier.predict(input_fn=dataset_input_fn))
    ids = pd.read_csv(arg.input_filepath, usecols=['id'])['id'].tolist()

    pred_df = pd.DataFrame.from_dict({"is_duplicate": pred, "id": ids})
    pred_df.to_csv('data/match_tensor_prediction.csv', index=None)
