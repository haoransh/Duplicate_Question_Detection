import argparse
import pickle
from os import path

import pandas as pd

import xgboost as xgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-filepath', type=str, required=True)
    parser.add_argument('--input-feature-filepath', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='data')
    parser.add_argument('--output-filename', type=str, default='predictions.csv')
    arg = parser.parse_args()

    test_X_df = pd.read_csv(arg.input_feature_filepath).set_index('id')
    dtest = xgb.DMatrix(test_X_df)

    # load serialized trees
    with open(arg.model_filepath, 'rb') as f:
        bst_tree = pickle.load(f)

    preds = bst_tree.predict(dtest)

    # save predictions
    pred_df = pd.DataFrame.from_dict({'id': test_X_df.index.values, 'is_duplicate': preds}).set_index('id')
    pred_df.to_csv(path.join(arg.output_dir, arg.output_filename), index_label='id')
