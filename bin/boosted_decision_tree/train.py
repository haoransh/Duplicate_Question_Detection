import argparse
import json
import pickle
from datetime import datetime
from os import path

import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-feature-filepath', type=str,
                        default='data/features_dev.csv')
    parser.add_argument('--dev-label-filepath', type=str,
                        default='data/qpairs_dev.csv')
    parser.add_argument('--param-filepath', type=str,
                        default='configs/xgb_params.json')
    parser.add_argument('--model-output-dir', type=str, default='output')
    parser.add_argument('--model-name-suffix', type=str, default='')
    arg = parser.parse_args()

    # read development set
    dev_X_df = pd.read_csv(arg.dev_feature_filepath).set_index('id')
    dev_y_df = pd.read_csv(arg.dev_label_filepath, usecols=['id', 'is_duplicate']).set_index('id')

    # join to make sure the order is correct
    joined = dev_y_df.merge(dev_X_df, left_index=True, right_index=True)
    dev_X_df = joined.drop(['is_duplicate'], axis=1)
    dev_y_df = joined[['is_duplicate']]

    # split dev set into a training set and a CV set
    train_X_df, cv_X_df, train_y_df, cv_y_df = \
        train_test_split(dev_X_df, dev_y_df, test_size=0.1, random_state=10701)

    # sample from training set to measure training loss
    train_df_sample = train_X_df.merge(train_y_df,
                                       left_index=True,
                                       right_index=True).sample(frac=0.1)
    train_X_df_sample = train_df_sample.drop('is_duplicate', axis=1)
    train_y_df_sample = train_df_sample[['is_duplicate']]

    # prepare data for xgboost
    dtrain = xgb.DMatrix(train_X_df, label=train_y_df)
    dtrain_sample = xgb.DMatrix(train_X_df_sample, label=train_y_df_sample)
    dcv = xgb.DMatrix(cv_X_df, label=cv_y_df)

    # load xgboost parameters
    with open(arg.param_filepath, 'r') as f:
        params = json.load(f)

    bst_tree = xgb.train(params=params,
                         dtrain=dtrain,
                         evals=[(dtrain_sample, "train"), (dcv, 'cross-validation')],
                         num_boost_round=1000,
                         verbose_eval=10,
                         early_stopping_rounds=50)

    # serialize boosted trees
    timestamp = datetime.now().strftime('%m-%d-%H%M%S')
    model_filepath = path.join(arg.model_output_dir, 'xgb_{}_{}.pickle'
                               .format(arg.model_name_suffix, timestamp))
    with open(model_filepath, 'wb') as f:
        pickle.dump(bst_tree, f)
