import argparse
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
    parser.add_argument('--model-output-dir', type=str, default='output')
    parser.add_argument('--model-suffix', type=str, default='')
    arg = parser.parse_args()

    # read development set
    dev_X_df = pd.read_csv(arg.dev_feature_filepath).set_index('id')
    dev_y_df = pd.read_csv(arg.dev_label_filepath, usecols=['id', 'is_duplicate']).set_index('id')
    train_X_df, cv_X_df, train_y_df, cv_y_df = \
        train_test_split(dev_X_df, dev_y_df, test_size=0.1, random_state=10701)

    dtrain = xgb.DMatrix(train_X_df, label=train_y_df)
    dcv = xgb.DMatrix(cv_X_df, label=cv_y_df)

    # TODO: put params in a JSON file
    params = {'objective': 'binary:logistic',
              'eval_metric': ['logloss'],
              'eta': 0.02,
              'max_depth': 8,
              "subsample": 0.7,
              "min_child_weight": 1,
              "colsample_bytree": 0.4,
              "silent": 1,
              "seed": 10701,
              'tree_method': 'exact'
              }

    bst_tree = xgb.train(params=params,
                         dtrain=dtrain,
                         evals=[(dcv, 'cross-validation')],
                         num_boost_round=2000,
                         early_stopping_rounds=50,
                         verbose_eval=10)

    # serialize boosted trees
    timestamp = datetime.now().strftime('%m-%d-%H%M%S')
    model_filepath = path.join(arg.model_output_dir, 'xgb_{}_{}.pickle'.format(arg.model_suffix, timestamp))
    with open(model_filepath, 'wb') as f:
        pickle.dump(bst_tree, f)
