"""
Split the development set into 87.5% training set (i.e. 70% original dataset),
12.5% cross-validation set (i.e. 10% original dataset).
"""

import argparse
from os import path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 10701

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-filepath', type=str,
                        default='data/questions.csv')
    parser.add_argument('--output-dir', type=str, default='data')
    arg = parser.parse_args()

    question_pair_df = pd.read_csv(arg.dev_filepath)

    dev_df, test_df = train_test_split(question_pair_df, test_size=0.125,
                                       random_state=RANDOM_STATE)

    dev_df.sort_values(by='id')\
        .to_csv(path.join(arg.output_dir, 'qpairs_train.csv'), index=None)
    test_df.sort_values(by='id')\
        .to_csv(path.join(arg.output_dir, 'qpairs_cv.csv'), index=None)
