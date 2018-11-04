"""
The script splits the original question-pair dataset into a development set
and a test set.
"""

import argparse
from os import path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 10701

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-filepath', type=str,
                        default='data/questions.csv')
    parser.add_argument('--output-dir', type=str, default='data')
    arg = parser.parse_args()

    question_pair_df = pd.read_csv(arg.input_filepath)

    dev_df, test_df = train_test_split(question_pair_df, test_size=0.2,
                                       random_state=RANDOM_STATE)

    dev_df.sort_values(by='id')\
        .to_csv(path.join(arg.output_dir, 'qpairs_dev.csv'), index=None)
    test_df.sort_values(by='id')\
        .to_csv(path.join(arg.output_dir, 'qpairs_test.csv'), index=None)
