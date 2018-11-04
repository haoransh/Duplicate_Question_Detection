"""
The script extract features using specified extractors.
"""

import argparse
from functools import reduce
from multiprocessing import cpu_count, Pool
from os import path

import pandas as pd
import numpy as np

from handcrafted_features.naive import naive_features, text_distance_features

# feature extractors, change this to extract different sets of features
extractors = [naive_features, text_distance_features]


def _extract_features_one_split(df_split):
    """Extracts features for one dataframe split."""
    feature_sets = [df_split.apply(lambda r: e(r['question1'], r['question2']), axis=1)
                    for e in extractors]
    return reduce(lambda s1, s2: s2.merge(s1, left_index=True, right_index=True), feature_sets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-filepath', type=str,
                        default='data/qpairs_dev.csv')
    parser.add_argument('--output-dir', type=str, default='data')
    parser.add_argument('--output-filename-suffix', type=str, default='features')
    arg = parser.parse_args()

    # empty string should be read as "" instead of na
    # both question1 and question2 should be read as string
    question_pairs_df = pd.read_csv(arg.input_filepath,
                                    keep_default_na=False,
                                    dtype={'question1': str, 'question2': str})
    question_pairs_df = question_pairs_df.set_index('id')

    # parallelized feature extraction
    splits = np.array_split(question_pairs_df, cpu_count())
    with Pool(cpu_count()) as pool:
        naive_features_df = pd.concat(pool.map(_extract_features_one_split, splits))

    # saves features in a csv
    input_filename = path.basename(arg.input_filepath).split('.')[0]
    output_filename = '{}_{}.csv'.format(input_filename, arg.output_filename_suffix)
    naive_features_df.to_csv(path.join(arg.output_dir, output_filename),
                             index_label='id')
