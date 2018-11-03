import argparse
from os import path

import pandas as pd

from handcrafted_features.naive import naive_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-filepath', type=str,
                        default='data/qpairs_dev.csv')
    parser.add_argument('--output-dir', type=str, default='data')
    arg = parser.parse_args()

    # empty string should be read as ""
    # both question1 and question2 should be read as string
    question_pairs_df = pd.read_csv(arg.input_filepath,
                                    keep_default_na=False,
                                    dtype={'question1': str, 'question2': str})
    question_pairs_df = question_pairs_df.set_index('id')

    naive_features_df = question_pairs_df\
        .apply(lambda r: naive_features(r['question1'], r['question2']), axis=1)

    # saves features in a csv
    input_filename = path.basename(arg.input_filepath).split('.')[0]
    output_filename = '{}_naive_features.csv'.format(input_filename)
    naive_features_df.to_csv(path.join(arg.output_dir, output_filename),
                             index_label='id')
