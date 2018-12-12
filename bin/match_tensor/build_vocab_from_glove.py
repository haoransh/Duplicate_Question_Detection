import argparse
import csv

import pandas as pd
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w2v-filepath', type=str, required=True)
    parser.add_argument('--output-filepath', type=str, required=True)
    arg = parser.parse_args()

    # build inverted vocab index
    predefined = pd.DataFrame([{'word': '<PAD>'}, {'word': '<OOV1>'}, {'word': '<OOV2>'}])

    glove_df = pd.read_table(
        arg.w2v_filepath,
        sep=" ",
        header=None,
        quoting=csv.QUOTE_NONE)\
        .drop_duplicates([0])

    glove_vocab = glove_df[[0]]
    glove_vocab.columns = ['word']
    inverted_index = pd.concat([predefined, glove_vocab], ignore_index=True)
    print(inverted_index.head())

    # tab separated since tensorflow projections takes tsv
    inverted_index.to_csv(
        arg.output_filepath,
        sep='\t',
        index_label='index',
        header=['word'])

    # build embedding matrix
    embed_mat = glove_df[list(range(1, glove_df.shape[1]))].values
    print(embed_mat.shape)
    np.save("{}.embed_mat".format(arg.output_filepath), embed_mat)
