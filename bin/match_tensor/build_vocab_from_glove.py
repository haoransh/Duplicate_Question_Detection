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
    predefined = pd.DataFrame([{'word': '<PAD>'}, {'word': '<OOV>'}])

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
    embed_mat = glove_df[list(range(1, 51))].values

    # add word vector for pad and oov
    pad_oov = np.zeros((2, 50))
    embed_mat = np.concatenate([pad_oov, embed_mat], axis=0)
    print(embed_mat.shape)
    np.save("{}.embed_mat".format(arg.output_filepath), embed_mat)
