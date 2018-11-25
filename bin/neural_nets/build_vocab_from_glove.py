import argparse
import csv
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w2v-filepath', type=str, required=True)
    parser.add_argument('--output-filepath', type=str, required=True)
    arg = parser.parse_args()

    predefined = pd.DataFrame([{'word': '<PAD>'}, {'word': '<OOV>'}])

    glove_vocab = pd.read_table(
        arg.w2v_filepath,
        sep=" ",
        header=None,
        names=['word'],
        usecols=[0],
        quoting=csv.QUOTE_NONE)

    inverted_index = pd.concat([predefined, glove_vocab], ignore_index=True)
    print(inverted_index.head())

    # tab separated since tensorflow projections takes tsv
    inverted_index.to_csv(
        arg.output_filepath,
        sep='\t',
        index_label='index',
        header=['word'])
