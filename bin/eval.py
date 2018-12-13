"""
The script evaluates predictions.
"""

import argparse
import json
from sys import stdout

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default=str, required=True)
    parser.add_argument('--label', default=str, required=True)
    arg = parser.parse_args()

    y_pred = pd.read_csv(arg.pred).set_index('id')['is_duplicate']
    # print(y_pred.columns)
    # exit()
    # [['bert_is_duplicate']]
    y_pred_bin = (y_pred.values > 0.5).astype(np.int)
    y_true = pd.read_csv(arg.label, usecols=['id', 'is_duplicate']).set_index('id')

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred_bin)
    f1score = f1_score(y_true=y_true, y_pred=y_pred_bin)
    logloss = log_loss(y_true=y_true, y_pred=y_pred)

    metrics = {'accuracy': accuracy, 'f1': f1score, 'logloss': logloss}
    json.dump(metrics, stdout)
