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
    parser.add_argument('--prediction-filepath', default=str, required=True)
    parser.add_argument('--ground-truth-filepath', default=str, required=True)
    arg = parser.parse_args()

    y_pred = pd.read_csv(arg.prediction_filepath, usecols=['id', 'is_duplicate']).set_index('id')
    y_pred_bin = (y_pred.values > 0.5).astype(np.int)
    y_true = pd.read_csv(arg.ground_truth_filepath, usecols=['id', 'is_duplicate']).set_index('id')

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred_bin)
    f1score = f1_score(y_true=y_true, y_pred=y_pred_bin)
    logloss = log_loss(y_true=y_true, y_pred=y_pred)

    metrics = {'accuracy': accuracy, 'f1': f1score, 'logloss': logloss}
    json.dump(metrics, stdout)
