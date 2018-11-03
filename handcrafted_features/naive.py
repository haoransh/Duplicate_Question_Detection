import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize


def naive_features(question1, question2):
    """Returns a series of naive document similarity features, case insensitive"""
    q1_lower = question1.lower()
    q2_lower = question2.lower()

    q1_words = word_tokenize(q1_lower)
    q2_words = word_tokenize(q2_lower)
    q1_chars = list(q1_lower)
    q2_chars = list(q2_lower)

    common_words = [w for w in q1_words if w in q2_words]
    common_chars = [c for c in q1_chars if c in q2_chars]

    # intersection over union
    if len(q1_words) == 0 and len(q2_words) == 0:
        word_iou = 0.
    else:
        word_iou = len(common_words) / float(len(q1_words) + len(q2_words))

    return pd.Series({
        'naive_num_words_q1': len(q1_words),
        'naive_num_words_q2': len(q2_words),
        'naive_num_chars_q1': len(q1_chars),
        'naive_num_chars_q2': len(q2_chars),
        'naive_num_shared_words_case_in': len(common_words),
        'naive_num_shared_chars_case_in': len(common_chars),
        'naive_word_iou_case_in': word_iou
    })
