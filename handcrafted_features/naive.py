from collections import Counter

import pandas as pd

from nltk.tokenize import word_tokenize
from textdistance import jaccard, hamming, cosine, lcsstr, \
    bag, levenshtein


def naive_features(question1, question2):
    """Returns a series of naive document similarity features, case insensitive"""
    q1_lower = question1.lower()
    q2_lower = question2.lower()

    q1_words = word_tokenize(q1_lower)
    q2_words = word_tokenize(q2_lower)
    q1_chars = list(q1_lower)
    q2_chars = list(q2_lower)

    num_common_words = sum((Counter(q1_words) & Counter(q2_words)).values())
    num_common_chars = sum((Counter(q1_chars) & Counter(q2_chars)).values())

    # intersection over union
    if len(q1_words) == 0 and len(q2_words) == 0:
        word_iou = 0.
    else:
        num_words = sum((Counter(q1_chars) | Counter(q2_chars)).values())
        word_iou = num_common_words / num_words

    return pd.Series({
        'naive_num_words_q1': len(q1_words),
        'naive_num_words_q2': len(q2_words),
        'naive_num_chars_q1': len(q1_chars),
        'naive_num_chars_q2': len(q2_chars),
        'naive_num_shared_words_case_in': num_common_words,
        'naive_num_shared_chars_case_in': num_common_chars,
        'naive_word_iou_case_in': word_iou
    })


def text_distance_features(question1, question2):
    """Returns a series of string distance features, case insensitive."""
    q1_lower = question1.lower()
    q2_lower = question2.lower()
    return pd.Series({
        'text_sim_jaccard': jaccard(q1_lower, q2_lower),
        'text_sim_hamming': hamming(q1_lower, q2_lower),
        'text_sim_Levenshtein': levenshtein(q1_lower, q2_lower),
        'text_sim_cosine': cosine(q1_lower, q2_lower),
        'text_sim_lcsstr_len': len(lcsstr(q1_lower, q2_lower)),
        'test_sim_bag': bag(q1_lower, q2_lower)
    })
