from collections import Counter

import pandas as pd

from nltk.tokenize import word_tokenize
from textdistance import jaccard, hamming, cosine, lcsstr, \
    bag, levenshtein


def naive_features(question1, question2):
    """Returns a series of naive document similarity features, case insensitive"""
    q1_lower = question1.lower()
    q2_lower = question2.lower()

    q1_words = Counter(word_tokenize(q1_lower))
    q2_words = Counter(word_tokenize(q2_lower))
    q1_chars = Counter(list(q1_lower))
    q2_chars = Counter(list(q2_lower))

    num_common_words = sum((q1_words & q2_words).values())
    num_common_chars = sum((q1_chars & q2_chars).values())

    # word intersection over union
    if len(q1_words) == 0 and len(q2_words) == 0:
        word_iou = 0.
    else:
        num_total_words = sum((q1_words | q2_words).values())
        word_iou = num_common_words / num_total_words

    return pd.Series({
        'naive_num_words_q1': len(q1_words),
        'naive_num_words_q2': len(q2_words),
        'naive_num_chars_q1': len(q1_chars),
        'naive_num_chars_q2': len(q2_chars),
        'naive_num_shared_words': num_common_words,
        'naive_num_shared_chars': num_common_chars,
        'naive_word_iou': word_iou
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
