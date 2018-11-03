import pandas as pd

from nltk.tokenize import word_tokenize


def naive_features(question1, question2):
    """Returns a series of naive document similarity features."""
    q1_tokens = word_tokenize(question1)
    q2_tokens = word_tokenize(question2)
    q1_chars = list(question1)
    q2_chars = list(question2)

    common_words_case_sensitive = set(q1_tokens) & set(q2_tokens)
    common_chars_case_sensitive = set(q1_chars) & set(q2_chars)

    return pd.Series({
        'naive_num_words_q1': len(q1_tokens),
        'naive_num_words_q2': len(q2_tokens),
        'naive_num_chars_q1': len(q1_chars),
        'naive_num_chars_q2': len(q2_chars),
        'naive_num_shared_words_case_se': len(common_words_case_sensitive),
        'naive_num_shared_chars_case_se': len(common_chars_case_sensitive)
    })
