import re

import pandas as pd
from nltk import word_tokenize
import numpy as np
import tensorflow as tf


class QuestionPairsGenerator:
    def __init__(self, input_filepath, vocab_filepath, fixed_question_len):
        self.question_len = fixed_question_len
        self._qpairs_df = \
            pd.read_csv(input_filepath,
                        keep_default_na=False,
                        memory_map=True,
                        dtype={'question1': str, 'question2': str}) \
                .set_index('id')
        vocab_df = pd.read_csv(vocab_filepath).set_index('word')
        # word => index
        self.inverted_vocab = vocab_df['index']
        self._vocab = set(vocab_df.index.values)

    def __call__(self):
        for _, r in self._qpairs_df.iterrows():
            x = (self._process_one_question(r['question1']), self._process_one_question(r['question2']))
            y = r['is_duplicate']
            yield (x, y)

    def _process_one_question(self, text):
        encoded = self._encode(self._tokenize(text))[:self.question_len]
        return np.pad(encoded, (0, self.question_len - len(encoded)), mode='constant')

    def _tokenize(self, text):
        """Converts to lower case and removes math formulas, and then tokenize."""
        no_formula = re.sub(r'\[math\]((?!\[\/math\]).)*\[\/math\]', '', text.lower())
        return word_tokenize(no_formula)

    def _encode(self, tokens):
        return [self.inverted_vocab[t]
                if t in self._vocab else self.inverted_vocab['<OOV>']
                for t in tokens]

    @property
    def vocab_size(self):
        return len(self._vocab)


class QuestionPairsDatasetInputFunc:
    def __init__(self, batch_size, shuffle_buffer_size, *args, **kwargs):
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size

        # instantiate a generator
        self.gen = QuestionPairsGenerator(*args, **kwargs)

        self._ds = tf.data.Dataset.from_generator(
            self.gen, ((tf.int32, tf.int32), tf.int32), (
                (tf.TensorShape([self.gen.question_len]),
                 tf.TensorShape([self.gen.question_len])),
                 tf.TensorShape([])))

    def __call__(self):
        iter = self._ds.shuffle(self.shuffle_buffer_size)\
            .repeat()\
            .batch(self.batch_size)\
            .make_one_shot_iterator()
        return iter.get_next()