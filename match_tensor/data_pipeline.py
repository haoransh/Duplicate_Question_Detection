import re
from datetime import datetime
from os import path

import pandas as pd
from nltk import word_tokenize
import numpy as np
import tensorflow as tf

from tqdm import tqdm


class QuestionPairsGenerator:
    def __init__(self, input_filepath, vocab_filepath, fixed_question_len):
        self.question_len = fixed_question_len

        # load question pairs
        self._qpairs_df = pd.read_csv(
            input_filepath,
            keep_default_na=False,
            memory_map=True,
            dtype={'question1': str, 'question2': str}) \
            .set_index('id')

        # load vocab file
        vocab_df = pd.read_csv(vocab_filepath, sep='\t', keep_default_na=False)

        # build vocab and inverted vocab index
        print("building vocab index...")
        self.inverted_vocab = {}
        for i, word in tqdm(vocab_df[['index', 'word']].values.tolist()):
            self.inverted_vocab[word] = i
        print(len(self.inverted_vocab.keys()))
        self._vocab = set(self.inverted_vocab.keys())
        print("vocab_size=%s" % self.vocab_size)

    def __call__(self):
        for _, r in self._qpairs_df.iterrows():
            x = (self._process_one_question(r['question1'], 1), self._process_one_question(r['question2'], 2))
            y = r['is_duplicate']
            yield (x, y)

    def _process_one_question(self, text, qid):
        encoded = self._encode(self._tokenize(text), qid)[:self.question_len]
        return np.pad(encoded, (0, self.question_len - len(encoded)), mode='constant')

    def _tokenize(self, text):
        """Converts to lower case and removes math formulas, and then tokenize."""
        no_formula = re.sub(r'\[math\]((?!\[\/math\]).)*\[\/math\]', '', text.lower())
        return word_tokenize(no_formula)

    def _encode(self, tokens, qid):
        encoded = []
        for t in tokens:
            if t in self._vocab:
                encoded.append(self.inverted_vocab[t])
            elif qid == 1:
                encoded.append(self.inverted_vocab['<OOV1>'])
            else:
                encoded.append(self.inverted_vocab['<OOV2>'])
        return encoded

    @property
    def vocab_size(self):
        return len(self._vocab)


class QuestionPairsDatasetInputFn:
    def __init__(self, batch_size, shuffle_buffer_size=-1, epochs=1, cache_filename=None, *args, **kwargs):
        self._batch_size = batch_size
        self._shuffle_buffer_size = shuffle_buffer_size
        self._epochs = epochs
        self._cache_filename = cache_filename \
            if cache_filename is not None \
            else path.join('tmp', datetime.now().strftime('%m%d%H%M%S'))

        # instantiate a generator
        self.gen = QuestionPairsGenerator(*args, **kwargs)

        # wrap the generator with tensorflow Dataset
        self._ds = tf.data.Dataset.from_generator(
            self.gen, ((tf.int32, tf.int32), tf.int32), (
                (tf.TensorShape([self.gen.question_len]),
                 tf.TensorShape([self.gen.question_len])),
                 tf.TensorShape([])))\
            .cache(filename=self._cache_filename)

        if self._shuffle_buffer_size > 0:
            self._ds = self._ds.shuffle(self._shuffle_buffer_size)

        self._ds = self._ds.repeat(self._epochs) \
            .batch(self._batch_size)

    def __call__(self):
        # make an iterator
        return self._ds.make_one_shot_iterator().get_next()
