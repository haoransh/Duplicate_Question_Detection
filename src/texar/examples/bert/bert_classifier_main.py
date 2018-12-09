# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example of building a sentence classifier based on pre-trained BERT
model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import importlib
import tensorflow as tf
import texar as tx
import pandas as pd
import logging

from utils import data_utils, model_utils, tokenization

# pylint: disable=invalid-name, too-many-locals, too-many-statements

flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "task", "quora",
    "The task to run experiment on. One of "
    "{'cola', 'mnli', 'mrpc', 'xnli', 'sst', 'quora'}.")
flags.DEFINE_string(
    "config_bert_pretrain", 'uncased_L-12_H-768_A-12',
    "The architecture of pre-trained BERT model to use.")
flags.DEFINE_string(
    "config_format_bert", "json",
    "The configuration format. Set to 'json' if the BERT config file is in "
    "the same format of the official BERT config file. Set to 'texar' if the "
    "BERT config file is in Texar format.")
flags.DEFINE_string(
    "config_downstream", "config_classifier",
    "Configuration of the downstream part of the model and optmization.")
flags.DEFINE_string(
    "config_data", "config_data_quora",
    "The dataset config.")
flags.DEFINE_string(
    "checkpoint", None,
    "Path to a model checkpoint (including bert modules) to restore from.")
flags.DEFINE_string(
    "output_dir", "output/",
    "The output directory where the model checkpoints will be written.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_test", False, "Whether to run test on the test set.")

config_data = importlib.import_module(FLAGS.config_data)
config_downstream = importlib.import_module(FLAGS.config_downstream)

def get_logger(log_path):
    """Returns a logger.

    Args:
        log_path (str): Path to the log file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
    logger.addHandler(fh)
    return logger

def main(_):
    """
    Builds the model and runs.
    """
    tx.utils.maybe_create_dir(FLAGS.output_dir)
    logging_file = os.path.join(FLAGS.output_dir, 'logging.txt')
    logger = get_logger(logging_file)
    print('logging file is saved in: %s', logging_file)

    bert_pretrain_dir = 'bert_pretrained_models/%s' % FLAGS.config_bert_pretrain

    # Loads BERT model configuration
    if FLAGS.config_format_bert == "json":
        bert_config = model_utils.transform_bert_to_texar_config(
            os.path.join(bert_pretrain_dir, 'bert_config.json'))
    elif FLAGS.config_format_bert == 'texar':
        bert_config = importlib.import_module(
            'bert_config_lib.config_model_%s' % FLAGS.config_bert_pretrain)
    else:
        raise ValueError('Unknown config_format_bert.')

    # Loads data
    processors = {
        "cola": data_utils.ColaProcessor,
        "mnli": data_utils.MnliProcessor,
        "mrpc": data_utils.MrpcProcessor,
        "xnli": data_utils.XnliProcessor,
        'sst': data_utils.SSTProcessor,
        'quora': data_utils.QuoraProcessor,
    }

    processor = processors[FLAGS.task.lower()]()

    num_classes = len(processor.get_labels())
    num_train_data = len(processor.get_train_examples(config_data.data_dir))

    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_pretrain_dir, 'vocab.txt'),
        do_lower_case=FLAGS.do_lower_case)

    logger.info('preparing dataset...')
    train_dataset = data_utils.get_dataset(
        processor, tokenizer, config_data.data_dir, config_data.max_seq_length,
        config_data.train_batch_size, mode='train', output_dir=FLAGS.output_dir)
    eval_dataset = data_utils.get_dataset(
        processor, tokenizer, config_data.data_dir, config_data.max_seq_length,
        config_data.eval_batch_size, mode='eval', output_dir=FLAGS.output_dir)
    test_dataset = data_utils.get_dataset(
        processor, tokenizer, config_data.data_dir, config_data.max_seq_length,
        config_data.test_batch_size, mode='test', output_dir=FLAGS.output_dir)

    iterator = tx.data.FeedableDataIterator({
        'train': train_dataset, 'eval': eval_dataset, 'test': test_dataset})
    batch = iterator.get_next()
    input_ids = batch["input_ids"]
    segment_ids = batch["segment_ids"]
    pair_ids = batch['pair_ids']
    batch_size = tf.shape(input_ids)[0]
    input_length = tf.reduce_sum(1 - tf.to_int32(tf.equal(input_ids, 0)),
                                 axis=1)

    logger.info('building the model...')
    # Builds BERT
    with tf.variable_scope('bert'):
        embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.vocab_size,
            hparams=bert_config.embed)
        word_embeds = embedder(input_ids)

        # Creates segment embeddings for each type of tokens.
        segment_embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.type_vocab_size,
            hparams=bert_config.segment_embed)
        segment_embeds = segment_embedder(segment_ids)

        input_embeds = word_embeds + segment_embeds

        # The BERT model (a TransformerEncoder)
        encoder = tx.modules.TransformerEncoder(hparams=bert_config.encoder)
        output = encoder(input_embeds, input_length)

        # Builds layers for downstream classification, which is also initialized
        # with BERT pre-trained checkpoint.
        with tf.variable_scope("pooler"):
            # Uses the projection of the 1st-step hidden vector of BERT output
            # as the representation of the sentence
            bert_sent_hidden = tf.squeeze(output[:, 0:1, :], axis=1)
            bert_sent_output = tf.layers.dense(
                bert_sent_hidden, config_downstream.hidden_dim,
                activation=tf.tanh)
            output = tf.layers.dropout(
                bert_sent_output, rate=0.1, training=tx.global_mode_train())

    # Adds the final classification layer
    logits = tf.layers.dense(
        output, num_classes,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    probs = tf.nn.softmax(logits, axis=-1)[:, 1]
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accu = tx.evals.accuracy(batch['label_ids'], preds)

    # Optimization

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=batch["label_ids"], logits=logits)
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, trainable=False)

    # Builds learning rate decay scheduler
    static_lr = config_downstream.lr['static_lr']
    num_train_steps = int(num_train_data / config_data.train_batch_size
                          * config_data.max_train_epoch)
    num_warmup_steps = int(num_train_steps * config_data.warmup_proportion)
    lr = model_utils.get_lr(global_step, num_train_steps, # lr is a Tensor
                            num_warmup_steps, static_lr)

    tf.summary.scalar('lr', lr)
    logger.info('train steps: %s' % (num_train_steps))
    train_op = tx.core.get_train_op(
        loss,
        global_step=global_step,
        learning_rate=lr,
        hparams=config_downstream.opt)

    merged = tf.summary.merge_all()

    # Train/eval/test routine

    def _run(sess, mode, writer=None, saver=None):
        fetches = {
            'accu': accu,
            'batch_size': batch_size,
            'step': global_step,
            'loss': loss,
        }

        eval_accu = 0

        if mode == 'train':
            fetches['train_op'] = train_op
            fetches['mgd'] = merged
            while True:
                try:
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, 'train'),
                        tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
                    }
                    rets = sess.run(fetches, feed_dict)
                    writer.add_summary(rets['mgd'], rets['step'])
                    if rets['step'] % 50 == 0:
                        logger.info(
                            'step:%d loss:%f' % (rets['step'], rets['loss']))
                    if rets['step'] == num_train_steps:
                        break
                    if rets['step'] % 500 == 0:
                        iterator.restart_dataset(sess, 'eval')
                        _dev_accu = _run(sess, mode='eval')
                        if _dev_accu > eval_accu:
                            logger.info('saving model...')
                            saver.save(sess, FLAGS.output_dir + '/model.ckpt')
                            eval_accu = _dev_accu
                            _run(sess, mode='test')
                        else:
                            exit()
                except tf.errors.OutOfRangeError:
                    break

        if mode == 'eval':
            cum_acc = 0.0
            nsamples = 0
            while True:
                try:
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, 'eval'),
                        tx.context.global_mode(): tf.estimator.ModeKeys.EVAL,
                    }
                    rets = sess.run(fetches, feed_dict)

                    cum_acc += rets['accu'] * rets['batch_size']
                    nsamples += rets['batch_size']
                except tf.errors.OutOfRangeError:
                    break
            dev_accu = cum_acc / nsamples
            logger.info('dev accu: {}'.format(cum_acc / nsamples))
            return dev_accu

        if mode == 'test':
            _all_probs = []
            _all_ids = []
            cum_acc = 0.0
            nsamples = 0
            fetches['prob'] = probs
            fetches['pair_ids'] = pair_ids
            while True:
                try:
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, 'test'),
                        tx.context.global_mode(): tf.estimator.ModeKeys.PREDICT,
                    }
                    rets = sess.run(fetches, feed_dict)
                    _all_probs.extend(rets['prob'].tolist())
                    _all_ids.extend(rets['pair_ids'].tolist())
                    cum_acc += rets['accu'] * rets['batch_size']
                    nsamples += rets['batch_size']
                except tf.errors.OutOfRangeError:
                    break

            output_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
            pred_df = pd.DataFrame.from_dict(
                {'id': [tf.compat.as_str_any(_id) for _id in _all_ids],
                 'is_duplicate': _all_probs}).set_index('id')
            pred_df.to_csv(output_file, index_label='id')

            test_accu = cum_acc / nsamples
            logger.info('test accu: {}'.format(cum_acc / nsamples))
            return test_accu

    logger.info('running...')

    with tf.Session() as sess:
        # Loads pretrained BERT model parameters
        init_checkpoint = os.path.join(bert_pretrain_dir, 'bert_model.ckpt')
        model_utils.init_bert_checkpoint(init_checkpoint)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        # Restores trained model if specified
        saver = tf.train.Saver()
        if FLAGS.checkpoint:
            saver.restore(sess, FLAGS.checkpoint)
        train_writer = tf.summary.FileWriter(FLAGS.output_dir + '/train',
                                             sess.graph)

        iterator.initialize_dataset(sess)

        if FLAGS.do_train:
            iterator.restart_dataset(sess, 'train')
            _run(sess, 'train', train_writer, saver)

        if FLAGS.do_test:
            iterator.restart_dataset(sess, 'test')
            _run(sess, 'test')

if __name__ == "__main__":
    tf.app.run()
