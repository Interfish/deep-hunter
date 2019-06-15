import prepare_data
import numpy as np
import tensorflow as tf
# from tensorflow.python import debug as tf_debug

import sqlite3
import datetime
import os
import sys

class BiDirectionalGRU:
    def load_hparams(self,
                 max_char_num=128,
                 fw_num_units=20,
                 bw_num_units=20,
                 loss_weight_fraction=8):
        self.max_char_num = max_char_num
        self.fw_num_units = fw_num_units
        self.bw_num_units = bw_num_units
        self.loss_weight_fraction = loss_weight_fraction
        return self

    def build_model(self):
        print('[info] Constructing Computation Graph ...', flush=True)
        X = tf.placeholder(tf.int32, shape=(None, None), name='X')
        batch_size = tf.cast(tf.shape(X)[0], tf.float32)
        X_one_hot = tf.one_hot(X, depth=self.max_char_num, axis=-1, name='X_one_hot')
        y = tf.placeholder(tf.int32, shape=(None, None), name='y')
        # effective sequence length vector (without padding)
        seq_length = tf.placeholder(tf.int32, shape=(None), name='seq_length')
        max_len = tf.argmax(seq_length, name="max_len")
        # max_len = tf.placeholder(tf.int32, name='max_len')
        mask = tf.sequence_mask(seq_length, maxlen=max_len, dtype=tf.float32, name='mask')
        fw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.fw_num_units)
        bw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.bw_num_units)
        (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            X_one_hot,
            sequence_length=seq_length,
            dtype=tf.float32
        )
        outputs_reshaped = tf.reshape(tf.concat([fw_outputs, bw_outputs], 2), (-1, self.fw_num_units + self.bw_num_units))
        w = tf.Variable(tf.random_normal(
            shape=(self.fw_num_units + self.bw_num_units, 2)), name='W')
        b = tf.Variable(tf.random_normal(shape=(1, 1)), name='b')
        y_output_reshaped = tf.matmul(outputs_reshaped, w) + b
        y_reshaped = tf.reshape(y, (-1,), name='y_reshaped')
        y_one_hot_reshaped = tf.one_hot(y_reshaped, depth=2, axis=-1, name='y_ont_hot_reshaped')
        mask_reshaped = tf.reshape(mask, (-1, ), name='mask_reshaped')
        loss_reshaped = tf.multiply(tf.reduce_sum(tf.multiply(-1. * tf.nn.log_softmax(y_output_reshaped), y_one_hot_reshaped), axis=1), mask_reshaped)
        loss_weight = tf.cast(y_reshaped, dtype=tf.float32) * (self.loss_weight_fraction - 1) + 1
        total_loss = tf.reduce_sum(loss_reshaped * loss_weight, name='total_loss')
        regularization_cost = 0.001 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tf.trainable_variables() ])
        loss = total_loss / batch_size + regularization_cost
        prediction = tf.argmax(tf.nn.softmax(y_output_reshaped, axis=-1), axis=-1, output_type=tf.int32, name='prediction')
        _, accuracy = tf.metrics.accuracy(labels=y_reshaped, predictions=prediction, weights=mask_reshaped)
        _, precision = tf.metrics.precision(labels=y_reshaped, predictions=prediction, weights=mask_reshaped)
        _, recall = tf.metrics.recall(labels=y_reshaped, predictions=prediction, weights=mask_reshaped)

        train_summary_loss = tf.summary.scalar('Train Loss', loss)
        test_summary_loss = tf.summary.scalar('Test Loss', loss)
        summary_acc = tf.summary.scalar('Accuracy', accuracy)
        summary_precision = tf.summary.scalar('Precision', precision)
        summary_recall = tf.summary.scalar('Recall', recall)

        train_merge = tf.summary.merge([train_summary_loss, summary_acc])
        test_merge = tf.summary.merge([test_summary_loss, summary_acc, summary_precision, summary_recall])

        train_op = tf.train.AdamOptimizer().minimize(loss)

        return {
            'X': X,
            'y': y,
            'seq_length': seq_length,
            'max_len': max_len,
            'loss': loss,
            'prediction': precision,
            'train_op': train_op,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'train_merge': train_merge,
            'test_merge': test_merge
        }