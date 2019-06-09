import prepare_data
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import sys
import os
import datetime
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class SVMHunter:
    def __init__(self, iteration):
        self.iteration = int(iteration)
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    def run(self):
        self.prepare_data()
        self.tf_idf()
        self.run_svm()

    def prepare_data(self):
        print('Preparing data ...')
        sys.stdout.flush()
        normal, unsure, leaked = prepare_data.load_data('data.sqlite3')
        normal_content = list(map(lambda x: x['content'],  normal))
        normal_file_name = list(map(lambda x: x['file_name'],  normal))

        unsure_content = list(map(lambda x: x['content'],  unsure))
        unsure_file_name = list(map(lambda x: x['file_name'],  unsure))

        leaked_content = list(map(lambda x: x['content'],  leaked))
        leaked_file_name = list(map(lambda x: x['file_name'],  leaked))

        self.content = normal_content + unsure_content + leaked_content
        self.file_name = normal_file_name + unsure_file_name + leaked_file_name
        self.y = np.array([-1 for i in range(0, len(normal_content))] +
                          [1 for i in range(0, len(leaked_content + unsure_content))])

    def tf_idf(self):
        print('TF-IDFING ...')
        sys.stdout.flush()
        vectorizer = TfidfVectorizer(
            min_df=0.0,
            analyzer='char',
            lowercase=True,
            ngram_range=(3, 5),
            norm='l2',
            use_idf=True,
            sublinear_tf=True
        )
        X_content = vectorizer.fit_transform(self.content)
        X_file_name = vectorizer.fit_transform(self.file_name)

        X = sp.hstack((X_file_name, X_content)).toarray()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, self.y, test_size=0.1)

    def mini_batch(self, iter_times, batchsize=128):
        for i in range(iter_times):
            start = (i * batchsize) % self.X_train.shape[0]
            end = ((i + 1) * batchsize) % self.X_train.shape[0]

            if start > end:
                yield i, np.append(self.X_train[start:], self.X_train[0:end], axis=0), np.append(self.y_train[start:], self.y_train[0:end], axis=0)

            else:
                yield i, self.X_train[start:end], self.y_train[start:end]

    def run_svm(self):
        print('Constructing Computation Graph ...')
        sys.stdout.flush()
        # computation graph
        X = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.float32)
        W = tf.Variable(tf.random_normal(shape=(self.X_train.shape[1], 1)))
        b = tf.Variable(tf.random_normal(shape=(1, 1)))
        y_hat = tf.add(b, tf.matmul(X, W))
        alpha = tf.constant(0.1)
        C = alpha * tf.reduce_sum(tf.square(W))
        loss = tf.reduce_mean(tf.maximum(
            0., 1. - tf.multiply(y, y_hat))) + C
        prediction = tf.squeeze(tf.sign(y_hat))
        transformed_pred = tf.maximum(0., prediction)
        trainsformed_y = tf.maximum(0., y)
        saver = tf.train.Saver()

        # for train set
        train_summary_loss = tf.summary.scalar('Loss', loss)
        _, acc_op = tf.metrics.accuracy(
            labels=trainsformed_y, predictions=transformed_pred)
        train_summary_acc = tf.summary.scalar(
            'Train_set_accuracy', acc_op)
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        # for test set
        test_summary_acc = tf.summary.scalar('Test_set_accuracy', acc_op)
        _, test_precision_op = tf.metrics.precision(
            labels=trainsformed_y, predictions=transformed_pred)
        test_summary_precision = tf.summary.scalar(
            'Test_set_precision', test_precision_op)
        _, test_recall_op = tf.metrics.recall(
            labels=trainsformed_y, predictions=transformed_pred)
        test_summary_recall = tf.summary.scalar(
            'Test_set_recall', test_recall_op)
        _, test_f1_score_op = tf.contrib.metrics.f1_score(
            labels=trainsformed_y, predictions=transformed_pred)
        test_summary_f1_score = tf.summary.scalar(
            'Test_set_f1_score', test_f1_score_op)

        # merge summary
        train_merged = tf.summary.merge([
            train_summary_loss,
            train_summary_acc
        ])
        test_merged = tf.summary.merge([
            test_summary_acc,
            test_summary_precision,
            test_summary_recall,
            test_summary_f1_score
        ])

        with tf.Session() as sess:
            print('Start trainning ...')
            writer = tf.summary.FileWriter('./logs/svm/', sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for i, X_batch, y_batch in self.mini_batch(self.iteration):
                sess.run(train_step, feed_dict={X: X_batch, y: y_batch})
                if i % 50 == 0:
                    step_loss, train_acc, summary = sess.run([loss, acc_op, train_merged], feed_dict={
                        X: X_batch, y: y_batch})
                    writer.add_summary(summary, i)
                    print("==========")
                    print("step: %s" % i)
                    print("train accuracy: %s" % train_acc)
                    print("loss: %s" % step_loss)

                    test_acc, test_precision, test_recall, test_f1_score, summary = sess.run([acc_op, test_precision_op, test_recall_op, test_f1_score_op, test_merged],
                                                                                             feed_dict={X: self.X_test, y: self.y_test})
                    writer.add_summary(summary, i)
                    print("test accuracy: %s" % test_acc)
                    print("test precision: %s" % test_precision)
                    print("test recall: %s" % test_recall)
                    print("test F1 score: %s" % test_f1_score)
                    sys.stdout.flush()
            trained_model_path = "./trained_models/svm/%s" % self.timestamp
            os.system('mkdir -p %s' % trained_model_path)
            save_path = saver.save(sess, trained_model_path + '/svm.ckpt')
            print("Model saved in : %s" % save_path)
            sys.stdout.flush()
            writer.close()


SVMHunter(sys.argv[1]).run()
