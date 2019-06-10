import sys
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import prepare_data
from hparams import *
from lib import bi_directional_gru

if __name__ == '__main__':
    print('[info] Preparing data ...', flush=True)
    data_hparams = data_hparams.params()
    hparams = bi_directional_gru_hparams.params()
    X = prepare_data.load_data_with_mask(data_hparams['db_path'])
    X = prepare_data.convert_to_ascii_code(X)
    y = list(map(lambda x: x['mask'], X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_test, y_test, test_seq_length = prepare_data.convert_to_ascii_code_with_padding(X_test, y_test)

    model = bi_directional_gru.BiDirectionalGRU().load_hparams(
        max_char_num=hparams['max_char_num'],
        fw_num_units=hparams['fw_num_units'],
        bw_num_units=hparams['bw_num_units'],
        loss_weight_fraction=hparams['loss_weight_fraction']
    )
    ops = model.build_model()

    with tf.Session() as sess:
        print('[info] Start Trainning ...', flush=True)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        writer = tf.summary.FileWriter(hparams['log_path'], sess.graph)
        saver = tf.train.Saver(max_to_keep=100)
        for i, X_batch, y_batch in prepare_data.mini_batch(X_train, y_train, batch_size=hparams['batch_size']):
            if i > hparams['iter']:
                break
            X_batch, y_batch, seq_length = prepare_data.convert_to_ascii_code_with_padding(X_batch, y_batch)
            step_loss, accuracy, _ , train_summary = sess.run(
                [ops['loss'], ops['accuracy'], ops['train_op'], ops['train_merge']],
                feed_dict={
                    ops['X']: X_batch,
                    ops['y']: y_batch,
                    ops['seq_length']: seq_length,
                    ops['max_len']: X_batch.shape[-1]
                }
            )
            writer.add_summary(train_summary, i)
            print("Iteration %d, loss: %f, accuracy: %f" % (i, step_loss, accuracy), flush=True)
            if i % 50 == 0:
                loss, accuracy, precision, recall, test_summary = sess.run(
                    [ops['loss'], ops['accuracy'], ops['precision'], ops['recall'], ops['test_merge']],
                    feed_dict={
                        ops['X']: X_test,
                        ops['y']: y_test,
                        ops['seq_length']: test_seq_length,
                        ops['max_len']: X_test.shape[-1]
                    }
                )
                writer.add_summary(test_summary, i)
                print('>>>>>>>>>>>>>>>>>>>>>>', flush=True)
                print("Iteration: %d" % i, flush=True)
                print("Loss: %f" % loss, flush=True)
                print("Accuracy: %f" % accuracy, flush=True)
                print("Precision: %f" % precision, flush=True)
                print("Recall: %f" % recall, flush=True)
            if (i % 500 == 0 or i == hparams['iter']) and i != 0:
                save_dir = hparams['checkpoint_save_path']
                os.system('mkdir -p %s' % save_dir)
                save_path = saver.save(sess, save_dir + "/model", global_step=i)
                print("Model saved in : %s" % save_path)
        writer.close()