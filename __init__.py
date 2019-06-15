import os.path
import tensorflow as tf
import numpy as np

class DeepHunter():
    def __init__(self, model_path=None):
        self.graph = tf.Graph()
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.default_model = os.path.join(self.root, 'trained_models', 'bi_directional_gru', 'model-10000')
        if model_path is None:
            model_path = self.default_model
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(model_path + '.meta')
            saver.restore(self.sess, model_path)

    def predict(self, code_str):
        X, seq_length, max_len, not_ascii_char_index = self.convert_to_ascii(code_str)
        X_place_holder = self.graph.get_tensor_by_name('X:0')
        seq_length_place_holder = self.graph.get_tensor_by_name('seq_length:0')
        max_len_place_holder = self.graph.get_tensor_by_name('max_len:0')
        prediction_tensor = self.graph.get_tensor_by_name('prediction:0')
        prediction = self.sess.run(
            [prediction_tensor],
            feed_dict={
                X_place_holder: X,
                seq_length_place_holder: seq_length,
                max_len_place_holder: max_len
            }
        )[0]
        prediction = prediction.tolist()
        prediction = self.insert_not_ascii_char_into_prediction(prediction, not_ascii_char_index, code_str)
        return prediction

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if hasattr(self, 'sess'):
            self.sess.close()

    def insert_not_ascii_char_into_prediction(self, prediction, not_ascii_char_index, code_str):
        i = 0
        result = []
        for idx in range(0, len(code_str)):
            if idx in not_ascii_char_index:
                result.append(0)
            else:
                result.append(prediction[i])
                i += 1
        return result

    def convert_to_ascii(self, code_str):
        ascii_char = []
        not_ascii_char_index = []
        for idx, char in enumerate(code_str):
            if ord(char) < 128:
                ascii_char.append(ord(char))
            else:
                not_ascii_char_index.append(idx)
        return np.array([ascii_char]), np.array([len(ascii_char)]), len(ascii_char), not_ascii_char_index

    def ___del__(self):
        self.close()