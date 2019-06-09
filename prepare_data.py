import sqlite3
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import yaml

def load_from_db(cursor, status):
    sql = "SELECT file_name, content, indices, status FROM code_snippets WHERE status = {}".format(
        str(status))
    cursor.execute(sql)
    snippets = []
    for row in cursor:
        snippets.append({
            'file_name': row[0],
            'content': row[1],
            'indices': (yaml.load(row[2] or '') or []),
            'status': row[3]
        })
    return snippets


def load_data(db):
    cursor = sqlite3.connect(db).cursor()
    normal_snippets = load_from_db(cursor, 0)
    unsure_snippets = load_from_db(cursor, 2)
    leaked_snippets = load_from_db(cursor, 1)
    return normal_snippets, unsure_snippets, leaked_snippets


def load_data_with_mask(db, treat_unsure_as_leak=False):
    normal, unsure, leaked = load_data(db)
    normal = list(map(lambda snippet_dict: {
        'file_name': snippet_dict['file_name'],
        'content': list(snippet_dict['content']),
        'mask': indice_to_mask(snippet_dict, True),
        'status': 0
    }, normal))
    unsure = list(map(lambda snippet_dict: {
        'file_name': snippet_dict['file_name'],
        'content': list(snippet_dict['content']),
        'mask': indice_to_mask(snippet_dict, treat_unsure_as_leak),
        'status': 1 if treat_unsure_as_leak else 0
    }, unsure))
    leaked = list(map(lambda snippet_dict: {
        'file_name': snippet_dict['file_name'],
        'content': list(snippet_dict['content']),
        'mask': indice_to_mask(snippet_dict, False),
        'status': 1
    }, leaked))
    return normal + unsure + leaked


def indice_to_mask(snippet_dict, force_normal=False):
    length = len(snippet_dict['content'])
    mask = [0] * length
    if len(snippet_dict['indices']) > 0 and not force_normal:
        for indice in snippet_dict['indices']:
            mask[indice[0]: indice[1] + 1] = [1] * (indice[1] - indice[0] + 1)
    return mask


def mini_batch(x_data_set, y_data_set, batch_size=128):
    i = 0
    while True:
        start = (i * batch_size) % len(x_data_set)
        end = ((i + 1) * batch_size) % len(x_data_set)
        if start > end:
            yield i, x_data_set[start:] + x_data_set[0:end], y_data_set[start:] + y_data_set[0:end]
        else:
            yield i, x_data_set[start:end], y_data_set[start:end]
        i += 1


def convert_to_ascii_code(data_set):
    for item in data_set:
        new_content = []
        new_mask = []
        for index, char in enumerate(item['content']):
            if ord(char) < 128:
                new_content.append(ord(char))
                new_mask.append(item['mask'][index])
        item['content'] = new_content
        item['new_mask'] = new_mask
    return data_set

def convert_to_ascii_code_with_padding(x_data_set, y_data_set):
    x = list(map(lambda item: item['content'], x_data_set))
    max_len = len(max(x, key=lambda content: len(content)))
    # length_tuple contains effective length for each sequence (without padding)
    seq_length = list(map(lambda content: len(content), x))
    x = pad_sequences(x, maxlen=max_len, dtype=int, padding='post', value=-1)
    y = pad_sequences(y_data_set, maxlen=max_len, dtype=int, padding='post', value=-1)
    return np.array(x), np.array(y), np.array(seq_length)
