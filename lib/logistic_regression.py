from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import scipy.sparse as sp
import os
import pickle


# Best hyper params
# {   'tfidf_lowercase': True,
#     'tfidf_analyzer': 'char',
#     'tfidf_ngram_range': (1, 3),
#     'tfidf_norm': None,
#     'tfidf_use_idf': True,
#     'tfidf_sublinear_tf': True,
#     'logistic_penalty': 'l2',
#     'logistic_C': 0.2,
#     'logistic_class_weight': {0: 0.06638550452983442, 1: 1.0}}
# }

def run_with_hyper_params(all_hyper_params, content, file_name, y):
    # initialize
    max_f1_score = 0.0
    hyper_params_with_max_f1_score = {
        k: all_hyper_params[k][0] for k in all_hyper_params.keys()
    }

    print('Total Hyper Params Combination: %d' %
          sum(1 for hp in hyper_params_generator(all_hyper_params)))

    index = 1
    # enumerate through all combination of hyper params
    for hyper_params in hyper_params_generator(all_hyper_params):
        f1_score, _ = run_one_pass(hyper_params, content, file_name, y)
        print('------------------')
        print('index: %d' % index)
        print('F1 Score: %f' % f1_score)
        print('Hyper Params: %s' % hyper_params)
        index += 1
        if f1_score > max_f1_score:
            max_f1_score = f1_score
            hyper_params_with_max_f1_score = hyper_params

    print('======= Final  ======')
    print('Highest F1 Score: %f' % max_f1_score)
    print('Best Hyper Params: %s' % hyper_params_with_max_f1_score)


def hyper_params_generator(all_hyper_params):
    all_keys = list(all_hyper_params.keys())
    all_values = [all_hyper_params[k] for k in all_keys]
    current_index = [0 for _ in all_keys]
    hyper_params = {all_keys[i]: all_values[i][current_index[i]]
                    for i in range(0, len(current_index))}
    yield hyper_params

    def next_index(current_index):
        pointer = len(current_index) - 1
        while pointer >= 0:
            if current_index[pointer] >= len(all_values[pointer]) - 1:
                current_index[pointer] = 0
                pointer -= 1
            else:
                current_index[pointer] += 1
                break
        return current_index if pointer >= 0 else None

    while True:
        current_index = next_index(current_index)

        if current_index is None:
            break
        hyper_params = {all_keys[i]: all_values[i][current_index[i]]
                        for i in range(0, len(current_index))}
        yield hyper_params


def run_one_pass(hyper_params, content, file_name, y):
    vectorizer = TfidfVectorizer(
        min_df=0.0,
        analyzer=hyper_params['tfidf_analyzer'],
        lowercase=hyper_params['tfidf_lowercase'],
        ngram_range=hyper_params['tfidf_ngram_range'],
        norm=hyper_params['tfidf_norm'],
        use_idf=hyper_params['tfidf_use_idf'],
        sublinear_tf=hyper_params['tfidf_sublinear_tf']
    )
    logistic = LogisticRegression(
        penalty=hyper_params['logistic_penalty'],
        C=hyper_params['logistic_C'],
        class_weight=hyper_params['logistic_class_weight']
    )
    X_content = vectorizer.fit_transform(content)
    X_file_name = vectorizer.fit_transform(file_name)
    X = sp.hstack((X_file_name, X_content))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    logistic.fit(X_train, y_train)
    predicted = logistic.predict(X_test)

    f1_score = metrics.f1_score(y_test, predicted)
    fpr, tpr, _ = metrics.roc_curve(
        y_test, (logistic.predict_proba(X_test)[:, 1]))
    auc = metrics.auc(fpr, tpr)
    print("Accuracy: %f" % logistic.score(
        X_test, y_test))  # checking the accuracy
    print("Precision: %f" % metrics.precision_score(y_test, predicted))
    print("Recall: %f" % metrics.recall_score(y_test, predicted))
    print("F1-Score: %f" % f1_score)
    print("AUC: %f" % auc)

    return f1_score, logistic


with open('pickles/normal_snippets.pickle', 'rb') as f:
    normal = pickle.load(f, encoding='ASCII')
    normal_content = list(map(lambda x: x['content'],  normal))
    normal_file_name = list(map(lambda x: x['file_name'],  normal))

with open('pickles/unsure_snippets.pickle', 'rb') as f:
    unsure = pickle.load(f)
    unsure_content = list(map(lambda x: x['content'],  unsure))
    unsure_file_name = list(map(lambda x: x['file_name'],  unsure))

with open('pickles/leaked_snippets.pickle', 'rb') as f:
    leaked = pickle.load(f)
    leaked_content = list(map(lambda x: x['content'],  leaked))
    leaked_file_name = list(map(lambda x: x['file_name'],  leaked))

content = normal_content + unsure_content + leaked_content
file_name = normal_file_name + unsure_file_name + leaked_file_name
y = [0 for i in range(0, len(normal_content))] + \
    [1 for i in range(0, len(leaked_content + unsure_content))]

all_hyper_params = {
    'tfidf_lowercase': [True, False],
    'tfidf_analyzer': ['word', 'char', 'char_wb'],
    'tfidf_ngram_range': [(1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (5, 6)],
    'tfidf_norm': ['l2', None],
    'tfidf_use_idf': [True, False],
    'tfidf_sublinear_tf': [True, False],

    'logistic_penalty': ['l1', 'l2'],
    'logistic_C': [1.0, 0.6, 0.2],
    'logistic_class_weight': [
        None,
        'balanced',
        {0: len(leaked_content + unsure_content) /
         len(normal_content), 1: 1.0},
        {0: len(leaked_content + unsure_content) /
         (2 * len(normal_content)), 1: 1.0},
        {0: len(leaked_content + unsure_content) /
         (5 * len(normal_content)), 1: 1.0}
    ]
}

# run_with_hyper_params(all_hyper_params, content, file_name, y)


best_hyper_params = {
    'tfidf_lowercase': True,
    'tfidf_analyzer': 'char',
    'tfidf_ngram_range': (1, 3),
    'tfidf_norm': None,
    'tfidf_use_idf': True,
    'tfidf_sublinear_tf': True,
    'logistic_penalty': 'l2',
    'logistic_C': 0.2,
    'logistic_class_weight': {0: 0.06638550452983442, 1: 1.0}
}

highest_f1_score, logistic = run_one_pass(
    best_hyper_params, content, file_name, y)

# Accuracy: 0.947658
# Precision: 0.783784
# Recall: 0.725000
# F1-Score: 0.753247
# AUC: 0.961378
