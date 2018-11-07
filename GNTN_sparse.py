import os
import copy
import numpy as np
import pandas as pd
import multiprocessing
from time import perf_counter
from scipy.sparse import csr_matrix, vstack
from sklearn import preprocessing
from sklearn.metrics import f1_score

def read_dataset(path, lbl_examples, id_file, filename):
    full_path_train = path + str(lbl_examples) + "/" + str(id_file) + "_" + filename + "_labeled.csv"
    full_path_test = path + str(lbl_examples) + "/" + str(id_file) + "_" + filename + "_unlabeled.csv"

    X_train = pd.read_csv(full_path_train)
    y_train = X_train.pop('class')

    X_test = pd.read_csv(full_path_test)
    y_test = X_test.pop('class')

    return X_train.to_sparse(), X_test.to_sparse(), y_train.values, y_test.values


def create_cooccurences_matrix(X):
    X[X > 0] = 1
    return X.T.dot(X)


def full_contingency_matrix(X_train, X_test):
    X = csr_matrix(vstack([X_train.to_coo(), X_test.to_coo()]), dtype=np.int32)

    cooc_matrix = create_cooccurences_matrix(X)
    total_docs = X.shape[0]

    p_11 = csr_matrix(cooc_matrix / total_docs, dtype=np.float16)

    diag = csr_matrix.diagonal(cooc_matrix)
    p_10 = np.divide(csr_matrix(diag[:, np.newaxis] - cooc_matrix, dtype=np.float16), total_docs, where=total_docs != 0).astype(np.float16)
    p_01 = np.divide(csr_matrix(diag - cooc_matrix, dtype=np.float16), total_docs, where=total_docs != 0).astype(np.float16)
    p_00 = (1.0 - (p_11 + p_10 + p_01).A).astype(np.float16)
    return np.array([p_11, p_10, p_01, p_00])

def support(cm):
    return cm[0]


def piatetsky_shapiro(cm):
    p_11 = cm[0]
    return csr_matrix(p_11.astype(np.float32) - (csr_matrix.diagonal(p_11) * csr_matrix.diagonal(p_11)[:, np.newaxis]), dtype=np.float16)


def spar(ps):
    return 1 - ps.getnnz() / (ps.shape[0] * ps.shape[1])


def yule(cm):
    p_11 = cm[0].astype(np.float32).A
    p_10 = cm[1].astype(np.float32).A #dense
    p_01 = cm[2].astype(np.float32).A
    p_00 = cm[3].astype(np.float32)   #dense

    part1 = p_11 * p_00
    del p_11
    del p_00
    part2 = p_10 * p_01
    del p_10
    del p_01
    denominator = (part1 + part2)

    return csr_matrix(np.divide(part1 - part2, denominator, where=denominator != 0), dtype=np.float16)


def mutual_information(cm):
    p_11 = cm[0].astype(np.float32).A
    p_10 = cm[1].astype(np.float32).A #dense
    p_01 = cm[2].astype(np.float32).A
    p_00 = cm[3].astype(np.float32)   #dense

    p_1_p_1 = (np.diag(p_11) * np.diag(p_11)[:, np.newaxis])
    p_1_p_0 = (np.diag(p_00) * np.diag(p_11)[:, np.newaxis])
    p_0_p_1 = (np.diag(p_11) * np.diag(p_00)[:, np.newaxis])
    p_0_p_0 = (np.diag(p_00) * np.diag(p_00)[:, np.newaxis])

    p1 = np.divide(p_11, p_1_p_1, where=p_1_p_1 != 0)
    del p_1_p_1
    p2 = np.divide(p_10, p_1_p_0, where=p_1_p_0 != 0)
    del p_1_p_0
    p3 = np.divide(p_01, p_0_p_1, where=p_0_p_1 != 0)
    del p_0_p_1
    p4 = np.divide(p_00, p_0_p_0, where=p_0_p_0 != 0)
    del p_0_p_0

    return csr_matrix(p_11 * np.log2(p1, where=p1 > 0) + \
           p_10 * np.log2(p2, where=p2 > 0) + \
           p_01 * np.log2(p3, where=p3 > 0) + \
           p_00 * np.log2(p4, where=p4 > 0), dtype=np.float16)


def kappa(cm):
    p_11 = cm[0].astype(np.float32).A
    p_00 = cm[3].astype(np.float32)

    p_1_p_1 = (np.diag(p_11) * np.diag(p_11)[:, np.newaxis])
    p_0_p_0 = (np.diag(p_00) * np.diag(p_00)[:, np.newaxis])

    aux = p_1_p_1 - p_0_p_0
    del p_1_p_1
    del p_0_p_0

    denominator = 1 - aux

    return csr_matrix(np.divide((p_11 + p_00 - aux), denominator, where=denominator != 0), dtype=np.float16)


def mutual_knn(adjacency_matrix, w=None):
    adj_matrix = np.minimum(adjacency_matrix, adjacency_matrix.T)

    # Solve isolated vertices problem adding an edge to its nearest neighbor
    if w is None:
        for i in np.where(~adj_matrix.any(axis=1))[0]:
            for j in range(adjacency_matrix[i].size):
                if adjacency_matrix[i, j] != 0:
                    adj_matrix[i, j] = adjacency_matrix[i,j]
                    adj_matrix[j, i] = adjacency_matrix[i,j]
                    break
    else:
        for i in np.where(~adj_matrix.any(axis=1))[0]:
            max_index, max_value = max(enumerate(w[i]), key=lambda x: x[1])
            adj_matrix[i, max_index] = max_value
            adj_matrix[max_index, i] = max_value

    return adj_matrix


def symmetric_knn(adjacency_matrix):
    return np.maximum(adjacency_matrix, adjacency_matrix.T)


# Top-K
def topK(W, k=2, kind=None):
    adj_matrix = np.zeros_like(W)
    max_values = np.argpartition(W, -k, axis=1)[:, -k:]

    for i in range(max_values.shape[0]):
        neighbors = np.array(max_values[i]).flatten()
        for j in range(neighbors.size):
            adj_matrix[i, neighbors[j]] = 1

    if kind:
        if kind == 'mutual':
            adj_matrix = mutual_knn(adj_matrix, W)
        elif kind == 'symmetric':
            adj_matrix = symmetric_knn(adj_matrix)

    result = np.multiply(adj_matrix, W)

    # Normalization
    min_number = np.min(result)
    if min_number < 0:
        result = np.abs(min_number) + result

    return result


def fractional_power(W):
    d = np.sum(W, axis=1)
    D = np.sqrt(d * d[:, np.newaxis])
    return np.divide(W, D, where=D != 0)


def initialize_scores_matrix(y, dt_matrix):
    sum_freq_total = dt_matrix.sum(axis=0)
    F = []

    with np.errstate(divide='ignore', invalid='ignore'):
        for j in np.unique(y):
            F.append(np.nan_to_num(np.squeeze(np.asarray(dt_matrix[np.where(y == j)].sum(axis=0) / sum_freq_total)),
                                   copy=False))

    return np.array(F).T


def llgc(alpha, S, F, Y_input_terms, n_iter, error = 0.0001):
    F_result = F
    F_old = copy.deepcopy(F)

    for iteration in range(n_iter):
        F_result = alpha * np.dot(S, F_result) + (1 - alpha) * Y_input_terms
        if np.sum(np.abs(F_old - F_result)) < error:
            break
        F_old = copy.deepcopy(F_result)

    return F_result, iteration


def classify(F):
    Y_result = np.zeros_like(F)
    Y_result[np.arange(len(F)), F.argmax(1)] = 1
    return Y_result


def classify_docs(X_train, X_test, F):
    X = csr_matrix(vstack([X_train.to_coo(), X_test.to_coo()]), dtype=np.int32)
    return X.dot(F)


def select_features_tfidf(X_train, X_test, percent=25):
    assert 0 < percent <= 100
    data = pd.concat([X_train, X_test])
    tf = data.iloc[:, :-1]
    idf = tf.fillna(0).astype(bool).sum(axis=0)
    idf = np.divide(tf.shape[0], idf, where=idf > 0)
    idf = np.log(idf, where=idf > 0)
    result = sorted(zip(tf.columns.values, (idf.values[:, np.newaxis].T * tf).sum(axis=0)), key=lambda x: x[1], reverse=True)
    return zip(*result[:len(result)*percent//100+1])


def select_features_tfidf_2(data, percent=25):
    assert 0 < percent <= 100
    tf = data.iloc[:, :-1]
    idf = tf.fillna(0).astype(bool).sum(axis=0)
    idf = np.divide(tf.shape[0], idf, where=idf > 0)
    idf = np.log(idf, where=idf > 0)
    result = sorted(zip(tf.columns.values, (idf.values[:, np.newaxis].T * tf).sum(axis=0)), key=lambda x: x[1], reverse=True)
    return zip(*result[:len(result)*percent//100+1])


def transform_data(data, features):
    return data[list(features)]


def tctn(datapath, lbl_examples, id_data, dataset_name, queue=None, alpha=0.1, n_iter=1000, k=7, sim_func=support):
    X_train, X_test, y_train, y_test = read_dataset(datapath, lbl_examples, id_data, dataset_name)

    fcm = full_contingency_matrix(X_train, X_test)

    A = sim_func(fcm)
    del fcm

    W = topK(A.astype(np.float32).A, k, 'mutual')
    del A

    F = initialize_scores_matrix(y_train, X_train.to_dense().values)
    Y_input_terms = copy.deepcopy(F)

    S = fractional_power(W)
    del W

    F, iteration = llgc(alpha, S, F, Y_input_terms, n_iter)

    del S
    del Y_input_terms

    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)

    F_docs = classify_docs(X_train, X_test, F)

    del F
    del X_train
    del X_test

    y_answer = classify(F_docs)

    del F_docs

    y_pred = lb.inverse_transform(y_answer)

    y = np.concatenate([y_train, y_test])

    f1_macro = f1_score(y_true=y, y_pred=y_pred, average='macro')
    f1_micro = f1_score(y_true=y, y_pred=y_pred, average='micro')
    f1_weighted = f1_score(y_true=y, y_pred=y_pred, average='weighted')

    if queue:
        queue.put((id_data, f1_macro, f1_micro, f1_weighted, iteration))

    return id_data, f1_macro, f1_micro, f1_weighted, iteration


def tctn_parallel(datapath, lbl_examples, dataset_name, alpha=0.1, n_iter=1000, k=7, n_jobs=10, n_folds=10, sim_func=support):
    if n_jobs < 1:
        n_jobs = multiprocessing.cpu_count()

    processes = []
    queue = multiprocessing.Queue()

    for id_data in range(1, n_folds+1, 1):  # 1-n_folds
        processes.append(multiprocessing.Process(target=tctn, args=(datapath, lbl_examples, id_data, dataset_name, queue, alpha, n_iter, k, sim_func)))

    i = 0
    while i < len(processes):
        sum_i = 0

        for _ in range(n_jobs):
            if i + sum_i < len(processes):
                processes[i + sum_i].start()
                sum_i += 1
            else:
                break

        for s in range(sum_i):
            processes[i + s].join()

        i += sum_i

    result = []
    while not queue.empty():
        res = np.array(queue.get())

        with open('./results/folds/' + dataset_name + '/' + str(k) + '_' + str(alpha) + '_' + str(lbl_examples) + '.txt', 'a+') as file:
            file.write("{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4}\n".format(*res))

        result.append(res)

    return np.vstack(result)


import warnings
warnings.filterwarnings('ignore')

def main():
    #filenames = ['tr23.wc', 'CSTR', 'tr12.wc', 'SyskillWebert', 'tr21.wc', 'tr11.wc', 'oh15.wc', 'oh5.wc', 'oh0.wc',
    #             'oh10.wc', 'tr45.wc', 'tr41.wc', 'tr31.wc', 'wap.wc', 're0.wc', 're1.wc', 'IrishEconomicSentiment',
    #             'review_polarity', 'Hitech', 'la2s']
    filenames = ['tr23.wc','tr12.wc','tr11.wc','tr21.wc','CSTR','tr41.wc','re0.wc','tr31.wc','tr45.wc','re1.wc']
    sim_funcs = [piatetsky_shapiro, mutual_information, kappa, yule, support]
    n_processors = 10
    n_iter = 1000
    n_folds = 10

    k_list = [7, 17, 37, 57]
    alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    lbl_list = [1, 10, 20, 30, 40, 50]

    path = './data_splited/'

    if not os.path.exists("./results/"):
        os.makedirs("./results/")

    for sfunc in sim_funcs:
        for filename in filenames:
            if not os.path.exists('./results/folds/' + filename + '/'):
                os.makedirs('./results/folds/' + filename + '/')

            with open("./results/" + filename + ".txt", "a+") as file:
                file.write("Dataset: {} n_folds: {} max_iter: {}\n".format(filename, n_folds, n_iter))
                file.write("Sim. Func.\tk\talpha\tn_lbl\tAvg. F1 Macro\tAvg. F1 Micro\tAvg. F1 Weighted\tAvg. Iteration\tStd. F1 Macro\tStd. F1 Micro\tStd. F1 Weighted\tStd. Iteration\n")

            for lbl in lbl_list:
                for k in k_list:
                    for alpha in alpha_list:
                        if len(os.listdir(path + filename + '/' + str(lbl) + '/')) == 0:
                            #print("Skipped.")
                            continue

                        print("\nfile {} k {} alpha {} lbl {} simf {}".format(filename, k, alpha, lbl, sfunc.__name__))

                        with open('./results/folds/' + filename + '/' + str(k) + '_' + str(alpha) + '_' + str(lbl) + '.txt', 'a+') as file:
                            file.write("Dataset: {} k: {} alpha: {} max_iter: {} sim.func.: {}\n".format(filename, k, alpha, n_iter, sfunc.__name__))
                            file.write("ID\tF1 Macro\tF1 Micro\tF1 Weighted\tIteration\n")

                        start_execution = perf_counter()
                        result = tctn_parallel(path+filename+'/', lbl, filename, alpha=alpha, n_iter=n_iter, k=k, n_jobs=n_processors, n_folds=n_folds, sim_func=sfunc)
                        print("\tAvg. Macro: {0:.4f}\tMicro: {1:.4f}\tWeighted: {2:.4f}\tIteration: {3:.4f}".format(*result.mean(axis=0)[1:]))
                        print("\tStd. Macro: {0:.4f}\tMicro: {1:.4f}\tWeighted: {2:.4f}\tIteration: {3:.4f}".format(*result.std(axis=0)[1:]))
                        print("\tExecution time: {:.4f}".format(perf_counter() - start_execution))

                        with open("./results/" + filename + ".txt", "a+") as file:
                            file.write(sfunc.__name__+ '\t' +str(k) + '\t' + str(alpha) + '\t' + str(lbl))
                            file.write("\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}".format(*result.mean(axis=0)[1:]))
                            file.write("\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\n".format(*result.std(axis=0)[1:]))


if __name__ == '__main__':
    main()