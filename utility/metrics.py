import numpy as np


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[:1] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def average_precision(r, k):
    r = np.asarray(r)[:k]
    out = [precision_at_k(r, i+1) for i in range(k) if r[i]]
    if not out:
        return 0.
    return np.sum(out) / float(np.sum(r))


def mrr_at_k(r, k):
    assert k >= 1
    res = 0
    r = np.asarray(r)[:k]
    for i in range(k):
        if r[i] == 1:
            res = 1/(i+1)
            break
    return res


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.
