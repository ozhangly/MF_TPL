import re
import numpy as np
import utility.config as config
import utility.utils as utils


args = config.args


def app_sim_computed(relation: np.ndarray) -> None:
    # 有两个需要保存的文件
    utils.ensure_dir(args.similarity_path)
    rmv_fold = re.findall('[0-9]', args.training_dataset)
    v_file_name = args.similarity_path + '%d_%d/maxVU.txt' % (rmv_fold[0], rmv_fold[1])
    p_file_name = args.similarity_path + '%d_%d/maxPU.txt' % (rmv_fold[0], rmv_fold[1])
    if utils.file_exists(v_file_name) and utils.file_exists(p_file_name):
        return

    remain = relation                                   # [size_app, size_lib]
    sum_relation = np.sum(remain, axis=0)               # [size_lib,]
    ref_relation = remain.T                             # [size_lib, size_app]
    sum_ref_relation = np.sum(ref_relation, axis=0)     # [size_app,]
    (size_app, size_lib) = relation.shape
    simiU = np.zeros(shape=(size_app, size_app))        # simiU: [size_app, size_lib]

    for u in range(size_app):
        user_u = ref_relation[:, u]         # user_u: [size_lib,]
        fz_tmp = np.dot(relation, user_u)   # fz_tmp: [size_app, ]
        fm_tmp = (sum_ref_relation[u] + sum_ref_relation).T - fz_tmp  # 可以进行逐元素运算
        simiU[:, u] = fz_tmp / fm_tmp
        simiU[u, u] = 0                     # 自己和自己的相似度为0

    # 需要对simiU进行排序运算
    # 对相似矩阵的列进行降序排序
    sortA = np.sort(simiU, axis=0)[::-1]            # sortA: [size_app, size_app]
    sortA_idx = np.argsort(simiU, axis=0)[::-1]
    maxVU = sortA[:args.top_k, :]                   # maxVU: [top_k, size_app]
    maxPU = sortA_idx[:args.top_k, :]               # maxPU: [top_k, size_app]
    maxW = np.sum(sortA, axis=0)                    # maxW: [size_app,]
    for u in range(size_app):
        maxVU[:, u] = maxVU[:, u] / maxW[u]

    np.savetxt(fname=v_file_name, X=maxVU)
    np.savetxt(fname=p_file_name, X=maxPU)


def lib_sim_computed(relation: np.ndarray) -> None:
    utils.ensure_dir(args.similarity_path)
    rmv_fold = re.findall('[0-9]', args.training_dataset)
    v_file_name = args.similarity_path + '%d_%d/maxVI.txt' % (rmv_fold[0], rmv_fold[1])
    p_file_name = args.similarity_path + '%d_%d/maxPI.txt' % (rmv_fold[0], rmv_fold[1])
    if utils.file_exists(v_file_name) and utils.file_exists(p_file_name):
        return

    remain = relation                                   # remain: [size_app, size_lib]
    sum_relation = np.sum(relation, axis=0)             # sum_relation: [size_lib, ]
    ref_relation = relation.T                           # ref_relation: [size_lib, size_app]
    sum_ref_relation = np.sum(ref_relation, axis=0)     # sum_ref_relation: [size_app, ]
    (size_app, size_lib) = relation.shape

    simiL = np.zeros(shape=(size_lib, size_lib))
    for i in range(size_lib):
        item_i = relation[:, i]                         # item_i: [size_app, ]
        fz_tmp = np.dot(ref_relation, item_i)           # fz_tmp: [size_lib, ]
        fm_tmp = (sum_relation[i] + sum_relation).T - fz_tmp
        simiL[:, i] = fz_tmp / fm_tmp
        simiL[i, i] = 0

    sortA = np.sort(simiL, axis=0)[::-1]
    sortA_idx = np.argsort(simiL, axis=0)[::-1]
    maxVI = sortA[:args.top_k, :]
    maxPI = sortA_idx[:args.top_k, :]

    maxW = np.sum(maxVI, axis=0)
    for i in range(size_lib):
        maxVI[:, i] = maxVI[:, i] / maxW[i]

    np.savetxt(fname=v_file_name, X=maxVI)
    np.savetxt(fname=p_file_name, X=maxPI)

