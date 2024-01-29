import re
import numpy as np
import utility.config as config
import utility.utils as utils


args = config.args


def app_sim_computed(relation: np.ndarray) -> None:
    # 有两个需要保存的文件
    rmv_fold = re.findall('[0-9]', args.training_dataset)
    utils.ensure_dir(args.similarity_path + '%s_%s/' % (rmv_fold[0], rmv_fold[1]))
    v_file_name = args.similarity_path + '%s_%s/maxVU.txt' % (rmv_fold[0], rmv_fold[1])
    p_file_name = args.similarity_path + '%s_%s/maxPU.txt' % (rmv_fold[0], rmv_fold[1])
    if utils.file_exists(v_file_name) and utils.file_exists(p_file_name):
        return

    remain = relation                                   # [size_app, size_lib]
    ref_relation = remain.T                             # [size_lib, size_app]
    sum_ref_relation = np.sum(ref_relation, axis=0)     # [size_app,]
    (size_app, size_lib) = relation.shape
    simiU = np.zeros(shape=(size_app, size_app), dtype=np.float16)        # simiU: [size_app, size_app]

    for u in range(size_app):
        user_u = ref_relation[:, u]         # user_u: [size_lib,]
        fz_tmp = np.dot(relation, user_u)   # fz_tmp: [size_app, ]
        fm_tmp = (sum_ref_relation[u] + sum_ref_relation).T - fz_tmp  # 可以进行逐元素运算
        simiU[:, u] = fz_tmp / fm_tmp
        simiU[u, u] = 0                     # 自己和自己的相似度为0

    del remain, ref_relation, sum_ref_relation

    # 需要对simiU进行排序运算
    # 对相似矩阵的列进行降序排序
    sortA = np.sort(simiU, axis=0)[::-1]            # sortA: [size_app, size_app]
    sortA_idx = np.argsort(simiU, axis=0)[::-1]
    maxVU = sortA[:args.top_k, :]                   # maxVU: [top_k, size_app]
    maxPU = sortA_idx[:args.top_k, :]               # maxPU: [top_k, size_app]
    maxW = np.sum(sortA, axis=0)                    # maxW: [size_app,]
    for u in range(size_app):
        maxVU[:, u] = maxVU[:, u] / maxW[u]

    np.savetxt(fname=v_file_name, X=maxVU, fmt='%.4f')
    np.savetxt(fname=p_file_name, X=maxPU, fmt='%.4f')


def lib_sim_computed(relation: np.ndarray) -> None:
    rmv_fold = re.findall('[0-9]', args.training_dataset)
    utils.ensure_dir(args.similarity_path + '%s_%s/' % (rmv_fold[0], rmv_fold[1]))
    v_file_name = args.similarity_path + '%s_%s/maxVI.txt' % (rmv_fold[0], rmv_fold[1])
    p_file_name = args.similarity_path + '%s_%s/maxPI.txt' % (rmv_fold[0], rmv_fold[1])
    if utils.file_exists(v_file_name) and utils.file_exists(p_file_name):
        return

    sum_relation = np.sum(relation, axis=0)             # sum_relation: [size_lib, ]
    ref_relation = relation.T                           # ref_relation: [size_lib, size_app]
    (size_app, size_lib) = relation.shape

    simiL = np.zeros(shape=(size_lib, size_lib))
    for i in range(size_lib):
        item_i = relation[:, i]                         # item_i: [size_app, ]
        fz_tmp = np.dot(ref_relation, item_i)           # fz_tmp: [size_lib, ]
        fm_tmp = (sum_relation[i] + sum_relation).T - fz_tmp
        simiL[:, i] = fz_tmp / fm_tmp
        simiL[i, i] = 0

    del sum_relation, ref_relation

    sortA = np.sort(simiL, axis=0)[::-1]
    sortA_idx = np.argsort(simiL, axis=0)[::-1]
    maxVI = sortA[:args.top_k, :]
    maxPI = sortA_idx[:args.top_k, :]

    maxW = np.sum(maxVI, axis=0)
    for i in range(size_lib):
        maxVI[:, i] = maxVI[:, i] / maxW[i]

    np.savetxt(fname=v_file_name, X=maxVI, fmt='%.4f')
    np.savetxt(fname=p_file_name, X=maxPI, fmt='%.4f')

