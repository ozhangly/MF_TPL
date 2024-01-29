import re
import numpy as np
import utility.config as config
import utility.utils as utils

from tqdm import tqdm


args = config.args


def app_sim_computed(relation: np.ndarray) -> None:
    # 有两个需要保存的文件
    rmv_fold = re.findall('[0-9]', args.training_dataset)
    utils.ensure_dir(args.similarity_path + '%s_%s/' % (rmv_fold[0], rmv_fold[1]))
    v_file_name = args.similarity_path + '%s_%s/maxVU.txt' % (rmv_fold[0], rmv_fold[1])
    p_file_name = args.similarity_path + '%s_%s/maxPU.txt' % (rmv_fold[0], rmv_fold[1])
    if utils.file_exists(v_file_name) and utils.file_exists(p_file_name):
        return

    ref_relation = relation.T                               # [size_lib, size_app]
    sum_ref_relation = np.sum(ref_relation, axis=0)         # [size_app,]
    (size_app, size_lib) = relation.shape
    simiU = np.zeros(shape=(size_app, size_app), dtype=np.float16)        # simiU: [size_app, size_app]

    app_sim_com_bar = tqdm(desc='computing app similarity...', leave=False, total=size_app)

    for u in range(size_app):
        user_u = ref_relation[:, u]         # user_u: [size_lib,]
        fz_tmp = np.dot(relation, user_u)   # fz_tmp: [size_app, ]
        fm_tmp = (sum_ref_relation[u] + sum_ref_relation).T - fz_tmp  # 可以进行逐元素运算
        simiU[:, u] = fz_tmp / fm_tmp
        simiU[u, u] = 0                     # 自己和自己的相似度为 0
        app_sim_com_bar.update()

    app_sim_com_bar.close()
    del ref_relation, sum_ref_relation, app_sim_com_bar, relation

    # 需要对simiU进行排序运算
    # 对相似矩阵的列进行降序排序
    sort_app_bar = tqdm(desc='sorting app similarity...', total=size_app, leave=False)
    maxPU = np.zeros(shape=(args.top_k, size_app)).astype(np.uint16)
    for u in range(size_app):
        user_u = simiU[:, u].astype(np.float16)
        sort_user = np.sort(user_u)[::-1].astype(np.float16)
        maxPU[:args.top_k, u] = np.argsort(user_u)[::-1][:args.top_k].astype(np.uint16)
        simiU[:, u] = sort_user
        sort_app_bar.update()

    sort_app_bar.close()
    del sort_app_bar, user_u, sort_user

    maxVU = simiU[:args.top_k, :].astype(np.float16)            # maxVU: [top_k, size_app]
    maxW = np.sum(simiU, axis=0).astype(np.float16)             # maxW: [size_app,]

    del simiU

    app_sim_normal_bar = tqdm(desc='normalizing sim...', total=size_app, leave=False)
    for u in range(size_app):
        maxVU[:, u] = maxVU[:, u] / maxW[u]
        app_sim_normal_bar.update()
    app_sim_normal_bar.close()
    del app_sim_normal_bar

    np.savetxt(fname=v_file_name, X=maxVU, fmt='%.4f')
    np.savetxt(fname=p_file_name, X=maxPU, fmt='%d')


def lib_sim_computed(relation: np.ndarray) -> None:
    rmv_fold = re.findall('[0-9]', args.training_dataset)
    utils.ensure_dir(args.similarity_path + '%s_%s/' % (rmv_fold[0], rmv_fold[1]))
    v_file_name = args.similarity_path + '%s_%s/maxVI.txt' % (rmv_fold[0], rmv_fold[1])
    p_file_name = args.similarity_path + '%s_%s/maxPI.txt' % (rmv_fold[0], rmv_fold[1])
    if utils.file_exists(v_file_name) and utils.file_exists(p_file_name):
        return

    sum_relation = np.sum(relation, axis=0).astype(np.float16)      # sum_relation: [size_lib, ]
    ref_relation = relation.T                                       # ref_relation: [size_lib, size_app]
    (size_app, size_lib) = relation.shape

    simiL = np.zeros(shape=(size_lib, size_lib), dtype=np.float16)

    lib_sim_com_bar = tqdm(desc='computing lib sim...', leave=False, total=size_lib)
    for i in range(size_lib):
        item_i = relation[:, i]                                     # item_i: [size_app, ]
        fz_tmp = np.dot(ref_relation, item_i)                       # fz_tmp: [size_lib, ]
        fm_tmp = (sum_relation[i] + sum_relation).T - fz_tmp
        simiL[:, i] = fz_tmp / fm_tmp
        simiL[i, i] = 0
        lib_sim_com_bar.update()
    lib_sim_com_bar.close()
    del sum_relation, ref_relation, lib_sim_com_bar

    sort_lib_bar = tqdm(desc='sorting lib similarity...', leave=False, total=size_lib)
    maxPI = np.zeros(shape=(args.top_k, size_lib), dtype=np.uint16)
    for i in range(size_lib):
        item_i = simiL[:, i].astype(np.float16)
        sort_item = np.sort(item_i)[::-1].astype(np.float16)
        maxPI[:args.top_k, i] = np.argsort(item_i)[::-1][:args.top_k].astype(np.uint16)
        simiL[:, i] = sort_item
        sort_lib_bar.update()

    sort_lib_bar.close()
    del sort_lib_bar, item_i, sort_item

    maxVI = simiL[:args.top_k, :].astype(np.float16)
    maxW = np.sum(maxVI, axis=0).astype(np.float16)

    del simiL

    lib_sim_normal_bar = tqdm(desc='normalize lib sim...', total=size_lib, leave=False)
    for i in range(size_lib):
        maxVI[:, i] = maxVI[:, i] / maxW[i]
        lib_sim_normal_bar.update()
    lib_sim_normal_bar.close()
    del lib_sim_normal_bar

    np.savetxt(fname=v_file_name, X=maxVI, fmt='%.4f')
    np.savetxt(fname=p_file_name, X=maxPI, fmt='%d')

