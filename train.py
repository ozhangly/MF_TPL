import re
import utility.utils
import utility.sim_computed
import utility.load_data
import utility.config

import numpy as np

from tqdm import tqdm
from time import time

args = utility.config.args


def train(epochs) -> None:
    fold_rmv = re.findall('[0-9]', args.training_dataset)

    train_file = args.training_path + args.training_dataset
    print('loading data...')
    load_st = time()
    relation = utility.load_data.load_relation_mat(train_file_path=train_file)
    print('load data completed. [%.2fs]' % (time() - load_st))

    # 计算app相似度和lib相似度
    print('compute app similarity...')
    app_sim_st = time()
    utility.sim_computed.app_sim_computed(relation)
    print('app similarity compute completed. [%.2fs]' % (time() - app_sim_st))
    print('compute lib similarity...')
    lib_sim_st = time()
    utility.sim_computed.lib_sim_computed(relation)
    print('lib similarity compute completed. [%.2fs]' % (time() - lib_sim_st))

    (size_app, size_lib) = relation.shape

    log_weight = args.weight / (np.log(np.sum(relation, axis=0) + 1) + 1)

    maxVI = np.loadtxt(fname=args.similarity_path + '%s_%s/maxVI.txt' % (fold_rmv[0], fold_rmv[1]))
    maxPI = np.loadtxt(fname=args.similarity_path + '%s_%s/maxPI.txt' % (fold_rmv[0], fold_rmv[1]), dtype=np.uint16)
    maxPU = np.loadtxt(fname=args.similarity_path + '%s_%s/maxPU.txt' % (fold_rmv[0], fold_rmv[1]), dtype=np.uint16)
    maxVU = np.loadtxt(fname=args.similarity_path + '%s_%s/maxVU.txt' % (fold_rmv[0], fold_rmv[1]))

    C = np.zeros(shape=(size_app, size_lib))
    rd = np.random.RandomState(200)
    X = rd.random(size=(size_app, args.factor)) + 0.01
    Y = rd.random(size=(size_lib, args.factor)) + 0.01

    for i in range(size_lib):
        C[:, i] = 1 + log_weight[i]*relation[:, i]

    position = np.zeros(shape=(size_app, 10), dtype=np.uint16)

    # 准备工作都已做好
    for epoch in range(epochs):
        print(f'>>>>>>>>>>>>>>>>>>>>>epoch{epoch}<<<<<<<<<<<<<<<<<<<<<<')
        epoch_st = time()
        YtY = np.dot(Y.T, Y)                             # YtY: [factor, factor]

        update_app_bar = tqdm(desc='updating app vector...', total=size_app, leave=True)
        for u in range(size_app):
            Cu = C[u, :].T                               # Cu: [size_lib, ]
            Pu = relation[u, :].T                        # Pu: [size_lib, ]
            hou = Cu * Pu                                # hou: [size_lib, ]
            hou = np.dot(Y.T, hou)                       # hou: [factor, ]
            Nu = X[maxPU[:, u], :].T                     # Nu: [factor, top_k]
            WuNormal = maxVU[:, u]                       # WuNormal: [top_k, ]
            Wu = WuNormal / np.sum(WuNormal)             # Wu: [top_k, ]
            hou = hou + args.alpha * np.dot(Nu, Wu)      # hou: [factor, ]

            Cu = Cu - 1                                  # Cu: [size_lib, ]
            qian = Y.T                                   # qian: [factor, size_lib]
            for j in range(size_lib):
                qian[:, j] = qian[:, j] * Cu[j]
            qian = np.dot(qian, Y)                       # qian: [factor, factor]
            qian = qian + YtY                            # qian: [factor, factor]
            qian = qian + args.lmda + args.alpha         # qian: [factor, factor]
            qian = np.nan_to_num(qian)
            condition_num = np.linalg.cond(qian)
            if condition_num > 1e10:                     #
                Xu = np.dot(np.linalg.pinv(qian), hou)        # Xu: [factor, ]
            else:
                Xu = np.dot(np.linalg.inv(qian), hou)
            X[u, :] = Xu
            update_app_bar.update()
        update_app_bar.close()

        XtX = np.dot(X.T, X)                             # XtX: [factor, factor]

        update_lib_bar = tqdm(desc='updating lib vector...', leave=True, total=size_lib)
        for i in range(size_lib):
            Ci = C[:, i]                                 # Ci: [size_app, ]
            Pi = relation[:, i]
            hou = Ci * Pi
            hou = np.dot(X.T, hou)                              # hou: [factor, ]
            Ni = Y[maxPI[:, i], :].T                            # Ni: [factor, top_k]
            WiNormal = maxVI[:, i]                              # WiNormal: [top_k, ]
            Wi = WiNormal / np.sum(WiNormal)                    # Wi: [top_k, ]
            hou = hou + args.alpha * np.dot(Ni, Wi)             # hou: [factor, ]
            Ci = Ci - 1
            qian = X.T                                          # qian: [factor, size_app]
            for j in range(size_app):
                qian[:, j] = qian[:, j] * Ci[j]
            qian = np.dot(qian, X)                              # qian: [factor, factor]
            qian = qian + XtX
            qian = qian + args.lmda + args.alpha
            qian = np.nan_to_num(qian)
            condition_num = np.linalg.cond(qian)
            if condition_num > 1e10:
                Yi = np.dot(np.linalg.pinv(qian), hou)          # Yi: [factor, ]
            else:
                Yi = np.dot(np.linalg.inv(qian), hou)
            Y[i, :] = Yi
            update_lib_bar.update()
        update_lib_bar.close()
        print('epoch%d  [%.3fs]' % (epoch, (time() - epoch_st)))

    del XtX, YtY, Cu, Ci, Pi, Pu, \
        C, maxVU, maxVI, maxPU, maxPI, hou, qian

    prediction = np.dot(X, Y.T)                                    # prediction: [size_app, size_lib]

    utility.utils.ensure_dir(args.rec_output + 'fold%s_rmv%s/' % (fold_rmv[0], fold_rmv[1]))
    np.savetxt(
        fname=args.rec_output + 'fold%s_rmv%s/X_%s_%s.txt' % (fold_rmv[0], fold_rmv[1], fold_rmv[0], fold_rmv[1]),
        X=X,
        fmt='%.5f')
    np.savetxt(
        fname=args.rec_output + 'fold%s_rmv%s/Y_%s_%s.txt' % (fold_rmv[0], fold_rmv[1], fold_rmv[0], fold_rmv[1]),
        X=Y,
        fmt='%.5f')

    del X, Y

    for u in range(size_app):
        pre_u = prediction[u, :]
        pre_u[np.argwhere(relation[u, :] == 1)] = 0
        xiabiao = np.argsort(pre_u)[::-1].astype(np.uint16)
        position[u, :10] = xiabiao[:10]

    del xiabiao, pre_u

    # 保存预测结果
    # 保存一下X, Y的结果

    np.savetxt(fname=args.rec_output + 'fold%s_rmv%s/prediction_%s_%s.txt' % (fold_rmv[0], fold_rmv[1], fold_rmv[0], fold_rmv[1]),
               X=position,
               fmt='%d')


if __name__ == '__main__':
    # 训练结束后保存预测结果
    st_time = time()
    train(args.epochs)
    print('training completed.   [%.3fm]' % ((time() - st_time)/60))
    # test()
