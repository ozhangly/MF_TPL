import re
import utility.utils
import utility.sim_computed

import numpy as np

from tqdm import tqdm
from time import time
from test_mf import test
from utility import load_data, config

args = config.args


def train(epochs) -> None:
    rmv_fold = re.findall('[0-9]', args.training_dataset)

    train_file = args.training_path + args.training_dataset
    print('load data...')
    load_st = time()
    relation = load_data.load_relation_mat(train_file_path=train_file)
    print('load data completed. [%.2fs]' % (time() - load_st))

    # 计算app相似度和lib相似度
    print('compute app similarity...')
    app_sim_st = time()
    utility.sim_computed.app_sim_computed(relation)
    print('app similarity compute completed. [%.2f]' % (time() - app_sim_st))
    print('compute lib similarity...')
    lib_sim_st = time()
    utility.sim_computed.lib_sim_computed(relation)
    print('lib similarity compute completed. [%.2f]' % (time() - lib_sim_st))

    (size_app, size_lib) = relation.shape

    log_weight = args.weight / (np.log(np.sum(relation, axis=0) + 1) + 1)

    maxVI = np.loadtxt(fname=args.similarity_path + '%d_%d/maxVI.txt')
    maxPI = np.loadtxt(fname=args.similarity_path + '%d_%d/maxPI.txt')
    maxPU = np.loadtxt(fname=args.similarity_path + '%d_%d/maxPU.txt')
    maxVU = np.loadtxt(fname=args.similarity_path + '%d_%d/maxVU.txt')

    C = np.zeros(shape=(size_app, size_lib))
    np.random.seed(int(time()))
    X = np.random.standard_normal(size=(size_app, args.factor))
    Y = np.random.standard_normal(size=(size_lib, args.factor))

    for i in range(size_lib):
        C[:, i] = 1 + log_weight[i]*relation[:, i]

    P = relation
    position = np.zeros(shape=(size_app, 10))

    # 准备工作都已做好
    # 用进度条还是用什么方法？统计每个epoch的用时，然后输出当前的epoch
    for epoch in range(epochs):
        epoch_st = time()
        YtY = np.dot(Y.T, Y)                           # YtY: [factor, factor]

        update_app_bar = tqdm(desc='update app vector...', total=size_app, leave=True)
        for u in range(size_app):
            Cu = C[u, :].T                               # Cu: [size_lib, ]
            Pu = P[u, :].T                               # Pu: [size_lib, ]
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
            Xu = np.dot(qian.I, hou)                     # Xu: [factor, ]
            X[u, :] = Xu
            update_app_bar.update()
        update_app_bar.close()

        XtX = np.dot(X.T, X)                             # XtX: [factor, factor]

        update_lib_bar = tqdm(desc='update lib vector...', leave=True, total=size_lib)
        for i in range(size_lib):
            Ci = C[:, i]                                 # Ci: [size_app, ]
            Pi = P[:, i]
            hou = Ci * Pi
            hou = np.dot(X.T, hou)                        # hou: [factor, ]
            Ni = Y[maxPI[:, i], :].T                     # Ni: [factor, top_k]
            WiNormal = maxVI[:, i]                       # WiNormal: [top_k, ]
            Wi = WiNormal / np.sum(WiNormal)             # Wi: [top_k, ]
            hou = hou + args.alpha * Wi * Ni             # hou: [factor, ]
            Ci = Ci - 1
            qian = X.T                                   # qian: [factor, size_app]
            for j in range(size_app):
                qian[:, j] = qian[:, j] * Ci[j]
            qian = np.dot(qian, X)                       # qian: [factor, factor]
            qian = qian + XtX
            qian = qian + args.lmda + args.alpha
            Yi = np.dot(qian.I, hou)                     # Yi: [factor, ]
            Y[i, :] = Yi
            update_lib_bar.update()
        update_lib_bar.close()
        print('>>>>>>>>>>>epoch%d  [%.3fs]<<<<<<<<<<<' % (epoch, (time() - epoch_st)))
    prediction = np.dot(X, Y.T)                          # prediction: [size_app, size_lib]

    del XtX, YtY, Cu, Ci, X, Y, P, Pi, Pu, \
        C, maxVU, maxVI, maxPU, maxPI

    for u in range(size_app):
        pre_u = prediction[u, :]
        pre_u[np.argwhere(relation[u, :] == 1)] = 0
        xiabiao = np.argsort(pre_u)[::-1]
        position[u, :10] = xiabiao[:10]

    del xiabiao, pre_u

    # 保存预测结果
    utility.utils.ensure_dir(args.rec_output + 'rmv%d_fold%d/' % (rmv_fold[0], rmv_fold[1]))
    np.savetxt(fname=args.rec_output + 'rmv%d_fold%d/prediction_%d_%d.txt' % (rmv_fold[0], rmv_fold[1], rmv_fold[0], rmv_fold[1]),
               X=prediction,
               fmt='%d')


if __name__ == '__main__':
    # 训练结束后保存预测结果
    train(args.epochs)
    print('>>>>>>>>>>>>>>>>>>>>>进行测试<<<<<<<<<<<<<<<<<<<<<')
    test()
