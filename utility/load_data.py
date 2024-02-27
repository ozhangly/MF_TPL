import os
import re
import json
import utility.config
import numpy as np

import utility.utils as utils
from typing import Dict, Tuple

args = utility.config.args


def load_relation_mat(train_file_path: str) -> np.ndarray:
    fold_rmv = re.findall('[0-9]', args.training_dataset)
    relation_file_name = 'relation_%s_%s.txt' % (fold_rmv[0], fold_rmv[1])
    # app的映射哈希表
    dict1_name = 'app_id2app_order_id.json'

    # lib的映射哈希表
    lib_dict1_name: str = 'lib_order_id2lib_id.json'

    if os.path.exists(args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + relation_file_name):
        relation = np.loadtxt(args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + relation_file_name, dtype=np.int8)
        return relation

    size_apk: int = 0
    size_lib: int = 0

    # 需要两个Dict, 映射appid和app的order id
    app_order_id2app_id: Dict = {}
    app_id2app_order_id: Dict = {}
    # 同上
    lib_order_id2lib_id: Dict = {}
    lib_id2lib_order_id: Dict = {}

    train_fp = open(file=train_file_path, mode='r')
    for line in train_fp.readlines():
        obj = json.loads(line.strip('\n'))
        app_id = obj['app_id']
        tpl_list = obj['tpl_list']
        app_order_id2app_id[size_apk] = app_id
        app_id2app_order_id[app_id] = size_apk
        size_apk += 1

        for tpl in tpl_list:                                # 或许能够避免之前的nan错误....
            if tpl not in lib_id2lib_order_id.keys():
                lib_order_id2lib_id[size_lib] = tpl
                lib_id2lib_order_id[tpl] = size_lib
                size_lib += 1

    relation = np.zeros(shape=(size_apk, size_lib), dtype=np.int8)

    train_fp.seek(0)
    for line in train_fp.readlines():
        obj = json.loads(line.strip('\n'))
        app_id = obj['app_id']
        tpl_list = obj['tpl_list']
        app_orderid = app_id2app_order_id[app_id]
        for tpl in tpl_list:
            lib_order_id = lib_id2lib_order_id[tpl]
            relation[app_orderid, lib_order_id] = 1

    utils.ensure_dir(args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/')
    np.savetxt(fname=args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + relation_file_name, X=relation, fmt='%d')
    # 保存app_id to app order id的映射关系，后面会用到
    app_fp1 = open(file=args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + dict1_name, mode='w')
    ws1 = json.dumps(app_id2app_order_id) + '\n'
    app_fp1.write(ws1)
    app_fp1.close()

    del ws1, app_fp1

    lib_fp1 = open(file=args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + lib_dict1_name, mode='w')
    ws1 = json.dumps(lib_order_id2lib_id) + '\n'
    lib_fp1.write(ws1)
    lib_fp1.close()

    del ws1, lib_fp1

    return relation
