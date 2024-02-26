import os
import re
import json
import utility.config
import numpy as np

import utility.utils as utils
from typing import Dict, Tuple

args = utility.config.args


def load_relation_mat(train_file_path: str) -> Tuple[np.ndarray, Dict, Dict]:
    fold_rmv = re.findall('[0-9]', args.training_dataset)
    relation_file_name = 'relation_%s_%s.txt' % (fold_rmv[0], fold_rmv[1])
    # 还有两个字典需要保存
    dict1_name = 'app_order_id2app_id_%s_%s.json' % (fold_rmv[0], fold_rmv[1])
    dict2_name = 'app_id2app_order_id_%s_%s.json' % (fold_rmv[0], fold_rmv[1])
    if os.path.exists(args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + relation_file_name):
        relation = np.loadtxt(args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + relation_file_name, dtype=np.int8)
        app_order_id2app_id = json.load(fp=open(file=args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + dict1_name, mode='r'))
        app_id2app_order_id = json.load(fp=open(file=args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + dict2_name, mode='r'))
        app_order_id2app_id = utils.process_dict_key(app_order_id2app_id)
        app_id2app_order_id = utils.process_dict_key(app_id2app_order_id)

        return relation, app_order_id2app_id, app_id2app_order_id               # 加载出来的键都是str类型，需要处理成int类型再使用

    size_apk: int = 0
    size_lib: int = 0

    # 需要两个Dict, 分别从旧的映射到新的, 和新 -> 旧的
    app_order_id2app_id: Dict = {}
    app_id2app_order_id: Dict = {}

    train_fp = open(file=train_file_path, mode='r')
    for line in train_fp.readlines():
        obj = json.loads(line.strip('\n'))
        app_id = obj['app_id']
        tpl_list = obj['tpl_list']
        # size_apk = max(size_apk, app_id)
        app_order_id2app_id[size_apk] = app_id
        app_id2app_order_id[app_id] = size_apk
        size_apk += 1
        size_lib = max(max(tpl_list), size_lib)

    relation = np.zeros(shape=(size_apk, size_lib), dtype=np.int8)

    train_fp.seek(0)
    for line in train_fp.readlines():
        obj = json.loads(line.strip('\n'))
        app_id = obj['app_id']
        tpl_list = obj['tpl_list']
        app_orderid = app_id2app_order_id[app_id]
        for tpl in tpl_list:
            relation[app_orderid, tpl-1] = 1

    utils.ensure_dir(args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/')
    np.savetxt(fname=args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + relation_file_name, X=relation, fmt='%d')
    # 两个字典也保存
    fp1 = open(file=args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + dict1_name, mode='w')
    ws1 = json.dumps(app_order_id2app_id) + '\n'
    fp1.write(ws1)
    fp1.close()
    fp2 = open(file=args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + dict2_name, mode='w')
    ws2 = json.dumps(app_id2app_order_id) + '\n'
    fp2.write(ws2)
    fp2.close()

    return relation, app_order_id2app_id, app_id2app_order_id
