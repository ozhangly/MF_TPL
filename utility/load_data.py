import os
import re
import json
import utility.config
import numpy as np

args = utility.config.args


def load_relation_mat(train_file_path: str) -> np.ndarray:
    rmv_fold = re.findall('[0-9]', args.training_dataset)
    relation_file_name = 'relation_%s_%s.txt' % (rmv_fold[0], rmv_fold[1])
    if os.path.exists(args.relation_path + relation_file_name):
        relation = np.loadtxt(args.relation + relation_file_name, dtype=np.int8)
        return relation

    size_apk: int = 0
    size_lib: int = 0
    train_fp = open(file=train_file_path, mode='r')
    for line in train_fp.readlines():
        obj = json.loads(line.strip('\n'))
        app_id = obj['app_id']
        tpl_list = obj['tpl_list']
        size_apk = max(size_apk, app_id)
        size_lib = max(max(tpl_list), size_lib)

    relation = np.zeros(shape=(size_apk+1, size_lib+1), dtype=np.int8)

    train_fp.seek(0)
    for line in train_fp.readlines():
        obj = json.loads(line.strip('\n'))
        app_id = obj['app_id']
        tpl_list = obj['tpl_list']
        for tpl in tpl_list:
            relation[app_id, tpl] = 1

    np.savetxt(fname=args.relation_path + relation_file_name, X=relation, fmt='%d')

    return relation
