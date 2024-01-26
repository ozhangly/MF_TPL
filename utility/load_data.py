import os
import re
import json
import config
import numpy as np


args = config.args


def load_relation_mat(train_file_path: str) -> np.ndarray:

    rmv_fold = re.findall('[0-9]', args.training_dataset)
    if os.path.exists(train_file_path):
        relation = np.loadtxt()

    size_apk: int = 0
    size_lib: int = 0
    train_fp = open(file=train_file_path, mode='r')
    for line in train_fp.readlines():
        obj = json.loads(line.strip('\n'))
        app_id = obj['app_id']
        tpl_list = obj['tpl_list']
        size_apk = max(size_apk, app_id)
        size_lib = max(max(tpl_list), size_lib)

    relation = np.zeros(shape=(size_apk, size_lib))

    for line in train_fp.readlines():
        obj = json.loads(line.strip('\n'))
        app_id = obj['app_id']
        tpl_list = obj['tpl_list']
        for tpl in tpl_list:
            relation[app_id, tpl] = 1

    return relation
