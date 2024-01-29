'''
一、重新生成一下训练数据
'''

import json
import random

rmv_num = [1, 3, 5]
fold_num = [0, 1, 2, 3, 4]
for remove in rmv_num:
    for fold in fold_num:
        train_file = './training dataset/train_GREC_%d_%d.json' % (fold, remove)
        origin_train_fp = open(file='./train dataset/train_GREC_%d.json' % fold, mode='r')
        origin_test_fp  = open(file='./testing dataset/testing_%d_removed_num_%d.json' % (fold, remove), mode='r')
        with open(file=train_file, mode='w') as write_fp:
            # 在训练数据中选择几个remove进行移除，如果移除的数量大于现有的，那么就剩下两个，余下的全移除
            for line in origin_train_fp.readlines():
                origin_train_obj = json.loads(line.strip('\n'))
                tpl_list = origin_train_obj['tpl_list']
                # 对tpl_list进行随机移除
                if len(tpl_list) - remove <= 2:
                    while len(tpl_list) > 2:
                        sample = random.choice(tpl_list)
                        tpl_list.remove(sample)
                else:
                    sampled = random.sample(population=tpl_list, k=remove)
                    for sample in sampled:
                        tpl_list.remove(sample)
                origin_train_obj['tpl_list'] = tpl_list
                write_fp.write(json.dumps(origin_train_obj) + '\n')

            for line in origin_test_fp.readlines():
                origin_test_obj = json.loads(line.strip('\n'))
                test_obj = {
                    'app_id': origin_test_obj['app_id'],
                    'tpl_list': origin_test_obj['tpl_list']
                }
                write_fp.write(json.dumps(test_obj) + '\n')
