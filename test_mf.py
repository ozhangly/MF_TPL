import re
import json
import utility.utils
import utility.config
import utility.metrics

import numpy as np

from tqdm import tqdm
from typing import Dict, List

args = utility.config.args
ks = [5, 10]


def get_performance(user_post, r, auc) -> Dict:
    precision, recall, ndcg, \
    map, fone, mrr = [], [], [], [], [], []

    for k in ks:
        precision.append(utility.metrics.precision_at_k(r, k))
        recall.append(utility.metrics.recall_at_k(r, k, len(user_post)))
        ndcg.append(utility.metrics.ndcg_at_k(r, k))
        fone.append(utility.metrics.F1(utility.metrics.precision_at_k(r, k), utility.metrics.recall_at_k(r, k, len(user_post)))),
        mrr.append(utility.metrics.mrr_at_k(r, k)),
        map.append(utility.metrics.average_precision(r, k))

    return {
        'recall': np.array(recall), 'precision': np.array(precision), 'map': np.array(map),
        'fone': np.array(fone), 'mrr': np.array(mrr), 'ndcg': np.array(ndcg)
    }


def test_one_user(user_pos, user_pre) -> Dict:
    r: List = []
    for i in user_pre:
        if i in user_pos:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return get_performance(user_pos, r, auc)


def test() -> None:
    test_num: int = 0
    fold_rmv = re.findall('[0-9]', args.testing_dataset)
    utility.utils.ensure_dir(args.rec_output + 'fold%s_rmv%s/' % (fold_rmv[0], fold_rmv[1]))
    recommend_res_fp = open(file=args.rec_output + 'fold%s_rmv%s/test_MF_%s_%s.json' % (fold_rmv[0], fold_rmv[1], fold_rmv[0], fold_rmv[1]), mode='w')
    recommend_metric_fp = open(file=args.rec_output + 'fold%s_rmv%s/metric_output.csv' % (fold_rmv[0], fold_rmv[1]), mode='w')

    res = {
        'recall': np.zeros(shape=(len(ks),)), 'precision': np.zeros(shape=(len(ks),)), 'ndcg': np.zeros(shape=(len(ks),)),
        'fone': np.zeros(shape=(len(ks),)), 'mrr': np.zeros(shape=(len(ks),)), 'map': np.zeros(shape=(len(ks),))
    }

    prediction = np.loadtxt(args.rec_output + 'fold%s_rmv%s/prediction_%s_%s.txt' % (fold_rmv[0], fold_rmv[1], fold_rmv[0], fold_rmv[1]))

    dict2 = json.load(fp=open(file=args.relation_path + 'app_id2app_order_id_%s_%s.json' % (fold_rmv[0], fold_rmv[1])))
    app_id2app_order_id = utility.utils.process_dict_key(dict2)

    del dict2

    with open(file=args.testing_path + args.testing_dataset, mode='r') as fp:
        for line in tqdm(fp.readlines(), desc='test progress...', leave=True):
            test_obj = json.loads(line.strip('\n'))
            app_id   = test_obj['app_id']
            pos_list = test_obj['removed_tpl_list']
            app_order_id = app_id2app_order_id[app_id]
            pre_list = prediction[app_order_id, :].astype(np.uint16).tolist()

            write_data = {
                'app_id': app_id,
                'recommend_tpl': pre_list,
                'removed_tpls': pos_list
            }
            recommend_res_fp.write(json.dumps(write_data) + '\n')

            test_num += 1
            one_user_res = test_one_user(pos_list, pre_list)
            res['recall'] += one_user_res['recall']
            res['map'] += one_user_res['map']
            res['mrr'] += one_user_res['mrr']
            res['ndcg'] += one_user_res['ndcg']
            res['fone'] += one_user_res['fone']
            res['precision'] += one_user_res['precision']

    recommend_res_fp.close()

    res['recall'] /= test_num
    res['precision'] /= test_num
    res['fone'] /= test_num
    res['ndcg'] /= test_num
    res['map'] /= test_num
    res['mrr'] /= test_num

    write_res = '%s\n%s\n%s\n%s\n%s\n%s\n' % (
        ','.join(['%.5f'% r for r in res['recall']]),
        ','.join(['%.5f' % r for r in res['precision']]),
        ','.join(['%.5f' % r for r in res['fone']]),
        ','.join(['%.5f' % r for r in res['mrr']]),
        ','.join(['%.5f' % r for r in res['map']]),
        ','.join(['%.5f' % r for r in res['ndcg']])
    )

    recommend_metric_fp.write(write_res)


if __name__ == '__main__':
    test()
