import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', nargs='?', default=24, type=int)

    parser.add_argument('--lmda', nargs='?', default=0.001, type=float)

    parser.add_argument('--factor', nargs='?', default=20, type=int)

    parser.add_argument('--epochs', nargs='?', default=10, type=int)

    parser.add_argument('--top_k', nargs='?', default=6, type=int)

    parser.add_argument('--alpha', nargs='?', default=0.5, type=float)

    # 实验的路径
    parser.add_argument('--relation_path', nargs='?', default='./relation/', type=str)

    parser.add_argument('--similarity_path', nargs='?', default='./similarity/', type=str)

    parser.add_argument('--rec_output', nargs='?', default='./output/', type=str)

    parser.add_argument('--training_path', nargs='?', default='./training dataset/', type=str)

    parser.add_argument('--testing_path', nargs='?', default='./testing dataset/', type=str)

    parser.add_argument('--training_dataset', nargs='?', default='train_GREC_0_1.json', type=str)

    parser.add_argument('--testing_dataset', nargs='?', default='testing_0_removed_num_1.json', type=str)

    return parser.parse_args()
