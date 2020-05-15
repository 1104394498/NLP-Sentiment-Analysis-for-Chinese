import csv
import random
import os
import jieba
import re
from typing import Tuple


class SentenceDistiller:
    def __init__(self, stop_words_path: str, cn_reg: str = '^[\u4e00-\u9fa5]+$'):
        self.cn_reg = cn_reg
        self.stop_words = [line.strip() for line in open(stop_words_path).readlines()]

    def __call__(self, sentence: str) -> str:
        print(f'sentence: {sentence}')
        sentence = sentence.strip()
        words_list = jieba.cut(sentence.split('\n')[0].replace(' ', ''))
        print(words_list)
        words_list_new = []

        for word in words_list:
            if re.search(self.cn_reg, word) and word not in self.stop_words:
                words_list_new.append(word)
        return ' '.join(words_list_new)


def divide_dataset(csv_path: str, stop_words_path: str, train_ratio: float = 0.7,
                   output_path: str = 'comments_dataset'):
    stream = open(csv_path, 'r')
    reader = csv.reader(stream)
    all_data = [line for line in reader]
    all_data = all_data[1:]
    random.shuffle(all_data)

    boundary = int(train_ratio * len(all_data))
    train_data = all_data[:boundary]
    test_data = all_data[boundary:]

    sentence_distiller = SentenceDistiller(stop_words_path=stop_words_path)

    f_txt = open(os.path.join(output_path, 'train.txt'), 'w')
    with open(os.path.join(output_path, 'train.csv'), 'w') as f:
        writer = csv.writer(f)
        for line in train_data:
            line[1] = sentence_distiller(line[1])
            writer.writerow(line)
            print(line[1], file=f_txt)
    f_txt.close()

    f_txt = open(os.path.join(output_path, 'test.txt'), 'w')
    with open(os.path.join(output_path, 'test.csv'), 'w') as f:
        writer = csv.writer(f)
        for line in test_data:
            line[1] = sentence_distiller(line[1])
            writer.writerow(line)
            print(line[1], file=f_txt)
    stream.close()
    f_txt.close()


if __name__ == '__main__':
    random.seed(11)
    csv_path = 'comments_dataset/waimai_10k.csv'
    divide_dataset(csv_path, 'comments_dataset/stop_words/all_stop_words.txt')
