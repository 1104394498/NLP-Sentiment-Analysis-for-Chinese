# coding:utf-8
import jieba
import os
from tqdm import tqdm


def divide_sentences(input_file_name: str = os.path.join('data', 'wiki.cn.simple.txt'),
                     output_file_name: str = os.path.join('data', 'wiki.cn.simple.separate.txt')):
    print('divide_sentences starts...')
    input_file = open(input_file_name, 'r', encoding='utf-8')
    output_file = open(output_file_name, 'w', encoding='utf-8')

    lines = input_file.readlines()

    for line in tqdm(lines):
        # jieba分词的结果是一个list，需要拼接，但是jieba把空格回车都当成一个字符处理
        output_file.write(' '.join(jieba.cut(line.split('\n')[0].replace(' ', ''))) + '\n')

    input_file.close()
    output_file.close()
