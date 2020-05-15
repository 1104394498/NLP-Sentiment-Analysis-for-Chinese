# coding:utf-8
import re
import os
from tqdm import tqdm


def remove_non_Chinese(input_file_name: str = os.path.join('data', 'wiki.cn.simple.separate.txt'),
                       output_file_name: str = os.path.join('data', 'wiki.txt')):
    print('remove_non_Chinese...')
    input_file = open(input_file_name, 'r', encoding='utf-8')
    output_file = open(output_file_name, 'w', encoding='utf-8')

    lines = input_file.readlines()

    cn_reg = '^[\u4e00-\u9fa5]+$'

    for line in tqdm(lines):
        line_list = line.split('\n')[0].split(' ')
        line_list_new = []
        for word in line_list:
            if re.search(cn_reg, word):
                line_list_new.append(word)
        # print(line_list_new)
        output_file.write(' '.join(line_list_new) + '\n')

    input_file.close()
    output_file.close()
