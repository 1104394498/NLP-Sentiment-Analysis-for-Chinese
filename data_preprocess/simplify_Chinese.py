import zhconv
from tqdm import tqdm
import os


def simplify_Chinese(input_file_name: str = os.path.join('data', 'wiki.cn.txt'),
                     output_file_name: str = os.path.join('data', 'wiki.cn.simple.txt')):
    print('simplify_Chinese start...')
    input_file = open(input_file_name, 'r', encoding='utf-8')
    output_file = open(output_file_name, 'w', encoding='utf-8')

    lines = input_file.readlines()

    for line in tqdm(lines):
        output_file.write(zhconv.convert(line, 'zh-hans'))

    input_file.close()
    output_file.close()
