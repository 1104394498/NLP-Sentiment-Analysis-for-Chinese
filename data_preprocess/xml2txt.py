from gensim.corpora import WikiCorpus
from tqdm import tqdm
import os


def xml2txt(file_path: str = os.path.join('..', 'zhwiki-latest-pages-articles.xml.bz2'),
            output_file_name: str = os.path.join('data', 'wiki.cn.txt')):
    print('xml2txt start')
    input_file = WikiCorpus(file_path, lemmatize=False, dictionary={})
    output_file = open(output_file_name, 'w', encoding="utf-8")

    for text in tqdm(input_file.get_texts()):
        output_file.write(' '.join(text) + '\n')
    output_file.close()
