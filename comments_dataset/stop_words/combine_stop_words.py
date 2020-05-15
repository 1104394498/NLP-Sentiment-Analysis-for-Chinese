import os


def combine_stop_words(stop_words_file_list: list, stop_words_folder: str, output_file: str = 'all_stop_words.txt'):
    all_stop_words = []
    for file_name in stop_words_file_list:
        path = os.path.join(stop_words_folder, file_name)
        with open(path, 'r') as f:
            for word in f.readlines():
                word = word.strip()
                if word not in all_stop_words:
                    all_stop_words.append(word)

    with open(os.path.join(stop_words_folder, output_file), 'w') as f:
        for word in all_stop_words:
            print(word, file=f)


if __name__ == '__main__':
    combine_stop_words(
        stop_words_file_list=['baidu_stopwords.txt', 'cn_stopwords.txt', 'hit_stopwords.txt', 'scu_stopwords.txt'],
        stop_words_folder='.',
        output_file='all_stop_words.txt'
    )
