import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os
from config.Config import Config
import gensim


def train_word2vec(cfg: Config,
                   input_file_name: str = os.path.join('data', 'wiki.txt'),
                   pretrain_model_path: str = '',
                   model_save_path: str = '',
                   epoch: int = -1):
    print(f'dataset path: {input_file_name}, train_word2vec starts...')

    if len(model_save_path) == 0:
        model_save_path = cfg.word2vec_file_path

    if not os.path.exists(pretrain_model_path):
        model = Word2Vec(LineSentence(input_file_name),
                         size=cfg.embedding_dim,
                         window=cfg.window_size,
                         min_count=cfg.min_count,
                         workers=multiprocessing.cpu_count())
    else:
        model = gensim.models.Word2Vec.load(cfg.word2vec_file_path)
        model.train(LineSentence(input_file_name), epochs=model.iter if epoch <= 0 else epoch,
                    total_examples=model.corpus_count)

    model.save(model_save_path)


if __name__ == '__main__':
    config = Config.from_json(os.path.join('config', 'config.json'))
    train_word2vec(cfg=config)
