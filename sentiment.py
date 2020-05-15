import torch
from data_preprocess.divide_sentences import divide_sentences
from data_preprocess.remove_non_Chinese import remove_non_Chinese
from data_preprocess.simplify_Chinese import simplify_Chinese
from data_preprocess.xml2txt import xml2txt
from train_models.SentimentAnalysis import SentimentAnalysis
from comments_dataset.divide_dataset import divide_dataset
import random
from config.Config import Config
import os
from train_models.train_word2vec import train_word2vec
from comments_dataset.divide_dataset import SentenceDistiller


def data_preprocess(dataset_path: str):
    xml2txt(file_path=dataset_path)
    simplify_Chinese()
    divide_sentences()
    remove_non_Chinese()

    random.seed(11)
    divide_dataset(
        csv_path='comments_dataset/waimai_10k.csv',
        stop_words_path='comments_dataset/stop_words/all_stop_words.txt'
    )


def word2vec_train_new_dataset(
        config_path: str,
        dataset_path: str,
        model_save_path: str,
        pretrain_model_path: str = '',
        epochs: int = -1
):
    assert os.path.exists(dataset_path), 'dataset path is invalid'
    cfg = Config.from_json(config_path)
    train_word2vec(cfg=cfg, input_file_name=dataset_path,
                   pretrain_model_path=pretrain_model_path,
                   model_save_path=model_save_path,
                   epoch=epochs)


def sentiment_rank(config_path: str, sentence: str) -> float:
    """
    :param config_path: configuration file path
    :param sentence: sentence to deal with
    :return: sentiment score, ranging in [0, 1]. The bigger the score is, the more positive sentiment is
    """
    # print(sentence)
    config = Config.from_json(config_path)
    sentiment_analysis = SentimentAnalysis(cfg=config)


    state_dict = torch.load(os.path.join(config.checkpoint_folder, 'optimal.pth'), map_location=config.device)
    sentiment_analysis.load_state_dict(state_dict)

    sentence_distiller = SentenceDistiller(
        stop_words_path=os.path.join('comments_dataset', 'stop_words', 'all_stop_words.txt'))

    distilled_sentence = sentence_distiller(sentence)
    print(distilled_sentence)

    return float(sentiment_analysis(distilled_sentence.split()))
