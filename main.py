import argparse
from config.Config import Config
from sentiment import word2vec_train_new_dataset, sentiment_rank

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str, help='configuration file path')
    # parser.add_argument('-p', "--preprocess", type=bool, default=False, help='preprocess dataset or not')
    parser.add_argument('-t', "--train", type=bool, default=False, help='Train word2vec model or not')

    parser.add_argument('-s', "--sentence", type=str, default='', help='Sentence to deal with')

    args = parser.parse_args()

    # data_preprocess(args.datasetPath)
    config = Config.from_json(args.config)
    if args.train:
        for dataset_path in config.sentence_datasets:
            word2vec_train_new_dataset(
                config_path=args.config,
                dataset_path=dataset_path,
                model_save_path=config.word2vec_file_path,
                pretrain_model_path=config.word2vec_file_path
            )

    if len(args.sentence) > 0:
        sentiment_score = sentiment_rank(config_path=args.config,
                                         sentence=args.sentence)
        print(f'sentiment score: {sentiment_score: .3f}')
