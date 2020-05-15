import os
import json


class Config:
    def __init__(self):
        self.embedding_dim = 128
        self.window_size = 5
        self.min_count = 5
        self.device = 'cpu'

        self.RNN_type = 'LSTM'
        self.hidden_size = 64

        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.num_layers = 1

        self.word2vec_file_path = ''

        self.epoch = 100

        self.stop_words_path = ''
        self.training_csv_path = ''
        self.test_csv_path = ''

        self.checkpoint_folder = ''

        self.sentence_datasets = []

    @staticmethod
    def from_json(json_path: str):
        assert os.path.exists(json_path)
        config = Config()
        with open(json_path, 'r') as stream:
            cfg = json.load(stream)
        for k in cfg:
            v = cfg[k]
            assert hasattr(config, k), f'invalid attr: {k}'
            if isinstance(v, list):
                v = tuple(v)
            setattr(config, k, v)
        return config
