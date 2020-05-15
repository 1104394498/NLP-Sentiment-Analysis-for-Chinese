import torch.nn as nn
import gensim
from config.Config import Config
import torch
from typing import List
from gensim.models.word2vec import LineSentence
import torch.nn.functional as F


class SentimentAnalysis(nn.Module):
    def __init__(self, cfg: Config):
        super(SentimentAnalysis, self).__init__()

        assert cfg.RNN_type == 'LSTM' or cfg.RNN_type == 'GRU', f'invalid RNN type: {cfg.RNN_type}'
        """
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(cfg.word2vec_file_path, binary=True,
                                                                         unicode_errors='ignore')
        """
        self.cfg = cfg
        self.word2vec_model = gensim.models.Word2Vec.load(self.cfg.word2vec_file_path)
        # self.embedding = nn.Embedding.from_pretrained(word2vec_weights)

        if self.cfg.RNN_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=self.cfg.embedding_dim,
                hidden_size=self.cfg.hidden_size,
                num_layers=self.cfg.num_layers
            ).to(self.cfg.device)
        else:
            self.RNN = nn.GRU(
                input_size=self.cfg.embedding_dim,
                hidden_size=self.cfg.hidden_size
            ).to(self.cfg.device)
        self.dense = nn.Linear(in_features=self.cfg.hidden_size,
                               out_features=1).to(self.cfg.device)
        self.activ = nn.Sigmoid().to(self.cfg.device)

    def train_word2vec_model(self, file_name: str, epochs: int):
        self.word2vec_model.train(LineSentence(file_name), epochs=epochs)

    def forward(self, sentence: List[str]):
        embeds = []
        for word in sentence:
            try:
                embeds.append(self.word2vec_model.wv[word])
            except:
                continue
        if len(embeds) == 0:
            # print(f"{' '.join(sentence)}: words not in dictionary")
            return torch.tensor(0.5, dtype=torch.float, requires_grad=True)
        embeds = torch.tensor(embeds, dtype=torch.float).to(self.cfg.device)
        RNN_out, _ = self.RNN(embeds.view(1, embeds.shape[0], -1))
        RNN_out = RNN_out.squeeze(0)
        RNN_out = torch.sum(RNN_out, dim=0) / RNN_out.shape[0]
        tag = self.dense(RNN_out)
        score = self.activ(tag)
        return score
