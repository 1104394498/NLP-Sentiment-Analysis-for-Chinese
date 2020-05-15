import torch
from typing import List, Dict
from train_models.SentimentAnalysis import SentimentAnalysis
from config.Config import Config
import os
import json
import csv
from tqdm import tqdm
import argparse


def prepare_sequence(seq: List[str], to_ix: Dict) -> torch.Tensor:
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


"""
def get_idx_dict(all_data: list, output_path: str = os.path.join('comments_dataset', 'word_idx.json'),
                 refresh: bool = False) -> dict:
    if not refresh and os.path.exists(output_path):
        with open(output_path, 'r') as f:
            idx_dict = json.load(f)
        return idx_dict
    idx_dict = {}
    for _, comment in all_data:
        for word in comment:
            if word not in idx_dict:
                idx_dict[word] = len(idx_dict)
    with open(output_path, 'w') as f:
        json.dump(idx_dict, f)
    return idx_dict
"""


class SentimentTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.training_data = []
        for tag, sentence in csv.reader(open(self.cfg.training_csv_path, 'r')):
            sentence_list = sentence.split()
            self.training_data.append((tag, sentence_list))

        self.test_data = []
        for tag, sentence in csv.reader(open(self.cfg.test_csv_path, 'r')):
            sentence_list = sentence.split()
            self.test_data.append((tag, sentence_list))
        all_data = []
        all_data.extend(self.training_data)
        all_data.extend(self.test_data)

        # self.idx_dict = get_idx_dict(all_data=all_data)

        self.model = SentimentAnalysis(cfg=self.cfg).to(self.cfg.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.cfg.lr,
                                          weight_decay=self.cfg.weight_decay)

    def train(self):
        opt_eval_loss = None
        for epoch in range(self.cfg.epoch):
            epoch_loss = 0.0
            self.model.train()
            for tag, sentence in tqdm(self.training_data):
                self.model.zero_grad()
                # sentence_tensor = prepare_sequence(sentence, self.idx_dict).to(self.cfg.device)
                pred_tag = self.model(sentence)
                # print(float(tag))
                # print(pred_tag)
                loss = torch.abs(pred_tag - float(tag))
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss)

            eval_loss = self.eval()
            print(
                f'EPOCH: {epoch}, avg train loss: {epoch_loss / len(self.training_data):.3f}, '
                f'avg eval loss: {eval_loss / len(self.test_data):.3f}')

            if opt_eval_loss is None or opt_eval_loss > eval_loss:
                print('** Optimal Model **')
                opt_eval_loss = eval_loss
                torch.save(self.model.state_dict(),
                           os.path.join(self.cfg.checkpoint_folder, 'optimal.pth'))
            torch.save(self.model.state_dict(), os.path.join(self.cfg.checkpoint_folder, f'epoch{epoch}.pth'))

    def eval(self) -> float:
        eval_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            self.model.eval()
            for tag, sentence in tqdm(self.test_data):
                # sentence_tensor = prepare_sequence(sentence, self.idx_dict).to(self.cfg.device)
                pred_tag = self.model(sentence)
                loss = torch.abs(pred_tag - float(tag))
                eval_loss += float(loss)
        return eval_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="config file path")
    args = parser.parse_args()

    config = Config.from_json(args.config)
    trainer = SentimentTrainer(cfg=config)
    trainer.train()
