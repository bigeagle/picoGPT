#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Sequence


class Dataset:
    pass


class TextDataset(Dataset):

    _text_files = None

    def __init__(self):

        self.chars = ''
        for text_file in self._text_files:
            with open(text_file) as f:
                self.chars = self.chars + f.read()

        self.vocabulary = sorted(set(self.chars))
        self._char2idx = {c: i for i, c in enumerate(self.vocabulary)}
        self._idx2char = {i: c for i, c in enumerate(self.vocabulary)}

        self.dataset_tokens = torch.tensor([self.tokenize(c) for c in self.chars])

        N = len(self.dataset_tokens)
        N_t = N // 10 * 9

        self._dataset = {
            "train": self.dataset_tokens[:N_t],
            "val": self.dataset_tokens[N_t:]
        }

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        logger.info("Text Files: ")
        for text_file in self._text_files:
            logger.info(f"    - {text_file}")

        logger.info(f"Loaded {len(self.chars)} characters, vocabulary size: {len(self.vocabulary)}")

    def get_batch(self, batch_size: int = 4, block_size: int = 8, split="train"):
        tokens = self._dataset[split]
        idx = torch.randint(len(tokens) - block_size, (batch_size, ))

        inputs = torch.stack([tokens[i: i + block_size] for i in idx])
        targets = torch.stack([tokens[i + 1: i + block_size + 1] for i in idx])
        return inputs.to(self._device), targets.to(self._device)

    def vocab_size(self):
        return len(self.vocabulary)

    def tokenize(self, c: str):
        return self._char2idx[c]

    def charify(self, i: int):
        return self._idx2char[i]

    def textify(self, tokens: Sequence[int]):
        return ''.join(self._idx2char[int(i)] for i in tokens)


class TinyShakespeare(TextDataset):

    _text_files = [Path(__file__).with_name("data") / "tiny-shakespeare.txt"]


class ChineseLiterature(TextDataset):

    _text_files = [
        Path(__file__).with_name("data") / "shuihu.txt",
        Path(__file__).with_name("data") / "hongloumeng.txt"
    ]


if __name__ == "__main__":

    ds = TinyShakespeare()

    xs, ys = ds.get_batch()

    for b in range(xs.shape[0]):
        print(ds.textify(xs[b]))
        print(ds.textify(ys[b]))
        print()

    ds = ChineseLiterature()

    xs, ys = ds.get_batch()

    for b in range(xs.shape[0]):
        print(ds.textify(xs[b]))
        print(ds.textify(ys[b]))
        print()
