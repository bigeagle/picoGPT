import argparse
from loguru import logger

from rich import print
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Prompt

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from dataset import ChineseLiterature
from model import BigramLangModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reply-size", type=int, default=128)
    args = parser.parse_args()

    pt = torch.load("model.pt")

    ds = ChineseLiterature()
    blm = BigramLangModel(**pt['hp'])
    if torch.cuda.is_available():
        blm = blm.cuda()

    blm.load_state_dict(pt['state'])
    blm.eval()

    while True:
        prompt = Prompt.ask(">>> ")
        if prompt == "exit":
            break

        generated_tokens = blm.sample(max_size=args.reply_size, start_token=[ds.tokenize(c) for c in prompt])
        generated_text = ds.textify(generated_tokens.cpu().numpy())
        generated_text = Text(generated_text, style="green")
        print(Panel(generated_text, width=120, title="Reply"))


if __name__ == "__main__":
    main()