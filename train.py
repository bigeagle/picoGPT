import argparse
from loguru import logger
from rich.logging import RichHandler

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from dataset import TinyShakespeare, ChineseLiterature
from model import BigramLangModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--embed-size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}", "level": 'INFO'}])

    logger.info(f"Batch size: {args.batch_size}, block size: {args.block_size}")

    # ds = TinyShakespeare()
    ds = ChineseLiterature()
    blm = BigramLangModel(
        vocab_size=ds.vocab_size(),
        block_size=args.block_size,
        embedding_size=args.embed_size,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    if torch.cuda.is_available():
        blm = blm.cuda()

    generated_tokens = blm.sample(max_size=128, start_token=ds.tokenize(" "))
    generated_text = ds.textify(generated_tokens.cpu().numpy())
    print(generated_text)

    opt = torch.optim.AdamW(blm.parameters(), lr=args.lr)

    for step in range(100001):
        opt.zero_grad()
        blm.train()

        inputs, targets = ds.get_batch(batch_size=args.batch_size, block_size=args.block_size)
        outputs: torch.Tensor = blm(inputs)  # shape: (batch_size, block_size, vocab_size)

        loss = F.cross_entropy(outputs.view(-1, ds.vocab_size()), targets.view(-1))

        loss.backward()
        opt.step()

        if step % 100 == 0:
            blm.eval()
            inputs, targets = ds.get_batch(batch_size=args.batch_size, block_size=args.block_size, split="val")
            outputs: torch.Tensor = blm(inputs)  # shape: (batch_size, block_size, vocab_size)
            vloss = F.cross_entropy(outputs.view(-1, ds.vocab_size()), targets.view(-1))

            logger.info(f"Step: {step}, train loss: {loss.item():.4f}, val loss: {vloss.item():.4f}")

        if step % 2000 == 0:
            blm.save("model.pt")

    generated_tokens = blm.sample(max_size=128, start_token=ds.tokenize(" "))
    generated_text = ds.textify(generated_tokens.cpu().numpy())
    print(generated_text)

    blm.save("model.pt")


if __name__ == "__main__":
    main()