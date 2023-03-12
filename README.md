# PicoGPT

A extremely simple toy example of a transformer-based language model.

The model and method is based on Andrew Karpathy's awsome youtube video: [Let’s build GPT: from scratch, in code, spelled out][KarpathyYTB].


## Quick Start

Requirements:
```text
python >= 3.7
pytorch
numpy
rich
loguru
```

Training a model:
```bash
python3 train.py \
	--lr=1e-3 \
	--batch-size=32 \
	--block-size=128 \   # contex block size
	--embed-size=512 \   # embedding size
	--depth=4 \          # number of transformer layers
	--num-heads=4 \      # head-size (width) of each transformer layer
	--dropout=0.1
```

Traning can converge on an `RTX2080Ti` in about 15 minutes. Run this cmd for an interactive demo 
```bash
python3 chat.py
```

The default training dataset is Chinese classical literatures "水浒传" and "红楼梦", which can be easily changed to anything you like.


## Acknowledgements

Thank you Andrew Karpathy for your excellent [youtube video][KarpathyYTB] and the [nanoGPT][NanoGPT] project.

[KarpathyYTB]: https://www.youtube.com/watch?v=kCc8FmEb1nY
[NanoGPT]: https://github.com/karpathy/nanoGPT
