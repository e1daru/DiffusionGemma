# DiffusionGemma

Run `google/diffusiongemma-26B-A4B-it` for free on Google Colab (T4 GPU).

## Quickstart — Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/e1daru/DiffusionGemma/blob/main/notebooks/diffusiongemma_colab.ipynb)

1. Click the badge above
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run all cells (Ctrl+F9)

The first cell checks your GPU, the second installs dependencies (~2 min), then the model loads from HuggingFace in 4-bit quantization to fit in 15 GB VRAM.

## What's in the notebook

| Cell | What it does |
|---|---|
| GPU check | Confirms you have a T4 |
| Install | `transformers`, `bitsandbytes`, `accelerate` |
| Load model | 4-bit NF4 quantized, auto device map |
| Basic generation | Single prompt → response |
| Multi-turn chat | Conversation with history |
| Interactive widget | Sliders for temperature / max tokens |
| Batch comparison | Same prompt at 4 different temperatures |

## Run in GitHub Codespace (no GPU — CPU only)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/e1daru/DiffusionGemma)

CPU-only inference is very slow for a 26B model — only use this if you want to edit and test the code, not run real inference. Use Colab for actual generation.
