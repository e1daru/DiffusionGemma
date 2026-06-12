# DiffusionGemma — NVIDIA Free API

Run `google/diffusiongemma-26b-a4b-it` for free via [NVIDIA build.nvidia.com](https://build.nvidia.com) — no GPU required.

## Quickstart in GitHub Codespace

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/e1daru/DiffusionGemma)

1. Click the badge above (or open the repo in Codespace from the GitHub UI)
2. Wait ~2 min for the container to build and `pip install` to finish
3. Get a free API key at **[build.nvidia.com](https://build.nvidia.com)** → sign in → top-right avatar → "API Key"
4. In the Codespace terminal:
   ```bash
   cp .env.example .env
   # then edit .env and paste your key
   ```
5. Try the CLI:
   ```bash
   python scripts/inference.py --prompt "What is a diffusion language model?"
   ```
6. Or open `notebooks/diffusiongemma_nvidia.ipynb` in VS Code and run cells interactively

## CLI options

```
python scripts/inference.py \
  --prompt "Your prompt here" \
  --max-tokens 512 \
  --temperature 0.7 \
  --stream
```

## Free tier limits

NVIDIA's free tier gives you 1000 API credits per month. DiffusionGemma-26B uses roughly 1 credit per ~100 output tokens. More than enough for experimentation.
