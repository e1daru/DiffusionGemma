# DiffusionGemma

Run `google/diffusiongemma-26B-A4B-it` using the correct `DiffusionGemmaForBlockDiffusion` class from HuggingFace Transformers ≥ 4.53.

## Option 1 — GitHub Codespace (Edupack, free)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/e1daru/DiffusionGemma)

The devcontainer requests a **16-core / 64 GB RAM** machine. The model loads in float16 on CPU (~52 GB), so you need this size or larger.

1. Click the badge → GitHub will use the devcontainer config to provision the right machine size
2. Wait ~3 min for container build + `pip install`
3. Open `notebooks/diffusiongemma_colab.ipynb` in VS Code and run cells

CPU inference is slower than GPU (~5–15 tok/s) but works fine for experimentation.

## Option 2 — Google Colab Pro / Kaggle

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/e1daru/DiffusionGemma/blob/main/notebooks/diffusiongemma_colab.ipynb)

- **Colab Pro**: Runtime → L4 (24 GB) or A100 (40 GB)
- **Kaggle**: select 2×T4 accelerator (30 GB total)
- Free Colab T4 (15 GB) is below the 18 GB minimum — won't work

## Key API

```python
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor

model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    "google/diffusiongemma-26B-A4B-it",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("google/diffusiongemma-26B-A4B-it")

inputs = processor.apply_chat_template(
    [{"role": "user", "content": "Your prompt here"}],
    tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=256, num_diffusion_steps=16)
print(processor.decode(output[0], skip_special_tokens=True))
```

`num_diffusion_steps`: 8 = fast, 16 = balanced, 24 = best quality.
