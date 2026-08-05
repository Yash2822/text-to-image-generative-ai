# Text-to-Image Generation: From CGAN to Fine-Tuned Diffusion

**Status:** 🚧 Scaffolding stage — core pipeline in progress.

An end-to-end generative vision-language project that builds up a
text-to-image system in stages: dataset analysis, text embeddings, a
conditional GAN, attention-augmented generation, and finally fine-tuning a
pretrained diffusion model (Stable Diffusion) on a custom dataset — with a
shared, reusable codebase across every stage rather than disconnected
scripts.

## Why this project

Most educational GAN/diffusion projects are single notebooks that generate
plausible-looking outputs once and are never touched again. This repository
is structured to mirror how these models are actually built and evaluated
in practice: config-driven experiments, reproducible training runs,
quantitative evaluation (not just "the images look okay"), and a codebase
that stays coherent as it grows from a CGAN baseline to a fine-tuned
diffusion pipeline.

## Roadmap

| Stage | Description | Status |
|---|---|---|
| 1 | Dataset analysis (COCO / Oxford-102 Flowers) | ⬜ Not started |
| 2 | Text preprocessing — tokenization & embeddings (Hugging Face) | ⬜ Not started |
| 3 | Conditional GAN (label/text-conditioned generation) | ⬜ Not started |
| 4 | Attention-augmented GAN (self-/cross-attention) | ⬜ Not started |
| 5 | Fine-tuning a pretrained diffusion model (LoRA / DreamBooth) | ⬜ Not started |
| 6 | End-to-end pipeline: text → tokenizer → embedding → generator → image | ⬜ Not started |

## Project structure

```
├── app/                # Gradio demo entrypoint (thin UI layer, no logic)
├── configs/            # YAML experiment configs (one per run)
├── data/                # raw/ and processed/ data (gitignored — see data/README.md)
├── src/
│   ├── inference/         # baseline pretrained Stable Diffusion wrapper
│   ├── data/               # dataset classes, preprocessing, tokenization
│   ├── models/              # generator, discriminator, attention blocks
│   ├── training/              # training loops, loss functions
│   ├── evaluation/             # FID, CLIP score, other metrics
│   └── utils/                   # config loading, logging, checkpointing
├── notebooks/           # exploration only — no production logic
├── experiments/          # checkpoints & run logs (gitignored)
├── outputs/                # generated images / sample grids (gitignored)
├── tests/                    # unit tests
└── docs/
    └── legacy/                # original provided files, kept for reference only
```

## Baseline: Stable Diffusion inference

The internship-provided project (a Colab notebook + script wrapping a
pretrained Stable Diffusion pipeline) has been refactored into
`src/inference/diffusion_pipeline.py`, with the Gradio demo moved to
`app/gradio_app.py`. Behavior is preserved and covered by tests in
`tests/test_diffusion_pipeline.py`, with one intentional fix documented
in `docs/legacy/README.md`. Run the demo with:

```bash
python -m app.gradio_app
```

Config is YAML, parsed into typed Python dataclasses (`src/utils/config.py`)
rather than passed around as raw dicts — see that file's docstring for why.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running tests

```bash
pytest tests/
```

## Experiment tracking

Weights & Biases is used for tracking training runs, starting from Task 3
(the CGAN baseline) once real training begins. Enable it by setting
`tracking.enabled: true` in a config file.

## License

MIT — see [LICENSE](LICENSE).

## Real-model baseline verification

The normal unit tests mock model weights so they stay fast and deterministic.
Before extending the project, run one real Stable Diffusion inference smoke test
on suitable hardware:

```bash
python scripts/smoke_test_sd.py
```

The default run fixes the prompt, seed, scheduler, resolution, inference steps,
and guidance scale, then records generation time, device, dtype, model ID, and
GPU memory information when CUDA is available. The generated image and JSON
metadata are written under `outputs/baseline/`. This gives later fine-tuning and
optimization work a reproducible inference baseline rather than a purely
qualitative reference.
