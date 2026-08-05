# Data

Raw and processed data are not committed to this repository (see `.gitignore`)
— datasets are large and regenerable, and committing them bloats repo history
permanently, even if later deleted.

## Structure

- `raw/` — untouched, as-downloaded dataset files.
- `processed/` — output of preprocessing scripts in `src/data/` (tokenized
  captions, resized images, cached embeddings, etc).

## Reproducing

Instructions for downloading and preparing each dataset will be added here
as Task 1 (dataset analysis) is implemented, along with the exact script/
command used, so anyone cloning this repo can regenerate `raw/` and
`processed/` from scratch.
