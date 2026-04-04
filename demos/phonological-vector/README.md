---
title: Phonological Vector
emoji: 🚀
colorFrom: pink
colorTo: pink
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
pinned: false
license: mit
---

# Phonological vector-based speech editing

Interactive demo for **[b]=[d]−[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic** ([arXiv:2602.18899](https://arxiv.org/abs/2602.18899)).

You can load an audio file, pick a time span and a learned phonological vector within WavLM representation, and hear how adding that vector changes the resynthesized audio, alongside spectrograms for before and after.

| Resource | Link |
|----------|------|
| Full codebase | [github.com/juice500ml/phonetic-arithmetic](https://github.com/juice500ml/phonetic-arithmetic) |
| Example audio / alignments | [LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1W) (TIMIT single-utterance sample from LDC) |

## Phonological vectors

The UI exposes three vector families (for TIMIT and VoxAngeles):

| Preset | Idea |
|--------|------|
| **Original** | Directions from the paper’s setup. |
| **Unconstrained** | Center pooling only; no separate consonant/vowel subspaces. |
| **Extended** | Unconstrained pooling, with positive and negative poles modeled as separate vectors. |

## Run locally

From this directory (`demos/phonological-vector`):

```bash
pip install -r requirements.txt
GRADIO_TEMP_DIR=$PWD/.gradio_tmp python app.py
```

Gradio will start a local URL; paths assume the working directory is the folder that contains `examples/` and `app.py`.

## Reproducing phonological vectors

Run from the **repository root** (`phonetic-arithmetic`), after you have the feature pickles and `dump_vectors.py` wired to your data. Replace `timit` with `voxangeles` if you want the other corpus.

```bash
dataset=timit

python3 dump_vectors.py \
  --feat-path feats/timit-wavlm-large-24-featslice.pkl \
  --output-path demos/phonological-vector/examples/original-${dataset}.pkl \
  --vector-type original --vector phn

python3 dump_vectors.py \
  --feat-path feats/timit-wavlm-large-24-center-featslice.pkl \
  --output-path demos/phonological-vector/examples/unconstrained-${dataset}.pkl \
  --vector-type full --vector phn

python3 dump_vectors.py \
  --feat-path feats/timit-wavlm-large-24-center-featslice.pkl \
  --output-path demos/phonological-vector/examples/extended-${dataset}.pkl \
  --vector-type extended --vector phn
```

Feature paths above match the naming used in this project; adjust `--feat-path` if your files differ.
