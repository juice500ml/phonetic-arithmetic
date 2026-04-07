import pickle
from pathlib import Path

import librosa
import numpy as np
import gradio as gr

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from specplotter import SpecPlotter

from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch


def cos_sim(XA, XB):
    XA_norm = XA / np.linalg.norm(XA, axis=1, keepdims=True)
    XB_norm = XB / np.linalg.norm(XB, axis=1, keepdims=True)
    return (XA_norm @ XB_norm.T)


def _read_pkl(path):
    with open(path, "rb") as f:
        vectors = pickle.load(f)["vectors"]
    feats = [key.split()[0] for key in vectors.keys() if "(0)" in key]
    return {
        feat: {
            loc: vectors.get(f"{feat} ({loc})")
            for loc in ["-4", "-3", "-2", "-1", "0", "+1", "+2", "+3", "+4"]
        }
        for feat in feats
    }

def _read_alignment(fname):
    data = []
    with open(fname, "r") as f:
        for line in f:
            start, end, text = line.strip().split()
            data.append({
                "start": int(start),
                "end": int(end),
                "text": text,
            })
    return data

print("Loading model...")
processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
ssl = AutoModel.from_pretrained("microsoft/wavlm-large")
print("Model loaded!")

print("Loading vectors...")
PHON_VECTORS = {
    "TIMIT (original)": _read_pkl("examples/original-timit.pkl"),
    "TIMIT (unconstrained)": _read_pkl("examples/unconstrained-timit.pkl"),
    "TIMIT (extended)": _read_pkl("examples/extended-timit.pkl"),
    "VoxAngeles (original)": _read_pkl("examples/original-voxangeles.pkl"),
    "VoxAngeles (unconstrained)": _read_pkl("examples/unconstrained-voxangeles.pkl"),
    "VoxAngeles (extended)": _read_pkl("examples/extended-voxangeles.pkl"),
}
DEFAULT_KEY = next(iter(PHON_VECTORS.keys()))
print("Vectors loaded!")

EXAMPLE_AUDIO = Path("examples/LDC93S1.wav")
EXAMPLE_PHN = _read_alignment("examples/LDC93S1.phn")
with open("examples/LDC93S1.pkl", "rb") as f:
    EXAMPLE_FEATS = pickle.load(f)


def run_orthogonal_subspace(path, vector_type, features, context_size, similarity_range):
    audio, _ = librosa.load(path, sr=16000, mono=True)
    if Path(path).name == EXAMPLE_AUDIO.name:
        feats = EXAMPLE_FEATS
        alignments = EXAMPLE_PHN
    else:
        inputs = processor(
            raw_speech=[audio],
            sampling_rate=16000,
            padding=False,
            return_tensors="pt",
        )
        out = ssl(**inputs)
        feats = out.last_hidden_state[0].detach().numpy()
        alignments = []

    keys, vectors = [], []
    for f in features:
        for i in ["-4", "-3", "-2", "-1", "0", "+1", "+2", "+3", "+4"][4-context_size:5+context_size]:
            if (PHON_VECTORS[vector_type][f] is not None) and (PHON_VECTORS[vector_type][f][i] is not None):
                keys.append(f"{f} ({i})")
                vectors.append(PHON_VECTORS[vector_type][f][i])
    vectors = np.stack(vectors)
    sims = cos_sim(vectors, feats)

    fig, ax = plt.subplots(1, figsize=(10, 2 + len(keys) // 5), constrained_layout=True)
    ax.axis("off")

    gs = fig.add_gridspec(
        nrows=1 + len(keys), ncols=1,
        height_ratios=[3] + [0.2] * len(keys)  # spectrogram taller than heatmaps
    )

    # Spectrogram plotting
    ax_spec = fig.add_subplot(gs[0, 0])

    sp = SpecPlotter()
    sp.plot_spectrogram(audio, ax=ax_spec, show_annotation=False)
    ax_spec.get_xaxis().set_visible(False)

    for row in alignments:
        start, end, label = row["start"] / 16000, row["end"] / 16000, row["text"]

        ax_spec.axvline(start, color="black", linestyle="-", alpha=0.7)
        ax_spec.axvline(end, color="black", linestyle="-", alpha=0.7)
        ax_spec.add_patch(
            plt.Rectangle(
                (start, 7),
                end - start,
                1,
                color="black",
                alpha=0.4,
                clip_on=False
            )
        )
        ax_spec.text(
            (start + end) / 2,
            7.5,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=9
        )

    x0, x1 = ax_spec.get_xlim()
    ims = []
    axes_hm = []
    for i, (hm, lab) in enumerate(zip(sims, keys), start=1):
        ax = fig.add_subplot(gs[i, 0], sharex=ax_spec)
        axes_hm.append(ax)

        hm = np.asarray(hm)
        if hm.ndim == 1:
            hm = hm[None, :]  # make it (1, T) so it looks like a single-row heatmap

        # Use extent so the heatmap x-axis is in seconds (aligned with spectrogram)
        im = ax.imshow(
            hm,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[x0, x1, 0, 1],
            vmin=-similarity_range,
            vmax=+similarity_range,
            cmap=plt.cm.PuOr,
        )
        ims.append(im)

        for row in alignments:
            start, end, label = row["start"] / 16000, row["end"] / 16000, row["text"]
            ax.axvline(start, color="black", linestyle="-", alpha=0.7)
            ax.axvline(end, color="black", linestyle="-", alpha=0.7)

        ax.set_yticks([])
        ax.tick_params(axis='x', length=0)

        feat, loc = lab.split()
        if loc == "(0)":
            if context_size == 0:
                label = f"[+{feat}]"
            else:
                label = f"[+{feat}] 0"
        else:
            label = loc[1:-1]
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontweight="bold" if loc == "(0)" else "normal")
        ax.yaxis.set_label_coords(-0.02, 0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # Only show x tick labels on the bottom-most axis
    plt.setp(ax_spec.get_xticklabels(), visible=False)
    for ax in axes_hm[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    axes_hm[-1].set_xlabel("Time [s]")

    ax_spec.set_xlim(x0, x1)

    cbar = fig.colorbar(ims[-1], ax=axes_hm, pad=0.01, fraction=0.03)
    cbar.set_label("Cosine similarity")

    return fig


with gr.Blocks(title="Orthogonal Subspace Demo") as demo:
    with gr.Row():
        gr.Markdown("""
## 🎙️ Orthogonal Subspace Demo

Demonstration for the paper [Self-Supervised Speech Models Encode Phonetic Context via Position-dependent Orthogonal Subspaces](https://arxiv.org/abs/2603.12642).
This demo reproduces Figure 10: cosine similarity between frame-level S3M representations and position-dependent phonological vectors over time, illustrating how each relative phone position occupies a distinct orthogonal subspace.

Upload, record, or use the example audio, configure the parameters, and click **Run**.
""")

    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(
                label="Input Audio",
                type="filepath",
                sources=["upload", "microphone"],
                recording=True,
                value=str(EXAMPLE_AUDIO),
            )
            gr.Markdown("""
### Parameters
- **Vector extraction method**: How phonological vectors are estimated from S3M representations. Different options correspond to different training dataset/calculating the vectors.
- **Phonological features**: Which phonological features to include in the plot. Deselect features to reduce clutter or isolate a single dimension of contrast.
- **Context size**: Number of relative phone positions. 0 = vectors from current phone only; k = vectors from relative positions −k through +k. Larger values reveal how far phonological features extend beyond current (or immediately adjacent) phones.
- **Cosine similarity range**: Upper bound of the cosine similarity (default +/- 0.4). Adjust to zoom in on fine-grained differences or accommodate low-similarity outputs.
""")

        with gr.Column(scale=1):
            vector_dropdown = gr.Dropdown(
                label="Vector extraction method",
                choices=list(PHON_VECTORS.keys()),
                value=DEFAULT_KEY,
                interactive=True,
            )
            feature_checkbox = gr.CheckboxGroup(
                choices=list(PHON_VECTORS[DEFAULT_KEY].keys()),
                value=list(PHON_VECTORS[DEFAULT_KEY].keys()),
                label="Phonological features",
                show_select_all=True,
                interactive=True,
            )
            context_size_slider = gr.Slider(label="Context size", value=2, minimum=0, maximum=4, step=1, interactive=True)
            similarity_slider = gr.Slider(label="Cosine similarity range", value=0.4, minimum=0.1, maximum=1.0, step=0.01, interactive=True)
            run_btn = gr.Button("▶ Run", variant="primary", scale=1)

    with gr.Row():
        plot = gr.Plot(
            label="Output Spectrogram and Phonological Representations",
            show_label=False,
        )

    # Connectors
    vector_dropdown.change(
        fn=lambda key: gr.CheckboxGroup(
            choices=list(PHON_VECTORS[key].keys()),
            value=list(PHON_VECTORS[key].keys()),
        ),
        inputs=vector_dropdown,
        outputs=feature_checkbox,
    )
    run_btn.click(
        fn=run_orthogonal_subspace,
        inputs=[audio, vector_dropdown, feature_checkbox, context_size_slider, similarity_slider],
        outputs=plot,
    )

if __name__ == "__main__":
    demo.launch()