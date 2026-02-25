import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from specplotter import SpecPlotter
from transformers import Wav2Vec2FeatureExtractor, AutoModel

from plot_everything import _calculate_contextual_vectors, filter_phones, cdist


if __name__ == "__main__":
    dataset = "timit"
    model = "wavlm-large-24"
    phones = filter_phones(pd.read_csv(f"feats/{dataset}.csv"), cutoff=0)

    df = pd.read_pickle(f"feats/timit-{model}-center-featslice.pkl")
    avg_phon_vectors = _calculate_contextual_vectors(df[df.split == "train"], phones, ["l_3", "l_2", "l_1", "ipa", "r_1", "r_2", "r_3"])

    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    ssl = AutoModel.from_pretrained("microsoft/wavlm-large")

    path = df[df.audio_path.str.contains("TRAIN/DR1/FDML0/SX339.WAV")].audio_path.iloc[0]
    start, end = 0.616187, 1.017500
    _df = df[(df.audio_path == path) & (~df.ipa.isna()) & (df["min"] >= start) & (df["max"] <= end)].copy()
    _df["min"] = _df["min"] - start
    _df["max"] = _df["max"] - start

    audio, _ = librosa.load(path, sr=16000, mono=True)
    inputs = processor(
        raw_speech=[audio],
        sampling_rate=16000,
        padding=False,
        return_tensors="pt",
    )
    out = ssl(**inputs)
    full_feats = out.last_hidden_state[0].detach().numpy()
    feats = full_feats[int(start * 16000 / 320):int(end * 16000 / 320)]

    keys = [f"{f} ({i})" for f in ["lo", "strid"] for i in ["-1", "0", "+1"]]
    vectors = np.stack([avg_phon_vectors[k] for k in keys])
    sims = 1.0 - cdist(vectors, feats, metric="cosine")

    # Plotting code
    fig = plt.figure(figsize=(6, 2))

    gs = fig.add_gridspec(
        nrows=len(keys)+2, ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.3, 1, 1, 1, 1.3, 1, 1, 1],
        hspace=0.01,
    )

    ax_spec = fig.add_subplot(gs[:, 0])

    sp = SpecPlotter()
    sp.plot_spectrogram(audio[int(start*16000):int(end*16000)], ax=ax_spec, show_annotation=False)

    for i, row in enumerate(_df.itertuples()):
        s, e, label = row.min, row.max, row.ipa
        if i != 0:
            ax_spec.axvline(s, color="black", alpha=0.7)
        ax_spec.add_patch(
            plt.Rectangle((s, 7), e - s, 1,
                        color="black", alpha=0.4, clip_on=False)
        )
        ax_spec.text((s + e) / 2, 7.5, label,
                    ha="center", va="center",
                    color="white", fontsize=10)

    x0, x1 = ax_spec.get_xlim()

    ims = []
    axes_hm = []

    for i, (hm, lab) in enumerate(zip(sims, keys)):
        if i < 3:
            ax = fig.add_subplot(gs[i+1, 1], sharex=ax_spec)
        else:
            ax = fig.add_subplot(gs[i+2, 1], sharex=ax_spec)
        axes_hm.append(ax)

        hm = np.asarray(hm)
        if hm.ndim == 1:
            hm = hm[None, :]

        im = ax.imshow(
            hm,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[x0, x1, 0, 1],
            vmin=-0.4,
            vmax=+0.4,
            cmap=plt.cm.PuOr,
        )
        ims.append(im)
        ax.margins(0)
        ax.set_yticks([])
        ax.set_ylabel("")

        for j, row in enumerate(_df.itertuples()):
            ax.axvline(row.min, color="black", alpha=0.7)

        feat, _ = lab.split()
        ax.set_ylabel(["Prev", "Curr", "Next"][i%3], rotation=0, ha="right", va="center")
        ax.yaxis.set_label_coords(-0.02, 0.5)

        if i < 5:
            ax.tick_params(axis="x", length=0)
        if i % 3 != 0:
            ax.spines["top"].set_visible(False)

    for ax in axes_hm[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    axes_hm[-1].set_xlabel("Time (seconds)")

    ax_spec.set_xlim(x0, x1)

    axes_hm[0].set_title("Which phone is [+lo]?",  fontsize=10, pad=6)
    axes_hm[3].set_title("Which phone is [+strid]?", fontsize=10, pad=6)

    cbar = fig.colorbar(ims[-1], ax=axes_hm, pad=0.02, fraction=0.06)
    cbar.set_label("Cos. sim.")

    plt.savefig(f"plots/contextual/fig1-small.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close()


    # Bigger plot
    path = df[df.audio_path.str.contains("TRAIN/DR1/FCJF0/SA1.WAV")].audio_path.iloc[0]
    _df = df[(df.audio_path == path)].copy()

    audio, _ = librosa.load(path, sr=16000, mono=True)
    inputs = processor(
        raw_speech=[audio],
        sampling_rate=16000,
        padding=False,
        return_tensors="pt",
    )
    out = ssl(**inputs)
    full_feats = out.last_hidden_state[0].detach().numpy()
    feats = full_feats

    keys = [f"{f} ({i})" for f in ["hi", "lo", "back", "round", "nas", "son", "strid", "voi"] for i in ["-2", "-1", "0", "+1", "+2"]]
    vectors = np.stack([avg_phon_vectors[k] for k in keys])
    sims = 1.0 - cdist(vectors, feats, metric="cosine")

    fig, ax = plt.subplots(1, figsize=(12, 8), constrained_layout=True)
    ax.axis("off")

    gs = fig.add_gridspec(
        nrows=1 + len(keys), ncols=1,
        height_ratios=[3] + [0.2] * len(keys)  # spectrogram taller than heatmaps
    )
    ax_spec = fig.add_subplot(gs[0, 0])

    sp = SpecPlotter()
    sp.plot_spectrogram(audio, ax=ax_spec, show_annotation=False)
    ax_spec.get_xaxis().set_visible(False)

    _df = df[df.audio_path == path]
    for i, row in enumerate(_df.itertuples()):
        start, end, label = row.min, row.max, row.ipa
        if type(row.ipa) == float:
            if row.timit_phn == "aw":
                label = "aʊ"
            else:
                continue
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
            vmin=-0.4,
            vmax=+0.4,
            cmap=plt.cm.PuOr,
        )
        ims.append(im)

        for j, row in enumerate(_df.itertuples()):
            start, end, label = row.min, row.max, row.ipa
            if type(row.ipa) == float and row.timit_phn != "aw":
                continue

            ax.axvline(start, color="black", linestyle="-", alpha=0.7)
            ax.axvline(end, color="black", linestyle="-", alpha=0.7)

        ax.set_yticks([])
        ax.tick_params(axis='x', length=0)

        feat, loc = lab.split()
        if loc == "(0)":
            label = f"[+{feat}] 0"
        else:
            label = loc[1:-1]
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
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
    cbar.set_label("Cos. sim.")

    y_start = axes_hm[0].get_position().y0 + 0.15
    y_end = axes_hm[0].get_position().y0 - 0.64

    for y in np.linspace(y_start, y_end, 9)[1:-1]:
        fig.add_artist(mlines.Line2D(
            [0.00, 0.91], [y, y],
            transform=fig.transFigure,
            color="black", lw=0.5, alpha=0.5
        ))

    plt.savefig(f"plots/contextual/fig1-big.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close()
