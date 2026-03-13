import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import panphon
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from pathlib import Path
import scipy.stats
from scipy.spatial.distance import cdist
from itertools import product
from collections import defaultdict

from estimate_similarity import filter_phones, get_quadruples, get_ci
from analyze_synth import _split_train_test, filter_phones, separate_phones

plt.style.use("tableau-colorblind10")


def _get_feature_sets(quadruples):
    ft = panphon.FeatureTable()
    features = np.array(ft.fts("a").names)

    feature_sets = {feature: [] for feature in features}
    _get_feat_arr = lambda p: np.array(ft.fts(p).numeric())

    feature_sets["total"] = list(quadruples)
    for a, b, c, d in quadruples:
        # a = b + (c - d)
        # a = c + (b - d)
        mask = (_get_feat_arr(c) != _get_feat_arr(d)) | (_get_feat_arr(b) != _get_feat_arr(d))
        for feature in features[mask]:
            feature_sets[feature].append((a, b, c, d))

    for key in [feature for feature, quads in feature_sets.items() if len(quads) == 0]:
        del feature_sets[key]

    return feature_sets


def _get_feature_count_sets(quadruples):
    ft = panphon.FeatureTable()
    features = np.array(ft.fts("a").names)
    _get_feat_arr = lambda p: np.array(ft.fts(p).numeric())

    feature_count_sets = {f"{i}": [] for i in range(1, 12)}
    for a, b, c, d in quadruples:
        count = max((_get_feat_arr(c) != _get_feat_arr(d)).sum(), (_get_feat_arr(b) != _get_feat_arr(d)).sum())
        feature_count_sets[str(count)].append((a, b, c, d))
    feature_count_sets["total"] = list(quadruples)

    return feature_count_sets


def _plot_individual_quadruplets(dataset):
    Path("plots/quadruplets").mkdir(parents=True, exist_ok=True)

    for model in tqdm(["hubert-large", "w2v2-large", "w2v2-phoneme", "w2v2-multipa", "xlsr-53", "wavlm-large"]):
        for slice in ["featslice", "audioslice"]:
            if not Path(f"feats/similarities-{dataset}-{model}-{slice}.pkl").exists():
                print(f"Similarities for {dataset}-{model}-{slice} do not exist, skipping.")
                continue

            similarities = pickle.load(open(f"feats/similarities-{dataset}-{model}-{slice}.pkl", "rb"))
            for quadruplet, similarity in similarities.items():
                plt.figure(figsize=(4, 4))
                for i, metric in enumerate(("arithmetic", "different", "same")):
                    y = np.array([m[metric] for m in similarity])
                    x = list(range(len(y)))
                    plt.plot(y[:, 0], "o-", label=metric, c=f"C{i}")
                    plt.errorbar(x, y=y[:, 0], yerr=y[:, 0] - y[:, 1], elinewidth=1, capsize=2, c=f"C{i}")
                    plt.fill_between(x, y[:, 1], y[:, 2], alpha=0.3)
                plt.legend()
                plt.title(f"{model}-{slice} {quadruplet}")
                plt.savefig(f"plots/quadruplets/quadruplets-{'-'.join(quadruplet)}-{dataset}-{model}-{slice}.pdf")
                plt.close()


def _get_success_rates(similarities, feature_sets):
    successes = {}
    for feature, quadruplets in feature_sets.items():
        if len(quadruplets) > 0:
            num_layers = len(next(iter(similarities.values())))
            success_rate = np.zeros(num_layers)

            counts = 0
            for quadruplet, similarity in similarities.items():
                if quadruplet in quadruplets:
                    arith_lo = np.array([m["arithmetic"][1] for m in similarity])
                    arith_hi = np.array([m["arithmetic"][2] for m in similarity])
                    same_lo = np.array([m["same"][1] for m in similarity])
                    diff_hi = np.array([m["different"][2] for m in similarity])
                    success_rate += ((same_lo > arith_hi) & (arith_lo > diff_hi)).astype(int)
                    counts += 1
            successes[feature] = success_rate / counts
    return successes


def _get_spectral_baselines(dataset, feature_sets, prefix="", postfix=""):
    successes = []

    for slice_type in ("featslice", "audioslice"):
        for model_name in ("mfcc", "melspec"):
            fname = f"feats/similarities-{dataset}-{model_name}-{prefix}{slice_type}{postfix}.pkl"
            if not Path(fname).exists():
                print(f"Similarities for {fname} do not exist, skipping.")
                success = None
            else:
                success = _get_success_rates(pickle.load(open(fname, "rb")), feature_sets)
            successes.append(success)
    return tuple(successes)


def _plot_success_rates(dataset, feature_sets, name):
    Path("plots/success-rates").mkdir(parents=True, exist_ok=True)
    mfcc_success, melspec_success, mfcc_audio_success, melspec_audio_success = _get_spectral_baselines(dataset, feature_sets)
    for model in tqdm(["hubert-large", "w2v2-large", "w2v2-phoneme", "w2v2-multipa", "xlsr-53", "wavlm-large"]):
        for slice in ["featslice", "audioslice"]:
            fname = f"feats/similarities-{dataset}-{model}-{slice}.pkl"
            if not Path(fname).exists():
                print(f"Similarities for {fname} do not exist, skipping.")
                continue

            success = _get_success_rates(pickle.load(open(fname, "rb")), feature_sets)

            n_rows = (len(success) - 1) // 4 + 1
            fig, axes = plt.subplots(n_rows, 4, figsize=(12, n_rows * 3))
            axes = axes.flatten()
            for plot_index, (feature, success_rate) in enumerate(success.items()):
                axes[plot_index].plot(success_rate, "o-")
                axes[plot_index].set_title(f"{feature} ({len(feature_sets[feature])})")
                axes[plot_index].set_ylim(-0.05, 1.05)
                axes[plot_index].axhline(mfcc_success[feature][0], ls="--", c="C0")
                axes[plot_index].axhline(melspec_success[feature][0], ls="--", c="C1")
                axes[plot_index].axhline(mfcc_audio_success[feature][0], ls="-", c="C0")
                axes[plot_index].axhline(melspec_audio_success[feature][0], ls="-", c="C1")
            plt.tight_layout()
            plt.savefig(f"plots/success-rates/success-rates-{name}-{dataset}-{model}-{slice}.pdf")
            plt.close()

def _plot_success_rates_with_target_col(dataset, feature_sets, model_name, pool, name):
    Path("plots/contextual").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(6, 2), constrained_layout=True)
    feature = "total"
    for plot_index, target_col in enumerate(("start_phone", "middle_phone", "end_phone")):
        fname = f"feats/similarities-{dataset}-{model_name}-{pool}-featslice-{target_col}.pkl"
        mfcc_success, melspec_success, mfcc_audio_success, melspec_audio_success = _get_spectral_baselines(dataset, feature_sets, prefix=f"{pool}-", postfix=f"-{target_col}")
        success = _get_success_rates(pickle.load(open(fname, "rb")), feature_sets)

        axes[plot_index].plot(success[feature], ".-", c="C3")
        axes[plot_index].set_title(f"{target_col}")
        axes[plot_index].set_ylim(-0.05, 1.05)

        l1 = axes[plot_index].axhline(mfcc_success[feature][0], ls="--", c="C0", label="MFCC")
        l2 = axes[plot_index].axhline(melspec_success[feature][0], ls="--", c="C1", label="MelSpec")
        legend_lines = [l1, l2]
        legend_labels = [l.get_label() for l in legend_lines]

        if plot_index % 3 == 0:
            axes[plot_index].set_ylabel("Success rate")
            axes[plot_index].legend(frameon=False, loc="upper left")
        axes[plot_index].set_xlabel("Layer index")

    fig.suptitle(f"{model_name} {pool}")
    plt.savefig(f"plots/contextual/{name}-{dataset}-{model_name}-{pool}.pdf")
    plt.close()

    fig_legend = plt.figure(figsize=(4, 1))
    fig_legend.legend(
        legend_lines,
        legend_labels,
        ncol=2,
        frameon=False,
        loc="center"
    )
    fig_legend.canvas.draw()
    plt.savefig("plots/contextual/model-comparison-legend.pdf", bbox_inches="tight", pad_inches=0.0)


def _plot_model_comparison(dataset, feature_sets, models, sliced, name, print_sliced=True, only_featslice_baselines=False, include_legend=True):
    Path("plots/main").mkdir(parents=True, exist_ok=True)
    mfcc_success, melspec_success, mfcc_audio_success, melspec_audio_success = _get_spectral_baselines(dataset, feature_sets)

    fig, axes = plt.subplots(1, 3, figsize=(6, 2), constrained_layout=True)
    axes = axes.flatten()
    feature = "total"
    for plot_index, model in enumerate(models):
        model_name = {
            "wavlm-large": "WavLM", "hubert-large": "HuBERT", "w2v2-large": "wav2vec 2.0",
            "w2v2-phoneme": "Wav2vec2Phoneme", "w2v2-multipa": "MultIPA", "xlsr-53": "XLSR-53",
        }[model]
        sliced_name = {"featslice": "feat", "audioslice": "audio"}[sliced]
        success = _get_success_rates(pickle.load(open(f"feats/similarities-{dataset}-{model}-{sliced}.pkl", "rb")), feature_sets)

        if model in ["wavlm-large", "hubert-large", "w2v2-large"]:
            print("success-rates", name, dataset, model, sliced, success["total"].max())

        axes[plot_index].plot(success["total"], ".-", c="C3")
        axes[plot_index].set_title(f"{model_name} ({sliced_name})" if print_sliced else f"{model_name}")
        axes[plot_index].set_ylim(-0.05, 1.05)

        l1 = axes[plot_index].axhline(mfcc_success[feature][0], ls="--", c="C0", label="MFCC (feat)" if not only_featslice_baselines else "MFCC")
        l2 = axes[plot_index].axhline(melspec_success[feature][0], ls="--", c="C1", label="MelSpec (feat)" if not only_featslice_baselines else "MelSpec")
        if not only_featslice_baselines:
            l3 = axes[plot_index].axhline(mfcc_audio_success[feature][0], ls="-",  c="C0", label="MFCC (audio)")
            l4 = axes[plot_index].axhline(melspec_audio_success[feature][0], ls="-",  c="C1", label="MelSpec (audio)")
        legend_lines = [l1, l2] if only_featslice_baselines else [l1, l2, l3, l4]
        legend_labels = [l.get_label() for l in legend_lines]

        print("success-rates MFCC (feat sliced) baseline", dataset, mfcc_success[feature][0])
        print("success-rates MelSpec (feat sliced) baseline", dataset, melspec_success[feature][0])
        print("success-rates MFCC (audio sliced) baseline", dataset, mfcc_audio_success[feature][0])
        print("success-rates MelSpec (audio sliced) baseline", dataset, melspec_audio_success[feature][0])

        if plot_index % 3 == 0:
            axes[plot_index].set_ylabel("Success rate")
            if only_featslice_baselines and include_legend:
                axes[plot_index].legend(frameon=False, loc="upper left")
        axes[plot_index].set_xlabel("Layer index")

    plt.savefig(f"plots/main/{name}-{dataset}.pdf")

    if (not only_featslice_baselines) and include_legend:
        fig_legend = plt.figure(figsize=(4, 1))
        fig_legend.legend(
            legend_lines,
            legend_labels,
            ncol=4,
            frameon=False,
            loc="center"
        )
        fig_legend.canvas.draw()
        plt.savefig("plots/main/model-comparison-legend.pdf", bbox_inches="tight", pad_inches=0.0)


def _get_consonant_vowel_feature_sets(feature_sets):
    ft = panphon.FeatureTable()
    feature_sets_cons = {
        feature: [q for q in quads if all(ft.fts(p).strings()[2] == "+" for p in q)]
        for feature, quads in feature_sets.items()
    }
    feature_sets_vowel = {
        feature: [q for q in quads if all(ft.fts(p).strings()[2] == "-" for p in q)]
        for feature, quads in feature_sets.items()
    }
    return feature_sets_cons, feature_sets_vowel


def _plot_consonant_vowel_comparison(dataset, f_sets, f_c_sets, f_v_sets, model="wavlm-large", sliced="featslice", include_legend=True, path="main"):
    Path(f"plots/{path}").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(6, 2), constrained_layout=True)
    axes = axes.flatten()
    feature = "total"
    model_name = {
        "wavlm-large": "WavLM", "hubert-large": "HuBERT", "w2v2-large": "wav2vec 2.0",
        "w2v2-phoneme": "Wav2vec2Phoneme", "w2v2-multipa": "MultIPA", "xlsr-53": "XLSR-53",
    }[model]

    for plot_index, (feature_name, feature_sets) in enumerate(zip((f"Total", "Consonants", "Vowels"), [f_sets, f_c_sets, f_v_sets])):
        if path == "contextual":
            pool, _, target_col = sliced.split("-")
            mfcc_success, melspec_success, mfcc_audio_success, melspec_audio_success = _get_spectral_baselines(dataset, feature_sets, prefix=f"{pool}-", postfix=f"-{target_col}")
        else:
            mfcc_success, melspec_success, mfcc_audio_success, melspec_audio_success = _get_spectral_baselines(dataset, feature_sets)

        success = _get_success_rates(pickle.load(open(f"feats/similarities-{dataset}-{model}-{sliced}.pkl", "rb")), feature_sets)
        axes[plot_index].plot(success["total"], ".-", c="C3")
        axes[plot_index].set_title(f"{feature_name} ({len(feature_sets[feature])})")
        axes[plot_index].set_ylim(-0.05, 1.05)

        axes[plot_index].axhline(mfcc_success[feature][0], ls="--", c="C0", label="MFCC")
        axes[plot_index].axhline(melspec_success[feature][0], ls="--", c="C1", label="MelSpec")
        # axes[plot_index].axhline(mfcc_audio_success[feature][0], ls="-",  c="C0")
        # axes[plot_index].axhline(melspec_audio_success[feature][0], ls="-",  c="C1")
        if plot_index % 3 == 0:
            axes[plot_index].set_ylabel("Success rate")
            if include_legend:
                axes[plot_index].legend(frameon=False, loc="upper left")
        axes[plot_index].set_xlabel("Layer index")

    if path == "contextual":
        fig.suptitle(f"{model_name} {sliced}")
    plt.savefig(f"plots/{path}/phone-comparison-{dataset}-{model}-{sliced}.pdf")


def _get_avgcos(similarities, feature_sets):
    sims = {"same": [], "different": [], "arithmetic": []}

    for feature, quadruplets in feature_sets.items():
        if len(quadruplets) > 0:
            for quadruplet, similarity in similarities.items():
                if quadruplet in quadruplets:
                    for key in sims.keys():
                        sims[key].append(np.array([m[key][0] for m in similarity]))
    return {k: np.array(v) for k, v in sims.items()}


def _plot_cossim_comparison(dataset, feature_sets):
    Path("plots/main").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1, 4,
        figsize=(5, 2.5),
        gridspec_kw={"width_ratios": [2, 2, 1, 1]},
        constrained_layout=True
    )

    for i, model in enumerate(("wavlm-large-featslice", "w2v2-large-featslice", "mfcc-audioslice", "mfcc-audioslice")):
        ax = axes[i]
        sims = _get_avgcos(pickle.load(open(f"feats/similarities-{dataset}-{model}.pkl", "rb")), feature_sets)

        for key, color in zip(("same", "arithmetic", "different", ), ("C0", "C3", "C1", )):
            vals = sims[key]
            avgs = []
            for layer in range(vals.shape[1]):
                func = lambda seed: vals[seed][layer]
                avgs.append(get_ci(func=func, N=len(vals)))
            avgs = np.array(avgs)
            ax.plot(avgs[:, 0], ".-", c=color, label={"same": r"$\overline{\cos}^+$", "arithmetic": r"$\overline{\cos}$", "different": r"$\overline{\cos}^-$"}[key])
            ax.fill_between(range(vals.shape[1]), avgs[:, 1], avgs[:, 2], color=color, alpha=0.2)
            ax.errorbar(range(vals.shape[1]), avgs[:, 0], yerr=avgs[:, 2]-avgs[:, 1], fmt=".", color=color, capsize=2, zorder=3, elinewidth=1.2)

        if i == 0:
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks([0, 10, 20])
            ax.set_ylabel("Cosine similarity")
            ax.set_xlabel("Layer index")
            ax.set_title("WavLM")
            ax.legend(frameon=False)

        if i == 1:
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks([0, 10, 20])
            ax.set_yticks([])
            ax.set_xlabel("Layer index")
            ax.set_title("wav2vec 2.0")

        if i == 2:
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(left=False, bottom=False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("MFCC")

        if i == 3:
            ax.tick_params(bottom=False)
            ax.set_xticks([])

    x_full = 0
    ymin_zoom, ymax_zoom = axes[3].get_ylim()
    con_bottom = ConnectionPatch(
        xyA=(x_full, ymin_zoom),     coordsA="data",   axesA=axes[2],
        xyB=(0, 0),                  coordsB="axes fraction", axesB=axes[3],
        color="gray", lw=2, linestyle=":", alpha=0.5,
    )
    con_top = ConnectionPatch(
        xyA=(x_full, ymax_zoom),     coordsA="data",   axesA=axes[2],
        xyB=(0, 1),                  coordsB="axes fraction", axesB=axes[3],
        color="gray", lw=2, linestyle=":", alpha=0.5,
    )
    fig.add_artist(con_bottom)
    fig.add_artist(con_top)
    plt.savefig(f"plots/main/cossim-comparison-{dataset}.pdf")


def _plot_synth_scatter(dataset, model, targets, metrics):
    assert len(targets) == len(metrics)

    Path("plots/main").mkdir(parents=True, exist_ok=True)

    title_fmt = r"$\mathbf{r}_{<feat>} + \lambda \mathbf{v}_{<feat>}$"

    fig, axes = plt.subplots(
        2, len(targets),
        figsize=(16, 5), constrained_layout=True,
    )
    for j, (target, metric) in enumerate(zip(targets, metrics)):
        df = pd.read_csv(f"feats/{dataset}-{model}-synth-{target}.csv")
        cv, pp = target.split("-")

        for i in range(2):
            if i == 0:
                _df = df[df.is_positive_phone]
                title = title_fmt.replace("<feat>", pp)
            else:
                _df = df[~df.is_positive_phone]
                title = title_fmt.replace("<feat>", pp).replace("r}_{", "r}_{\\sim ")
                axes[i][j].sharey(axes[i-1][j])

            x = _df.weight
            y = _df[f"modified_{metric}"] - _df[f"original_{metric}"]

            x = x[~np.isnan(y)]
            y = y[~np.isnan(y)]
            stats = scipy.stats.spearmanr(x, y)

            if metric == "ZCR":
                y *= 100
                axes[i][j].set_ylabel(f"Δ {metric} (%)")
            elif metric == "HNR":
                axes[i][j].set_ylabel(f"Δ {metric} (dB)")
            else:
                y /= 1000
                axes[i][j].set_ylabel(f"Δ {metric} (kHz)")

            axes[i][j].set_xlabel(r"$\lambda$")
            axes[i][j].scatter(x, y, color="C0" if cv == "vowel" else "C1", s=0.2, rasterized=True)
            axes[i][j].set_title(f"{title}\n$\\rho=${stats.statistic:.3f}")

    plt.savefig(f"plots/main/synth-scatter-{dataset}-{model}.pdf")


def _plot_synth_density(dataset, model, targets, metrics):
    assert len(targets) == len(metrics)

    Path("plots/main").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, len(targets),
        figsize=(16, 4), constrained_layout=True,
    )
    for j, (target, metric) in enumerate(zip(targets, metrics)):
        df = pd.read_csv(f"feats/{dataset}-{model}-synth-{target}.csv")
        cv, pp = target.split("-")

        for i in range(2):
            if i == 0:
                _df = df[df.is_positive_phone]
            else:
                _df = df[~df.is_positive_phone]
                axes[i][j].sharey(axes[i-1][j])

            y = _df[f"synth_{metric}"] - _df[f"original_{metric}"]
            y = y[~np.isnan(y)]

            y_mod = _df[f"modified_{metric}"] - _df[f"original_{metric}"]
            y_mod = y_mod[~np.isnan(y_mod)]

            if metric == "ZCR":
                y *= 100
                y_mod *= 100
                title = f"Δ {metric} (%)"
            elif metric == "HNR":
                title = f"Δ {metric} (dB)"
            else:
                y /= 1000
                y_mod /= 1000
                title = f"Δ {metric} (kHz)"

            axes[i][j].hist(y, color="C0" if cv == "vowel" else "C1", range=(y_mod.min(), y_mod.max()), bins=40, density=True)
            axes[i][j].set_ylabel("Density")
            axes[i][j].set_title(f"{title}")

    plt.savefig(f"plots/main/synth-density-{dataset}-{model}.pdf")


def _plot_cos_density(feats, sample_phon_vectors, avg_phon_vectors, counts, dataset, phone_phon_vectors=None):
    show_phonewise = True if phone_phon_vectors is not None else False

    fig, axes = plt.subplots(
        1, 8,
        figsize=(14, 2.4),
        constrained_layout=True,
        sharey=True,
    )
    axes = axes.flatten()

    for ax, feat in zip(axes, feats):
        for i, count in enumerate(counts):
            dists = cdist(sample_phon_vectors[count][feat], avg_phon_vectors[feat][np.newaxis, :], metric="cosine")
            kwargs = dict(histtype="step", alpha=0.7) if show_phonewise else dict(alpha=0.5)
            ax.hist(1.0 - dists, range=(-0.2, 1), bins=50, density=True, label=str(count), **kwargs)

        if show_phonewise:
            dists = cdist(phone_phon_vectors[feat], avg_phon_vectors[feat][np.newaxis, :], metric="cosine")
            ax.hist(1.0 - dists, range=(-0.2, 1), bins=50, density=True, alpha=0.8, label="Phone pair")

        if feat == feats[0]:
            ax.legend(frameon=False)
            ax.set_ylabel("Density")
            ax.set_xlabel("Cosine similarity")
        ax.set_title(feat)
    plt.savefig(f"plots/main/phonovector-{dataset}-density-{show_phonewise}.pdf")


def _plot_cos_matrix(feats, avg_phon_vectors, fname, path="main"):
    matrix = np.zeros((len(feats), len(feats)))

    for i in range(len(feats)):
        for j in range(i, len(feats)):
            # samplewise vs. whole
            # dists = cdist(avg_phon_vectors[feats[i]][np.newaxis, :], sample_phon_vectors[feats[j]], metric="cosine")
            # phonewise vs whole
            # dists = cdist(avg_phon_vectors[feats[i]][np.newaxis, :], phone_phon_vectors[feats[j]], metric="cosine")
            # whole vs whole
            dists = cdist(avg_phon_vectors[feats[i]][np.newaxis, :], avg_phon_vectors[feats[j]][np.newaxis, :], metric="cosine")

            matrix[i, j] = matrix[j, i] = (1.0 - dists).mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap=plt.cm.PuOr, vmin=-1, vmax=1)

    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    if len(matrix) > 8:
        ax.set_title("Contextual Phonological Vector Comparison")
        ax.set_xticks(np.arange(-0.5, len(matrix), 8), minor=True)
        ax.set_yticks(np.arange(-0.5, len(matrix), 8), minor=True)
        n = len(matrix) // 8 // 2
        tickrange = [f"{i:+d}" if i != 0 else "0" for i in range(-n, n + 1)]
        ax.set_xticks(np.arange(4, len(matrix), 8), tickrange, minor=False)
        ax.set_yticks(np.arange(4, len(matrix), 8), tickrange, minor=False)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=1, alpha=0.5)
        ax.tick_params(which="minor", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', length=0)
    else:
        ax.set_xticklabels(feats)
        ax.set_yticklabels(feats)
        ax.set_title("Phonological Vector Comparison")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                color = "white" if abs(matrix[i, j]) > 0.5 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center",
                        color=color, fontsize=9)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"plots/{path}/phonovector-{fname}-matrix.pdf")


def _plot_phonological_vector_analysis(dataset, model="wavlm-large-24-featslice", count_analysis=True, **kwargs):
    df = pd.read_pickle(f"feats/{dataset}-{model}.pkl")
    df_train, df_test = _split_train_test(df)
    phones = filter_phones(df_test)

    cases = [
        ("hi", [("cons", -1)]),
        ("lo", [("cons", -1)]),
        ("back", [("cons", -1)]),
        ("round", [("cons", -1)]),
        ("nas", [("cons", 1)]),
        ("son", [("cons", 1)]),
        ("strid", [("cons", 1)]),
        ("voi", [("cons", 1)]),
    ]

    pos_neg_phones = {
        feat: separate_phones(phones, feat, constraints)
        for feat, constraints in cases
    }

    # Calculate various types of phonological vectors
    avg_phon_vectors = {}
    phone_phon_vectors = {}
    phone_phon_dists = {}

    counts = (1, 4, 16, 64, 256)
    sample_phon_vectors = {count: {} for count in counts}

    rng = np.random.default_rng(42)
    sample_size = 1000

    for feat, (pos_phones, neg_phones) in tqdm(pos_neg_phones.items()):
        df_pos = df_train[df_train.ipa.isin(pos_phones)]
        df_neg = df_train[df_train.ipa.isin(neg_phones)]

        pos = np.stack(df_pos.feat.tolist())
        neg = np.stack(df_neg.feat.tolist())

        # Average phonological vector (ours)
        avg_phon_vectors[feat] = pos.mean(0) - neg.mean(0)

        # Phonewise phonological vectors
        vecs, dists = [], []
        phnwise_avg = {}
        for phn in (pos_phones | neg_phones):
            _df = df_train[df_train.ipa == phn]
            if len(_df) > 0:
                phnwise_avg[phn] = np.stack(_df.feat.tolist()).mean(0)
        for pos_phn in pos_phones:
            for neg_phn in neg_phones:
                if pos_phn in phnwise_avg and neg_phn in phnwise_avg:
                    vecs.append(phnwise_avg[pos_phn] - phnwise_avg[neg_phn])
        phone_phon_vectors[feat] = np.array(vecs)

        if count_analysis:
            # Samplewise phonological vectors
            for count in counts:
                pos_indices = rng.choice(len(pos), size=sample_size * count, replace=True)
                neg_indices = rng.choice(len(neg), size=sample_size * count, replace=True)
                feats = pos[pos_indices] - neg[neg_indices]
                if count == 1:
                    sample_phon_vectors[count][feat] = feats
                else:
                    sample_phon_vectors[count][feat] = feats.reshape(sample_size, count, -1).mean(axis=1)

    if count_analysis:
        # Plot cos density plots
        _plot_cos_density(list(pos_neg_phones.keys()), sample_phon_vectors, avg_phon_vectors, counts, dataset)
        _plot_cos_density(list(pos_neg_phones.keys()), sample_phon_vectors, avg_phon_vectors, counts, dataset, phone_phon_vectors)

    # Plot cos matrix
    _plot_cos_matrix(list(pos_neg_phones.keys()), avg_phon_vectors, fname=dataset, **kwargs)


def _calculate_contextual_vectors(df, phones, locations):
    column_name = {"ipa": "0"}
    for i in range(5):
        column_name[f"l_{i}"] = f"-{i}"
        column_name[f"r_{i}"] = f"+{i}"

    cases = [
        ("hi", [("cons", -1)]),
        ("lo", [("cons", -1)]),
        ("back", [("cons", -1)]),
        ("round", [("cons", -1)]),
        ("nas", [("cons", 1)]),
        ("son", [("cons", 1)]),
        ("strid", [("cons", 1)]),
        ("voi", [("cons", 1)]),
    ]

    pos_neg_phones = {
        feat: separate_phones(phones, feat, constraints)
        for feat, constraints in cases
    }

    # Calculate various types of phonological vectors
    avg_phon_vectors = {}

    for col in locations:
        for feat, (pos_phones, neg_phones) in pos_neg_phones.items():
            df_pos = df[df[col].isin(pos_phones)]
            df_neg = df[df[col].isin(neg_phones)]

            pos = np.stack(df_pos.feat.tolist())
            neg = np.stack(df_neg.feat.tolist())

            # Average phonological vector (ours)
            avg_phon_vectors[f"{feat} ({column_name[col]})"] = pos.mean(0) - neg.mean(0)

    return avg_phon_vectors


def _plot_contextual_vector_analysis(dataset, model, phones, locations=["l_2", "l_1", "ipa", "r_1", "r_2"]):
    df = pd.read_pickle(f"feats/{dataset}-{model}-center-featslice.pkl")
    avg_phon_vectors = _calculate_contextual_vectors(df, phones, locations)
    _plot_cos_matrix(list(avg_phon_vectors.keys()), avg_phon_vectors, fname=f"{dataset}-{model}-position", path="contextual")


def _calculate_layerwise_vectors(dataset, model, phones, locations=["l_2", "l_1", "ipa", "r_1", "r_2"]):
    path = f"feats/{dataset}-{model}-layerwise-vectors.pkl"
    if Path(path).exists():
        print(f"Loading layerwise vectors from {path}")
        return pickle.load(open(path, "rb"))

    layerwise_vectors = []
    for layer in tqdm(range(25)):
        df = pd.read_pickle(f"feats/{dataset}-{model}-{layer}-center-featslice.pkl")
        layerwise_vectors.append(_calculate_contextual_vectors(df[df.split == "test"], phones, locations))
    with open(path, "wb") as f:
        pickle.dump(layerwise_vectors, f)
    return layerwise_vectors


def _plot_contextual_orthogonality(dataset, models, phones, locations=["l_2", "l_1", "ipa", "r_1", "r_2"]):
    fig, axes = plt.subplots(1, len(models), figsize=(5, 1.7), sharey=True, constrained_layout=True)

    for ax, model in zip(axes, models):
        layerwise_vectors = _calculate_layerwise_vectors(dataset, model, phones, locations)
        same_pos, diff_pos = [], []

        for layer in range(25):
            _same_pos, _diff_pos = [], []
            for k1, k2 in product(layerwise_vectors[layer].keys(), layerwise_vectors[layer].keys()):
                d = 1.0 - cdist(layerwise_vectors[layer][k1][np.newaxis, :], layerwise_vectors[layer][k2][np.newaxis, :], metric="cosine").item()
                k1_feat, k1_pos = k1.split()
                k2_feat, k2_pos = k2.split()
                v_phon = ["hi", "lo", "back", "round"]
                c_phon = ["nas", "son", "strid", "voi"]
                if k1 == k2:
                    pass
                elif ((k1_feat in v_phon and k2_feat in v_phon) or (k1_feat in c_phon and k2_feat in c_phon)) and k1_pos == k2_pos:
                    _same_pos.append(d)
                else:
                    _diff_pos.append(d)
            same_pos.append(np.abs(np.array(_same_pos)).mean())
            diff_pos.append(np.abs(np.array(_diff_pos)).mean())

        model_name = {
            "wavlm-large": "WavLM", "hubert-large": "HuBERT", "w2v2-large": "wav2vec 2.0",
        }[model]
        ax.plot(same_pos, ".-", color="C0", label="Within")
        ax.plot(diff_pos, ".-", color="C1", label="Across")
        ax.set_title(model_name)
        ax.set_xticks([0, 10, 20])
        ax.set_xlabel("Layer index")
        if model == models[0]:
            ax.set_ylabel("Avg. abs. cos. sim.")
        if model == models[-1]:
            ax.legend(frameon=False, labelspacing=0.3, borderpad=0.2,)
    plt.savefig(f"plots/contextual/orthogonality-layerwise-{dataset}.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close()


def _plot_contextual_layerwise_norm(dataset, models, phones, locations=["l_2", "l_1", "ipa", "r_1", "r_2"]):
    fig, axes = plt.subplots(1, len(models), figsize=(5, 1.8), constrained_layout=True)

    for ax, model in zip(axes, models):
        layerwise_vectors = _calculate_layerwise_vectors(dataset, model, phones, locations)
        layerwise_lengths = defaultdict(list)
        for layer in range(25):
            lengths = defaultdict(list)
            for name, vector in layerwise_vectors[layer].items():
                feat, loc = name.split()
                lengths[loc].append(np.linalg.norm(vector))
            for key, lens in lengths.items():
                layerwise_lengths[key].append(np.mean(lens))

        for key, lens in layerwise_lengths.items():
            ax.plot(lens, "--" if key[1] == "-" else "-", c=f"C{key[-2]}", label=key)

        model_name = {
            "wavlm-large": "WavLM", "hubert-large": "HuBERT", "w2v2-large": "wav2vec 2.0",
        }[model]
        ax.set_title(model_name)
        ax.set_xticks([0, 10, 20])
        ax.set_xlabel("Layer index")
        if model == models[0]:
            ax.set_ylabel("Avg. L2 norm")
            ax.legend(frameon=False, labelspacing=0.3, borderpad=0.2,)

    plt.savefig(f"plots/contextual/norm-layerwise-{dataset}.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close()


def _plot_masked_similarity(models, dataset="timit", bootstrapping=1000):
    Path("plots/contextual").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(models), figsize=(5, 2), sharey=True)
    for ax, model in zip(axes, models):
        df = pd.read_pickle(f"feats/{dataset}-{model}-masking-similarity.pkl")
        model_name = {
            "wavlm-large": "WavLM", "hubert-large": "HuBERT", "w2v2-large": "wav2vec 2.0",
        }[model]

        values = []
        num_layers = len(df.cos.iloc[0])
        for layer in tqdm(range(num_layers)):
            mean = np.concatenate(df.cos.apply(lambda x: x[layer]).values).mean()
            means = [
                np.concatenate(df.cos.apply(lambda x: x[layer]).sample(n=len(df), replace=True).values).mean()
                for _ in range(bootstrapping)
            ]
            ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])
            values.append((mean, ci_lo, ci_hi))
        values = np.array(values)

        ax.plot(values[:, 0], ".-", color="C3")
        ax.fill_between(np.arange(num_layers), values[:, 1], values[:, 2], alpha=0.3, color="C3", edgecolor="none")
        ax.set_title(model_name)
        ax.set_xticks([0, 10, 20])

    fig.tight_layout(rect=(0.05, 0.07, 1, 1))
    fig.supylabel("Avg. cos. sim.")
    fig.supxlabel("Layer index", y=0.05)
    plt.savefig("plots/contextual/masked-similarity.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close()



def _plot_pooling_comparison(dataset, feature_sets, models):
    Path("plots/contextual").mkdir(parents=True, exist_ok=True)
    mfcc_success, melspec_success, mfcc_audio_success, melspec_audio_success = _get_spectral_baselines(dataset, feature_sets)

    fig, axes = plt.subplots(1, 3, figsize=(6, 2), constrained_layout=True)
    axes = axes.flatten()
    feature = "total"

    for plot_index, model in enumerate(models):
        model_name = {
            "wavlm-large": "WavLM", "hubert-large": "HuBERT", "w2v2-large": "wav2vec 2.0",
        }[model]
        avg_success = _get_success_rates(pickle.load(open(f"feats/similarities-{dataset}-{model}-featslice.pkl", "rb")), feature_sets)
        center_success = _get_success_rates(pickle.load(open(f"feats/similarities-{dataset}-{model}-center-featslice.pkl", "rb")), feature_sets)

        p1 = axes[plot_index].plot(avg_success["total"], ".--", c="C6", label="Mean Pool.")[0]
        p2 = axes[plot_index].plot(center_success["total"], ".-", c="C3", label="Center Pool.")[0]
        axes[plot_index].set_title(f"{model_name}")
        axes[plot_index].set_ylim(-0.05, 1.05)

        l1 = axes[plot_index].axhline(mfcc_success[feature][0], ls="--", c="C0", label="MFCC")
        l2 = axes[plot_index].axhline(melspec_success[feature][0], ls="--", c="C1", label="MelSpec")

        legend_lines = [p2, p1, l1, l2]
        legend_labels = [l.get_label() for l in legend_lines]

        if plot_index % 3 == 0:
            axes[plot_index].set_ylabel("Success rate")
        axes[plot_index].set_xlabel("Layer index")

    plt.savefig(f"plots/contextual/pooling-comparison-{dataset}.pdf", bbox_inches="tight", pad_inches=0.0)

    fig_legend = plt.figure(figsize=(4, 1))
    fig_legend.legend(
        legend_lines,
        legend_labels,
        ncol=4,
        frameon=False,
        loc="center"
    )
    fig_legend.canvas.draw()
    plt.savefig("plots/contextual/pooling-comparison-legend.pdf", bbox_inches="tight", pad_inches=0.0)


def _plot_position_comparison(dataset, model, feature_sets, columns=["-l_2", "-l_1", "", "-r_1", "-r_2"]):
    column_name = {"": "0"}
    for i in range(5):
        column_name[f"-l_{i}"] = f"-{i}"
        column_name[f"-r_{i}"] = f"+{i}"

    Path("plots/contextual").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(6, 2), constrained_layout=True, sharey=True)
    axes = axes.flatten()
    feature = "total"

    for plot_index, (ax, col) in enumerate(zip(axes, columns)):
        fname = f"feats/similarities-{dataset}-{model}-center-featslice{col}.pkl"
        if not Path(fname).exists():
            print(f"Similarities for {fname} do not exist, skipping.")
            continue
        success = _get_success_rates(pickle.load(open(fname, "rb")), feature_sets)
        mfcc_success, melspec_success, mfcc_audio_success, melspec_audio_success = _get_spectral_baselines(dataset, feature_sets, postfix=col)

        p1 = ax.plot(success["total"], ".-", c="C3", label="S3M")[0]
        ax.set_title(f"{column_name[col]}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 10, 20])

        l1 = ax.axhline(mfcc_success[feature][0], ls="--", c="C0", label="MFCC")
        l2 = ax.axhline(melspec_success[feature][0], ls="--", c="C1", label="MelSpec")

        legend_lines = [p1, l1, l2]
        legend_labels = [l.get_label() for l in legend_lines]

        if plot_index == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="upper center")

    fig.supxlabel("Layer index")
    fig.supylabel("Success rate")

    plt.savefig(f"plots/contextual/position-comparison-{dataset}-{model}.pdf", bbox_inches="tight", pad_inches=0.0)


def _plot_position_all_models_comparison(dataset, models, feature_sets, columns=["-l_2", "-l_1", "", "-r_1", "-r_2"]):
    column_name = {"": "0"}
    for i in range(5):
        column_name[f"-l_{i}"] = f"-{i}"
        column_name[f"-r_{i}"] = f"+{i}"

    Path("plots/contextual").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(6, 2), constrained_layout=True, sharey=True)
    axes = axes.flatten()
    feature = "total"

    for plot_index, (ax, col) in enumerate(zip(axes, columns)):
        legend_lines = []
        for model in models:
            model_name, color = {
                "wavlm-large": ("WavLM", "C0"),
                "hubert-large": ("HuBERT", "C4"),
                "w2v2-large": ("wav2vec 2.0", "C1"),
            }[model]
            fname = f"feats/similarities-{dataset}-{model}-center-featslice{col}.pkl"
            if not Path(fname).exists():
                print(f"Similarities for {fname} do not exist, skipping.")
                continue
            success = _get_success_rates(pickle.load(open(fname, "rb")), feature_sets)

            p1 = ax.plot(success["total"], ".-", c=color, label=model_name)[0]
            legend_lines.append(p1)

        mfcc_success, melspec_success, mfcc_audio_success, melspec_audio_success = _get_spectral_baselines(dataset, feature_sets, postfix=col)
        l1 = ax.axhline(mfcc_success[feature][0], ls="--", c="C3", label="MFCC")
        l2 = ax.axhline(melspec_success[feature][0], ls="--", c="C2", label="MelSpec")

        ax.set_title(f"{column_name[col]}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 10, 20])

        legend_lines = legend_lines + [l1, l2]
        legend_labels = [l.get_label() for l in legend_lines]

    fig.supxlabel("Layer index")
    fig.supylabel("Success rate")

    plt.savefig(f"plots/contextual/position-comparison-{dataset}.pdf", bbox_inches="tight", pad_inches=0.0)

    fig_legend = plt.figure(figsize=(4, 1))
    fig_legend.legend(
        legend_lines,
        legend_labels,
        ncol=5,
        frameon=False,
        loc="center"
    )
    fig_legend.canvas.draw()
    plt.savefig("plots/contextual/position-comparison-legend.pdf", bbox_inches="tight", pad_inches=0.0)


def _plot_random_position_comparison(dataset, model, feature_sets, columns=["l_2", "l_1", "ipa", "r_1", "r_2"]):
    column_name = {"ipa": "0"}
    for i in range(5):
        column_name[f"l_{i}"] = f"-{i}"
        column_name[f"r_{i}"] = f"+{i}"

    Path("plots/contextual").mkdir(parents=True, exist_ok=True)

    results = pickle.load(open(f"feats/similarities-{dataset}-{model}-random-featslice.pkl", "rb"))
    bins = results["bins"]
    xs = [(lo + hi) / 2 for lo, hi in zip(bins[:-1], bins[1:])]

    fig, axes = plt.subplots(1, len(columns), figsize=(6, 2), constrained_layout=True, sharey=True)
    axes = axes.flatten()

    for ax, column in zip(axes, columns):
        rates = [_get_success_rates(r, feature_sets)["total"] for r in results[column]]
        ax.plot(xs, rates, ".-", c="C3")
        ax.set_title(f"{column_name[column]}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0.0, 0.5, 1.0])

    fig.supxlabel("Relative position")
    fig.supylabel("Success rate")

    plt.savefig(f"plots/contextual/random-position-comparison-{dataset}-{model}.pdf", bbox_inches="tight", pad_inches=0.0)


def _plot_random_position_modelwise_comparison(models, dataset, feature_sets, columns=["l_2", "l_1", "ipa", "r_1", "r_2"]):
    Path("plots/contextual").mkdir(parents=True, exist_ok=True)

    lines = {}
    for i, model in enumerate(models):
        results = pickle.load(open(f"feats/similarities-{dataset}-{model}-random-featslice.pkl", "rb"))
        bins = results["bins"]
        x_range = [(lo + hi) / 2 for lo, hi in zip(bins[:-1], bins[1:])]
        xs = []
        ys = []
        for i, column in enumerate(columns[::-1]):
            ys += [_get_success_rates(r, feature_sets)["total"] for r in results[column]]
            xs += [x - (len(columns) // 2 - i) for x in x_range]
        lines[model] = {
            "xs": xs,
            "ys": ys,
        }

    def _get_model(s):
        if "wavlm-large" in s:
            return "WavLM", "C0", False
        if "hubert-large" in s:
            return "HuBERT", "C4", False
        if "w2v2-large" in s:
            return "wav2vec 2.0", "C1", False
        if "melspec" in s:
            return "MelSpec", "C3", True
        if "mfcc" in s:
            return "MFCC", "C2", True
        raise ValueError(f"Unknown model: {s}")

    comparisons = [k for k, meta in lines.items() if not _get_model(k)[2]]
    baselines = [k for k, meta in lines.items() if _get_model(k)[2]]

    fig, ax = plt.subplots(1, 1, figsize=(6, 2), constrained_layout=True, sharex=True, sharey=True)
    legend_lines = []

    for key in comparisons:
        meta = lines[key]
        name, color, is_baseline = _get_model(key)
        legend_lines.append(ax.plot(meta["xs"], meta["ys"], ".-", label=name, color=color)[0])
    for baseline in baselines:
        meta = lines[baseline]
        name, color, is_baseline = _get_model(baseline)
        legend_lines.append(ax.plot(meta["xs"], meta["ys"], ".-", label=name, color=color)[0])

    ax.axvspan(0, 1, alpha=0.1, color="C0")
    for i, _ in enumerate(columns + [None]):
        ax.axvline(- (len(columns) // 2 - i), ls="--", c="C3", alpha=0.5)
    ax.set_ylabel("Success rate")
    ax.set_xticks([])
    ax.tick_params(axis='x', length=0)

    ax.set_xlabel("Relative phone position (normalized)")
    ticks = [- (len(columns) // 2 - i) for i in range(len(columns))]
    ax.set_xticks([t + 0.5 for t in ticks], ticks)

    plt.savefig(f"plots/contextual/random-position-comparison-{dataset}.pdf", bbox_inches="tight", pad_inches=0.0)

    fig_legend = plt.figure(figsize=(4, 1))
    fig_legend.legend(
        legend_lines,
        [l.get_label() for l in legend_lines],
        ncol=len(legend_lines),
        frameon=False,
        loc="center"
    )
    fig_legend.canvas.draw()
    plt.savefig("plots/contextual/random-position-comparison-legend.pdf", bbox_inches="tight", pad_inches=0.0)


def _plot_edges(dataset, model):
    Path("plots/contextual").mkdir(parents=True, exist_ok=True)

    _df = pd.read_pickle(f"feats/edges-{dataset}-{model}-center-featslice.pkl")
    for feat in _df.feat.unique():
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2), sharey=True)

        blue = np.mean(_df[(_df.direction == "left") & (_df.feat == feat)].curr.tolist(), axis=0)
        orange = np.mean(_df[(_df.direction == "left") & (_df.feat == feat)].prev.tolist(), axis=0)
        axes[0].plot(blue, ".-", c="C0", label="$cos(f, v^0)$")
        axes[0].plot(orange, ".-", c="C1", label="$cos(f, v^{+1})$")
        axes[0].axvspan(5, 10.5, alpha=0.3, color="C2")
        axes[0].set_xlim(-0.5, 10.5)
        axes[0].axhline(0, ls="--", c="C2")
        axes[0].text(0.25, 0.5, f"-{feat}", transform=axes[0].transAxes, ha="center", va="center", fontsize=18, alpha=0.3, weight="bold", zorder=1)
        axes[0].text(0.75, 0.5, f"+{feat}", transform=axes[0].transAxes, ha="center", va="center", fontsize=18, alpha=0.3, weight="bold", zorder=1)
        axes[0].set_ylabel("Avg. cos. sim.")
        axes[0].set_xlabel("Frame index")
        axes[0].set_xticks([1, 3, 5, 7, 9])

        blue = np.mean(_df[(_df.direction == "right") & (_df.feat == feat)].curr.tolist(), axis=0)
        orange = np.mean(_df[(_df.direction == "right") & (_df.feat == feat)].next.tolist(), axis=0)
        axes[1].plot(blue, ".-", c="C0", label="$cos(f, v^0)$")
        axes[1].plot(orange, ".-", c="C1", label="$cos(f, v^{-1})$")
        axes[1].axhline(0, ls="--", c="C2")
        axes[1].axvspan(-0.5, 5, alpha=0.3, color="C2")
        axes[1].set_xlim(-0.5, 10.5)
        axes[1].text(0.25, 0.5, f"+{feat}", transform=axes[1].transAxes, ha="center", va="center", fontsize=18, alpha=0.3, weight="bold", zorder=1)
        axes[1].text(0.75, 0.5, f"-{feat}", transform=axes[1].transAxes, ha="center", va="center", fontsize=18, alpha=0.3, weight="bold", zorder=1)
        axes[1].set_xlabel("Frame index")
        axes[1].set_xticks([1, 3, 5, 7, 9])

        plt.tight_layout()
        plt.savefig(f"plots/contextual/edges-{dataset}-{model}-{feat}.pdf", bbox_inches="tight", pad_inches=0.0)
        plt.close()


if __name__ == "__main__":
    # Figures for ACL 2026 submission
    # Dataset comparison
    timit_phs = filter_phones(pd.read_csv(f"feats/timit.csv"))
    voxangeles_quads = get_quadruples(filter_phones(pd.read_csv(f"feats/voxangeles.csv")))
    print(len(voxangeles_quads), "voxangeles quadruples")
    print(len([
        quad for quad in voxangeles_quads
        if any(ph not in timit_phs for ph in quad)
    ]), "voxangeles quadruples where unseen phone (timit) exist")

    # Plots
    for dataset in ["timit", "voxangeles"]:
        phones = filter_phones(pd.read_csv(f"feats/{dataset}.csv"))
        quadruples = get_quadruples(phones)

        feature_sets = _get_feature_sets(quadruples)
        feature_count_sets = _get_feature_count_sets(quadruples)
        feature_sets_cons, feature_sets_vowel = _get_consonant_vowel_feature_sets(feature_sets)

        _plot_individual_quadruplets(dataset)
        _plot_success_rates(dataset, feature_sets, "panphon")
        _plot_success_rates(dataset, feature_sets_cons, "panphon-cons")
        _plot_success_rates(dataset, feature_sets_vowel, "panphon-vowel")
        _plot_success_rates(dataset, feature_count_sets, "pfer")

        _plot_model_comparison(dataset, feature_sets, [
            "w2v2-large", "hubert-large", "wavlm-large",
        ], "featslice", name="model-comparison", print_sliced=False, only_featslice_baselines=True, include_legend=(dataset == "timit"))
        _plot_model_comparison(dataset, feature_sets, [
            "w2v2-large", "hubert-large", "wavlm-large",
        ], "featslice", name="model-comparison-full-featslice")
        _plot_consonant_vowel_comparison(dataset, feature_sets, feature_sets_cons, feature_sets_vowel, include_legend=(dataset == "timit"))
        _plot_cossim_comparison(dataset, feature_sets)
        _plot_synth_scatter(
            dataset, "wavlm",
            targets=["vowel-hi", "vowel-lo", "vowel-back", "vowel-round", "consonant-nas", "consonant-son", "consonant-strid", "consonant-voi"],
            metrics=["F1", "F1", "F2", "F3", "F1BW", "HNR", "COG", "COG"],
        )
        _plot_synth_scatter(
            dataset, "wavlm",
            targets=["consonant-nas", "consonant-nas", "consonant-nas", "consonant-nas", "consonant-nas", "consonant-nas", "consonant-nas", ],
            metrics=["F1", "F2", "F3", "F1BW", "ZCR", "HNR", "COG"],
        )
        _plot_synth_density(
            dataset, "wavlm",
            targets=["vowel-hi", "vowel-lo", "vowel-back", "vowel-round", "consonant-nas", "consonant-son", "consonant-strid", "consonant-voi"],
            metrics=["F1", "F1", "F2", "F2", "F1BW", "HNR", "COG", "COG"],
        )
        _plot_synth_scatter(
            dataset, "mfcc",
            targets=["vowel-hi", "vowel-lo", "vowel-back", "vowel-round", "consonant-nas", "consonant-son", "consonant-strid", "consonant-voi"],
            metrics=["F1", "F1", "F2", "F2", "F1BW", "HNR", "COG", "COG"],
        )
        _plot_synth_scatter(
            dataset, "mfcc-featslice",
            targets=["vowel-hi", "vowel-lo", "vowel-back", "vowel-round", "consonant-nas", "consonant-son", "consonant-strid", "consonant-voi"],
            metrics=["F1", "F1", "F2", "F2", "F1BW", "HNR", "COG", "COG"],
        )

        if dataset == "timit":
            _plot_model_comparison(dataset, feature_sets, [
                "w2v2-large", "hubert-large", "wavlm-large",
            ], "audioslice", name="model-comparison-full-audioslice")
            _plot_model_comparison(dataset, feature_sets, [
                "xlsr-53", "w2v2-phoneme", "w2v2-multipa",
            ], "featslice", name="model-comparison-pr", print_sliced=False)

        _plot_phonological_vector_analysis(dataset)

    # Figures for Interspeech 2026 submission
    # Figure 11
    _plot_masked_similarity(["w2v2-large", "hubert-large", "wavlm-large"])

    for dataset in ["timit", "voxangeles"]:
        phones = filter_phones(pd.read_csv(f"feats/{dataset}.csv"))
        quadruples = get_quadruples(phones)
        feature_sets = _get_feature_sets(quadruples)

        # Figure 2
        _plot_pooling_comparison(dataset, feature_sets, ["w2v2-large", "hubert-large", "wavlm-large"])
        for model in ["w2v2-large", "hubert-large", "wavlm-large"]:
            _plot_position_comparison(dataset, model, feature_sets=feature_sets)
        for model in ["wavlm-large-24", "hubert-large-24", "w2v2-large-9", "melspec-0", "mfcc-0"]:
            _plot_random_position_comparison(dataset, model, feature_sets=feature_sets)

        # Figure 3
        _plot_position_all_models_comparison(dataset, ["w2v2-large", "hubert-large", "wavlm-large"], feature_sets)
        # Figure 4
        _plot_random_position_modelwise_comparison(
            ["wavlm-large-24", "hubert-large-24", "w2v2-large-9", "melspec", "mfcc"],
            dataset, feature_sets, ["l_2", "l_1", "ipa", "r_1", "r_2"],
        )

        # Figure 8
        unfiltered_phones = filter_phones(pd.read_csv(f"feats/{dataset}.csv"), cutoff=0)
        _plot_contextual_orthogonality(dataset, ["w2v2-large", "hubert-large", "wavlm-large", ], unfiltered_phones)

        # Figure 6
        for model in ["wavlm-large-24", "hubert-large-24", "w2v2-large-9", "w2v2-large-22"]:
            _plot_contextual_vector_analysis(dataset, model, unfiltered_phones)

        # Figure 5
        _plot_phonological_vector_analysis(dataset, slice="center-featslice")

        # Figure 9
        _plot_edges(dataset, "wavlm-large-24")

        # Figure 7
        _plot_contextual_layerwise_norm(dataset, ["w2v2-large", "hubert-large", "wavlm-large"], unfiltered_phones)
