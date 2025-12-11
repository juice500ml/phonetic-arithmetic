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

plt.style.use("tableau-colorblind10")

from estimate_similarity import filter_phones, get_quadruples, get_ci


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
    
            for quadruplet, similarity in similarities.items():
                if quadruplet in quadruplets:
                    arith_lo = np.array([m["arithmetic"][1] for m in similarity])
                    arith_hi = np.array([m["arithmetic"][2] for m in similarity])
                    same_lo = np.array([m["same"][1] for m in similarity])
                    diff_hi = np.array([m["different"][2] for m in similarity])
                    success_rate += ((same_lo > arith_hi) & (arith_lo > diff_hi)).astype(int)
            successes[feature] = success_rate / len(quadruplets)
    return successes


def _get_spectral_baselines(dataset, feature_sets):
    mfcc_success = _get_success_rates(pickle.load(open(f"feats/similarities-{dataset}-mfcc-featslice.pkl", "rb")), feature_sets)
    melspec_success = _get_success_rates(pickle.load(open(f"feats/similarities-{dataset}-melspec-featslice.pkl", "rb")), feature_sets)
    mfcc_audio_success = _get_success_rates(pickle.load(open(f"feats/similarities-{dataset}-mfcc-audioslice.pkl", "rb")), feature_sets)
    melspec_audio_success = _get_success_rates(pickle.load(open(f"feats/similarities-{dataset}-melspec-audioslice.pkl", "rb")), feature_sets)
    return mfcc_success, melspec_success, mfcc_audio_success, melspec_audio_success


def _plot_success_rates(dataset, feature_sets, name):
    Path("plots/success-rates").mkdir(parents=True, exist_ok=True)
    mfcc_success, melspec_success, mfcc_audio_success, melspec_audio_success = _get_spectral_baselines(dataset, feature_sets)
    for model in tqdm(["hubert-large", "w2v2-large", "w2v2-phoneme", "w2v2-multipa", "xlsr-53", "wavlm-large"]):
        for slice in ["featslice", "audioslice"]:
            if not Path(f"feats/similarities-{dataset}-{model}-{slice}.pkl").exists():
                print(f"Similarities for {dataset}-{model}-{slice} do not exist, skipping.")
                continue

            success = _get_success_rates(pickle.load(open(f"feats/similarities-{dataset}-{model}-{slice}.pkl", "rb")), feature_sets)

            fig, axes = plt.subplots(5, 4, figsize=(12, 16))
            axes = axes.flatten()
            for plot_index, (feature, success_rate) in enumerate(success.items()):
                axes[plot_index].plot(success_rate, "o-")
                axes[plot_index].set_title(f"{feature} ({len(feature_sets[feature])})")
                axes[plot_index].set_ylim(-0.05, 1.05)
                axes[plot_index].axhline(mfcc_success[feature][0], ls="--", c="C1")
                axes[plot_index].axhline(melspec_success[feature][0], ls="--", c="C2")
                axes[plot_index].axhline(mfcc_audio_success[feature][0], ls="-", c="C1")
                axes[plot_index].axhline(melspec_audio_success[feature][0], ls="-", c="C2")
            plt.tight_layout()
            plt.savefig(f"plots/success-rates/success-rates-{name}-{dataset}-{model}-{slice}.pdf")


def _plot_model_comparison(dataset, feature_sets, models, sliced, print_sliced=True, name=""):
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

        axes[plot_index].plot(success["total"], ".-", c="C3", label=model_name)
        axes[plot_index].set_title(f"{model_name} ({sliced_name})" if print_sliced else f"{model_name}")
        axes[plot_index].set_ylim(-0.05, 1.05)

        l1 = axes[plot_index].axhline(mfcc_success[feature][0], ls="--", c="C0", label="MFCC (feat)")
        l2 = axes[plot_index].axhline(melspec_success[feature][0], ls="--", c="C1", label="MelSpec (feat)")
        l3 = axes[plot_index].axhline(mfcc_audio_success[feature][0], ls="-",  c="C0", label="MFCC (audio)")
        l4 = axes[plot_index].axhline(melspec_audio_success[feature][0], ls="-",  c="C1", label="MelSpec (audio)")
        legend_lines = [l1, l2, l3, l4]
        legend_labels = [l.get_label() for l in legend_lines]

        print("success-rates MFCC (audio sliced) baseline", dataset, mfcc_audio_success[feature][0])

        if plot_index % 3 == 0:
            axes[plot_index].set_ylabel("Success rate")
        axes[plot_index].set_xlabel("Layer index")

    plt.savefig(f"plots/main/model-comparison{name}-{sliced}-{dataset}.pdf")

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


def _plot_consonant_vowel_comparison(dataset, f_sets, f_c_sets, f_v_sets, model="wavlm-large", sliced="featslice"):
    Path("plots/main").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(6, 2), constrained_layout=True)
    axes = axes.flatten()
    feature = "total"
    model_name = {
        "wavlm-large": "WavLM", "hubert-large": "HuBERT", "w2v2-large": "wav2vec 2.0",
        "w2v2-phoneme": "Wav2vec2Phoneme", "w2v2-multipa": "MultIPA", "xlsr-53": "XLSR-53",
    }[model]

    for plot_index, (feature_name, feature_sets) in enumerate(zip((f"Total", "Consonants", "Vowels"), [f_sets, f_c_sets, f_v_sets])):
        mfcc_success, melspec_success, mfcc_audio_success, melspec_audio_success = _get_spectral_baselines(dataset, feature_sets)

        sliced_name = {"featslice": "feat", "audioslice": "audio"}[sliced]
        success = _get_success_rates(pickle.load(open(f"feats/similarities-{dataset}-{model}-{sliced}.pkl", "rb")), feature_sets)
        axes[plot_index].plot(success["total"], ".-", c="C3", label=model_name)
        axes[plot_index].set_title(f"{feature_name} ({len(feature_sets[feature])})")
        axes[plot_index].set_ylim(-0.05, 1.05)
        axes[plot_index].axhline(mfcc_success[feature][0], ls="--", c="C0")
        axes[plot_index].axhline(melspec_success[feature][0], ls="--", c="C1")
        axes[plot_index].axhline(mfcc_audio_success[feature][0], ls="-",  c="C0")
        axes[plot_index].axhline(melspec_audio_success[feature][0], ls="-",  c="C1")
        if plot_index % 3 == 0:
            axes[plot_index].set_ylabel("Success rate")
        axes[plot_index].set_xlabel("Layer index")

    plt.savefig(f"plots/main/phone-comparison-{dataset}-{model}-{sliced}.pdf")


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
        
            if metric != "ZCR":
                y /= 1000
                axes[i][j].set_ylabel(f"Δ {metric} (kHz)")
            else:
                axes[i][j].set_ylabel(f"Δ {metric} (\%)")

            axes[i][j].set_xlabel(r"$\lambda$")
            axes[i][j].scatter(x, y, color="C0" if cv == "vowel" else "C1", s=0.2)
            axes[i][j].set_title(f"{title}\n$\\rho=${stats.statistic:.3f}")

    plt.savefig(f"plots/main/synth-scatter-{dataset}-{model}.pdf")

if __name__ == "__main__":
    for dataset in ["timit", "voxangeles"]:
        phones = filter_phones(pd.read_csv(f"feats/{dataset}.csv"))
        quadruples = get_quadruples(phones)

        feature_sets = _get_feature_sets(quadruples)
        feature_count_sets = _get_feature_count_sets(quadruples)
        feature_sets_cons, feature_sets_vowel = _get_consonant_vowel_feature_sets(feature_sets)

        # _plot_individual_quadruplets(dataset)
        # _plot_success_rates(dataset, feature_sets, "panphon")
        # _plot_success_rates(dataset, feature_count_sets, "pfer")

        # _plot_consonant_vowel_comparison(dataset, feature_sets, feature_sets_cons, feature_sets_vowel)
        _plot_cossim_comparison(dataset, feature_sets)
        # _plot_synth_scatter(
        #     dataset, "wavlm",
        #     targets=["vowel-hi", "vowel-lo", "vowel-back", "vowel-round", "consonant-nas", "consonant-son", "consonant-strid", "consonant-voi"],
        #     metrics=["F1", "F1", "F2", "F2", "F1BW", "ZCR", "COG", "COG"],
        # )

        # if dataset == "timit":
        #     _plot_model_comparison(dataset, feature_sets, [
        #         "w2v2-large", "hubert-large", "wavlm-large",
        #     ], "featslice")
        #     _plot_model_comparison(dataset, feature_sets, [
        #         "w2v2-large", "hubert-large", "wavlm-large", 
        #     ], "audioslice")
        #     _plot_model_comparison(dataset, feature_sets, [
        #         "xlsr-53", "w2v2-phoneme", "w2v2-multipa", 
        #     ], "featslice", name="-PR", print_sliced=False)
        # else:
        #     _plot_model_comparison(dataset, feature_sets, [
        #         "w2v2-large", "hubert-large", "wavlm-large",
        #     ], "featslice", print_sliced=False)

