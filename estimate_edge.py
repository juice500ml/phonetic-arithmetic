import argparse
import functools
from pathlib import Path

import panphon
import numpy as np
import pandas as pd
import librosa
from scipy.spatial.distance import cdist
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm

from estimate_similarity import filter_phones
from plot_everything import _calculate_contextual_vectors
from analyze_synth import _split_train_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--stride_size", type=int, default=320)
    parser.add_argument("--model", type=str, default="microsoft/wavlm-large")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    df = pd.read_pickle(args.feature_path)
    phones = filter_phones(df)
    train_df, df = _split_train_test(df)

    # Feature slicing
    stride_size = args.stride_size
    def _sec_to_index(f, t):
        i = int(t * args.sr) // stride_size
        return np.clip(i, 0, len(f) - 1)

    # Panphon features
    ft = panphon.FeatureTable()
    features = {name: i for i, name in enumerate(ft.fts("a").names)}
    _get_feat_arr = lambda p: np.array(ft.fts(p).numeric())
    v_phono_feats = ["hi", "lo", "back", "round"]
    c_phono_feats = ["nas", "son", "strid", "voi"]

    # Phonological vectors
    keys = [f"{f} ({i})" for f in v_phono_feats + c_phono_feats for i in ["-1", "0", "+1"]]
    avg_phon_vectors = _calculate_contextual_vectors(train_df, phones, ["l_1", "ipa", "r_1"])
    vectors = np.stack([avg_phon_vectors[k] for k in keys])

    # Look only at triplets
    df = df[(~df.ipa.isna()) & (~df.l_1.isna()) & (~df.r_1.isna())]
    df = df[df.ipa.isin(phones) & df.l_1.isin(phones) & df.r_1.isin(phones)]

    # Prepare models
    if args.model in ("mfcc", "melspec"):
        feat_func = {
            "melspec": functools.partial(librosa.feature.melspectrogram, sr=args.sr),
            "mfcc": functools.partial(librosa.feature.mfcc, sr=args.sr),
        }[args.model]
    else:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).to(args.device)

    # Calculate similarities
    results = []
    for path in tqdm(df.audio_path.unique()):
        audio, _ = librosa.load(path, sr=16000, mono=True)
        if args.model in ("mfcc", "melspec"):
            feats = feat_func(y=audio, n_fft=min(2048, len(audio))).T
        else:
            inputs = processor(
                raw_speech=[audio],
                sampling_rate=16000,
                padding=False,
                return_tensors="pt",
            )
            out = model(**{k: v.to(args.device) for k, v in inputs.items()})
            feats = out.last_hidden_state[0].detach().cpu().numpy()

        sims = 1.0 - cdist(vectors, feats, metric="cosine")
        sims = {k: sims[i] for i, k in enumerate(keys)}

        for row in df[df.audio_path == path].itertuples():
            start, end, label = row.min, row.max, row.ipa

            ipa_feat = _get_feat_arr(row.ipa)
            l_1_feat = _get_feat_arr(row.l_1)
            r_1_feat = _get_feat_arr(row.r_1)

            # Consonants
            if ipa_feat[features["cons"]] == +1:
                phono_feats = c_phono_feats
            # Vowels
            elif ipa_feat[features["cons"]] == -1:
                phono_feats = v_phono_feats

            for phn_name in phono_feats:
                result = {
                    "audio_path": path,
                    "min": row.min,
                    "max": row.max,
                    "ipa": row.ipa,
                    "l_1": row.l_1,
                    "r_1": row.r_1,
                    "feat": phn_name,
                }

                # Left case
                if l_1_feat[features[phn_name]] == -1 and ipa_feat[features[phn_name]] == +1:
                    edge_index = _sec_to_index(sims[f"{phn_name} (0)"], row.min)
                    max_index = len(sims[f"{phn_name} (0)"])
                    start_index = edge_index - args.frames
                    end_index = edge_index + args.frames + 1
                    if 0 <= start_index and end_index <= max_index:
                        result["direction"] = "left"
                        result["curr"] = sims[f"{phn_name} (0)"][start_index:end_index]
                        result["prev"] = sims[f"{phn_name} (+1)"][start_index:end_index]
                        results.append(result)

                # Right case
                if r_1_feat[features[phn_name]] == -1 and ipa_feat[features[phn_name]] == +1:
                    edge_index = _sec_to_index(sims[f"{phn_name} (0)"], row.max)
                    max_index = len(sims[f"{phn_name} (0)"])
                    start_index = edge_index - args.frames
                    end_index = edge_index + args.frames + 1
                    if 0 <= start_index and end_index <= max_index:
                        result["direction"] = "right"
                        result["curr"] = sims[f"{phn_name} (0)"][start_index:end_index]
                        result["next"] = sims[f"{phn_name} (-1)"][start_index:end_index]
                        results.append(result)

    # Save results
    pd.DataFrame(results).to_pickle(args.output_path)
