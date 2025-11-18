import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import librosa
import panphon
import parselmouth
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from vocos import Vocos

from estimate_similarity import filter_phones


PANPHON_FEATURES = panphon.FeatureTable().fts("a").names


def get_phonological_vector(pos_phones, neg_phones, df_train):
    df_pos = df_train[df_train.ipa.isin(pos_phones)]
    df_neg = df_train[df_train.ipa.isin(neg_phones)]

    pos = np.stack(df_pos.feat.tolist())
    neg = np.stack(df_neg.feat.tolist())

    return pos.mean(0) - neg.mean(0)


def separate_phones(phones, target_name, fixed_features):
    pos_phones = []
    neg_phones = []

    ft = panphon.FeatureTable()
    for phone in phones:
        feat = ft.fts(phone)

        fixed = all(feat[name] == value for name, value in fixed_features)
        if fixed and feat[target_name] == 1:
            pos_phones.append(phone)
        if fixed and feat[target_name] == -1:
            neg_phones.append(phone)

    return set(pos_phones), set(neg_phones)


class ModifyPhone:
    def __init__(self, ssl_model, synth_model, device="cpu"):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(ssl_model)
        self.ssl = AutoModel.from_pretrained(ssl_model).to(device)
        self.synth = Vocos.from_pretrained(synth_model).to(device)
        self.device = device
        self.sr = 16000
        self.stride = 320

    def modify(self, audio, vec, start, end):
        inputs = self.processor(
            raw_speech=[audio],
            sampling_rate=self.sr,
            padding=False,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self.ssl(**{k: t.to(self.device) for k, t in inputs.items()})
            feats = out.last_hidden_state
            feats = self.modify_feats(feats, vec, start, end)
            x_hat = self.synth(feats)
        return x_hat[0].cpu().numpy()

    def modify_feats(self, feats, vec, start, end):
        _, T, _ = feats.shape
        def _sec_to_index(t):
            i = int(t * self.sr) // self.stride
            return np.clip(i, 0, T - 1)
        start_index = _sec_to_index(start)
        end_index = _sec_to_index(end)
        vec_tensor = torch.from_numpy(vec).to(feats.device).to(feats.dtype)
        feats[:, start_index:end_index+1, :] += vec_tensor
        return feats

    def load_audio(self, path):
        x, _ = librosa.load(path, sr=self.sr, mono=True)
        return x


def analyze_audio(audio, start, end, sr=16000):
    result = {}

    snd = parselmouth.Sound(values=audio, sampling_frequency=sr)
    times = snd.xs()

    formant = snd.to_formant_burg()
    f1 = [formant.get_value_at_time(1, t) for t in times if start < t < end]
    result["F1"] = np.mean(f1)
    f2 = [formant.get_value_at_time(2, t) for t in times if start < t < end]
    result["F2"] = np.mean(f2)

    harmonicity = snd.to_harmonicity()
    hnr = [harmonicity.get_value(t) for t in times if start < t < end]
    result["HNR"] = np.mean(hnr)

    segment = snd.extract_part(from_time=start, to_time=end, preserve_times=False)
    result["COG"] = segment.to_spectrum().get_center_of_gravity()

    return result


def _parse_feature_spec(token):
    if len(token) < 2:
        raise argparse.ArgumentTypeError(f"Invalid feature: {token}")

    feature, sign = token[:-1], token[-1]

    if feature not in PANPHON_FEATURES:
        raise argparse.ArgumentTypeError(f"Unknown feature: {feature}")

    if sign == "+":
        return feature, 1
    elif sign == "-":
        return feature, -1
    elif sign == "0":
        return feature, 0
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid feature sign '{sign}' in token: {token}"
        )


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats", type=Path, help="Path to the features")
    parser.add_argument("--target_feature", choices=PANPHON_FEATURES, help="Phonological (panphon) feature to use")
    parser.add_argument("--fixed_features", nargs="+", type=_parse_feature_spec, help="Phonological (panphon) feature to keep fixed, ex. syl- cons+ voi0")
    parser.add_argument("--sample_size", default=3000, type=int, help="Number of phone samples to test")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--range_min", type=int, default=-4, help="Minimum lambda value")
    parser.add_argument("--range_max", type=int, default=5, help="Maximum lambda value")
    parser.add_argument("--output_path", type=Path, help="Path to the output file")
    parser.add_argument("--ssl_model", type=str, default="microsoft/wavlm-large", help="Huggingface model name or path to the SSL model")
    parser.add_argument("--synth_model", type=str, default="juice500/vocos-wavlm-libritts", help="Huggingface model name or path to the vocos model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = _get_args()

    mp = ModifyPhone(ssl_model=args.ssl_model, synth_model=args.synth_model, device=args.device)

    df = pd.read_pickle(args.feats)
    df_train = df[df.split == "train"]
    df_test = df[df.split == "test"]

    phones = filter_phones(df_test)
    pos_phones, neg_phones = separate_phones(phones, args.target_feature, args.fixed_features)
    vec = get_phonological_vector(pos_phones, neg_phones, df_train)

    rng = np.random.default_rng(seed=args.seed)
    results = []
    for _df, is_positive_phone in [
        (df_test[df_test.ipa.isin(pos_phones)], True),
        (df_test[df_test.ipa.isin(neg_phones)], False),
    ]:
        _df = _df.sample(args.sample_size, random_state=args.seed)
        for row in tqdm(_df.itertuples()):
            weight = rng.uniform(args.range_min, args.range_max)
            weight = -weight if is_positive_phone else +weight

            audio = mp.load_audio(row.audio_path)
            synth_audio = mp.modify(audio, vec * 0.0, row.min, row.max)
            modified_audio = mp.modify(audio, vec * weight, row.min, row.max)

            result = {
                "phone": row.ipa,
                "audio_path": row.audio_path,
                "start": row.min,
                "end": row.max,
                "weight": weight,
                "is_positive_phone": is_positive_phone,
            }
            for x, name in [
                (audio, "original"),
                (synth_audio, "synth"),
                (modified_audio, "modified")
            ]:
                for key, value in analyze_audio(x, row.min, row.max).items():
                    result[f"{name}_{key}"] = value
            results.append(result)

    pd.DataFrame(results).to_csv(args.output_path, index=False)
