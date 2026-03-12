import argparse
from pathlib import Path
import hashlib

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
    def __init__(self, model, synth_model, device="cpu"):
        self.synth = Vocos.from_pretrained(synth_model).to(device)
        self.device = device

        if model in ("mfcc", ):
            from extract_features import get_mfcc_vocos
            self.feat_func = get_mfcc_vocos()
            self.is_ssl = False
            self.sr = 24000
            self.stride = 256

            class FeatureExtractor(torch.nn.Module):
                def forward(self, x):
                    return x.permute(0, 2, 1)
            self.synth.feature_extractor = FeatureExtractor()
        else:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model)
            self.ssl = AutoModel.from_pretrained(model).to(device)
            self.is_ssl = True
            self.sr = 16000
            self.stride = 320

    def extract_feats(self, audio):
        if self.is_ssl:
            inputs = self.processor(
                raw_speech=[audio],
                sampling_rate=self.sr,
                padding=False,
                return_tensors="pt",
            )
            out = self.ssl(**{k: t.to(self.device) for k, t in inputs.items()})
            feats = out.last_hidden_state
        else:
            feats = self.feat_func(y=audio)
            feats = torch.from_numpy((feats.T)[None, :, :])
            feats = feats.to(self.device).to(torch.float32)

        return feats

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

    def modify(self, audio, vec, start, end):
        with torch.no_grad():
            feats = self.extract_feats(audio)
            feats = self.modify_feats(feats, vec, start, end)
            x_hat = self.synth(feats)
        return x_hat[0].cpu().numpy()

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
    f3 = [formant.get_value_at_time(3, t) for t in times if start < t < end]
    result["F3"] = np.mean(f3)

    f1bw = [formant.get_bandwidth_at_time(1, t) for t in times if start < t < end]
    result["F1BW"] = np.mean(f1bw)

    harmonicity = snd.to_harmonicity()
    hnr = [harmonicity.get_value(t) for t in times if start < t < end]
    result["HNR"] = np.mean(hnr)

    segment = snd.extract_part(from_time=start, to_time=end, preserve_times=False)
    result["COG"] = segment.to_spectrum().get_center_of_gravity()

    result["ZCR"] = np.mean(librosa.zero_crossings(audio[int(start*sr):int(end*sr)]))

    rms = librosa.feature.rms(y=audio[int(start*sr):int(end*sr)])
    result["RMSMIN"] = rms.min() / rms.mean()

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


def _split_train_test(df):
    # TIMIT
    if "train" in df.split.unique():
        train_df = df[df.split == "train"]
        test_df = df[df.split == "test"]

    # VoxAngeles
    else:
        assert "language" in df.columns
        df["language"] = df["language"].fillna("nan").astype("string")  # There's "nan" language

        keys = [(lang, int(hashlib.md5(lang.encode()).hexdigest(), 16)) for lang in df.language.unique()]
        keys = sorted(keys, key=lambda x: x[1])
        print(len(keys))
        assert len(keys) == 95
        mid = len(keys) // 2
        train_langs = [lang for lang, _ in keys[:mid]]
        test_langs  = [lang for lang, _ in keys[mid:]]
        train_df = df[df.language.isin(train_langs)]
        test_df  = df[df.language.isin(test_langs)]

    return train_df, test_df


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats", type=Path, help="Path to the features")
    parser.add_argument("--target_feature", choices=PANPHON_FEATURES, help="Phonological (panphon) feature to use")
    parser.add_argument("--fixed_features", nargs="+", type=_parse_feature_spec, help="Phonological (panphon) feature to keep fixed, ex. syl- cons+ voi0")
    parser.add_argument("--sample_size", default=3000, type=int, help="Number of phone samples to test")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--range_min", type=int, default=-5, help="Minimum lambda value")
    parser.add_argument("--range_max", type=int, default=5, help="Maximum lambda value")
    parser.add_argument("--output_path", type=Path, help="Path to the output file")
    parser.add_argument("--model", type=str, default=None, help="Huggingface model name or path to the SSL model (or 'mfcc' or 'melspec' for baseline)")
    parser.add_argument("--synth_model", type=str, default="juice500/vocos-wavlm-libritts", help="Huggingface model name or path to the vocos model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = _get_args()

    mp = ModifyPhone(model=args.model, synth_model=args.synth_model, device=args.device)

    df = pd.read_pickle(args.feats)
    df_train, df_test = _split_train_test(df)

    phones = filter_phones(df_test)
    pos_phones, neg_phones = separate_phones(phones, args.target_feature, args.fixed_features)
    vec = get_phonological_vector(pos_phones, neg_phones, df_train)

    rng = np.random.default_rng(seed=args.seed)
    results = []
    for _df, is_positive_phone in [
        (df_test[df_test.ipa.isin(pos_phones)], True),
        (df_test[df_test.ipa.isin(neg_phones)], False),
    ]:
        _df = _df.sample(args.sample_size, random_state=args.seed, replace=True)
        for row in tqdm(_df.itertuples()):
            weight = rng.uniform(args.range_min, args.range_max)
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
