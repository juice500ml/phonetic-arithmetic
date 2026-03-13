import argparse
import functools
import pickle
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/wavlm-large", help="Huggingface model name or spectral feature name")
    parser.add_argument("--dataset_csv", type=Path, help="Dataset to extract features")
    parser.add_argument("--split", default="test", choices=("train", "test", "both"), help="Dataset split to use")
    parser.add_argument("--output_path", type=Path, help="Output pkl path")
    parser.add_argument("--device", default="cpu", help="Device to infer, cpu or cuda:0 (gpu)")
    parser.add_argument("--layer_index", type=int, help="Layer index", default=-1)
    parser.add_argument("--pool", default="average", choices=("center", "average", "1q", "2q", "4q", "5q", "none", "random"), help="Pooling method")
    parser.add_argument("--slice", action="store_true", help="Slice audio")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (default: 16000)")
    args = parser.parse_args()

    print(args)
    return args


def _slice_feats(row, feats, stride_size, sr):
    f = feats[row.audio_path]
    def _sec_to_index(t):
        i = int(t * sr) // stride_size
        return np.clip(i, 0, len(f) - 1)
    start_index = _sec_to_index(row["min"])
    end_index = _sec_to_index(row["max"])
    return f[start_index:end_index+1]


def _pool_feats(row, pool):
    if pool == "center":
        return row.feat[int(len(row.feat) / 2)]
    elif pool == "1q":
        return row.feat[0]
    elif pool == "2q":
        return row.feat[int(len(row.feat) / 4)]
    elif pool == "4q":
        return row.feat[int(len(row.feat) * 3 / 4)]
    elif pool == "5q":
        return row.feat[-1]
    elif pool == "average":
        return row.feat.mean(0)
    elif pool == "random":
        index = np.random.randint(0, len(row.feat))
        return row.feat[index], index, len(row.feat)
    elif pool == "none":
        return row.feat
    else:
        raise ValueError(f"Wrong parameter for pool: {pool}")


def _get_stride_size(model):
    if model in ("melspec", "mfcc"):
        return 512
    elif model in ("melspec-kaldi", "mfcc-kaldi"):
        return 160
    else:
        return 320


def _get_window_size(model):
    if model in ("melspec", "mfcc"):
        return 1 # minimum size
    elif model in ("melspec-kaldi", "mfcc-kaldi"):
        return 400
    else: # ssl models
        return 400


def _slice_with_min_window(x, min_i, max_i, window_size):
    n = len(x)
    current_size = max_i - min_i
    if current_size >= window_size:
        return x[min_i:max_i]

    # Compute the extra size needed
    extra = window_size - current_size
    left_expand = extra // 2
    right_expand = extra - left_expand

    # Expand min_i and max_i
    new_min_i = min_i - left_expand
    new_max_i = max_i + right_expand

    # Adjust if new_min_i is out of bounds
    if new_min_i < 0:
        shift = -new_min_i
        new_min_i = 0
        new_max_i = min(n, new_max_i + shift)

    # Adjust if new_max_i is out of bounds
    if new_max_i > n:
        shift = new_max_i - n
        new_max_i = n
        new_min_i = max(0, new_min_i - shift)

    assert (new_max_i - new_min_i) >= window_size
    return x[new_min_i:new_max_i]


def _infer(x, processor, model, args):
    x = processor(raw_speech=[x], sampling_rate=args.sr, padding=False, return_tensors="pt")

    if args.layer_index == -1:
        outputs = model(**{k: t.to(args.device) for k, t in x.items()})
        return outputs.last_hidden_state.cpu().detach().numpy()[0]
    else:
        outputs = model(output_hidden_states=True, **{k: t.to(args.device) for k, t in x.items()})
        return outputs.hidden_states[args.layer_index].cpu().detach().numpy()[0]


def get_mfcc_vocos():
    import torch
    import torchaudio
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=24000,
        n_mfcc=40,
        log_mels=False,
        melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 100},
    )
    def _mfcc(y, n_fft=None):
        y = torch.from_numpy(y)
        if len(y) < 514:
            pad = 514 - len(y)
            y = torch.nn.functional.pad(y, (pad // 2, pad // 2), mode="constant", value=0)
        feat = mfcc(y)
        feat = torch.log(torch.clip(feat, min=1e-7))
        return feat.numpy()
    return _mfcc


# Closely follow Lhotse's implementation of Kaldi
def kaldi_povey_window(n):
    a = 2 * np.pi / (n - 1)
    return np.power(0.5 - 0.5 * np.cos(a * np.arange(n)), 0.85)


def melspec_kaldi(y, sr, n_mels=80, n_fft=None): # n_fft not used
    assert sr == 16000

    win_length = int(round(0.025 * sr))
    hop_length = int(round(0.01 * sr))

    y = y - np.mean(y)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=512,
        hop_length=hop_length,
        win_length=win_length,
        window=kaldi_povey_window(win_length),
        center=True,
        pad_mode="reflect",
        power=1.0,
        n_mels=n_mels,
        fmin=20,
        fmax=(sr // 2) - 400,
        htk=False,
        norm=None,
    )
    return np.log(np.maximum(S, 1e-10))


def mfcc_kaldi(y, sr, n_fft=None): # n_fft not used
    log_S = melspec_kaldi(y, sr, n_mels=23)
    n_mfcc = 13
    mfccs = librosa.feature.mfcc(
        S=log_S,
        n_mfcc=n_mfcc,
        dct_type=2,
        norm=None,
        lifter=0,
    )

    L = 22
    n = np.arange(1, n_mfcc + 1)
    lift = 1.0 + (L / 2.0) * np.sin(np.pi * n / L)
    mfccs *= lift[:, np.newaxis]

    return mfccs

if __name__ == "__main__":
    args = _get_args()
    np.random.seed(42)

    traditional_features = {
        "melspec": functools.partial(librosa.feature.melspectrogram, sr=args.sr),
        "mfcc": functools.partial(librosa.feature.mfcc, sr=args.sr),
        "melspec-kaldi": functools.partial(melspec_kaldi, sr=args.sr),
        "mfcc-kaldi": functools.partial(mfcc_kaldi, sr=args.sr),
        "mfcc-vocos": get_mfcc_vocos(),
    }

    df = pd.read_csv(args.dataset_csv)
    if args.split != "both":
        df = df[df.split == args.split]

    print("Extracting features...")
    if args.model in traditional_features.keys():
        feat_func = traditional_features[args.model]
        data = {}
        if args.slice:
            df["feat"] = None
            for path in tqdm(df.audio_path.unique()):
                x, _ = librosa.load(path, sr=args.sr, mono=True)
                for row in df[df.audio_path == path].itertuples():
                    sliced_x = _slice_with_min_window(x, int(row.min * args.sr), int(row.max * args.sr), _get_window_size(args.model))
                    df.at[row.Index, "feat"] = feat_func(y=sliced_x, n_fft=min(2048, len(sliced_x))).T
        else:
            data = {}
            for path in tqdm(df.audio_path.unique()):
                x, _ = librosa.load(path, sr=args.sr, mono=True)
                data[path] = feat_func(y=x, n_fft=min(2048, len(x))).T
            df["feat"] = df.apply(functools.partial(_slice_feats, feats=data, stride_size=_get_stride_size(args.model), sr=args.sr), axis=1)
    else:
        assert args.sr == 16000, "Only 16kHz is supported for SSL models."
        processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).to(args.device)

        if args.slice:
            df["feat"] = None
            for path in tqdm(df.audio_path.unique()):
                x, _ = librosa.load(path, sr=args.sr, mono=True)
                for row in df[df.audio_path == path].itertuples():
                    sliced_x = _slice_with_min_window(x, int(row.min * args.sr), int(row.max * args.sr), _get_window_size(args.model))
                    df.at[row.Index, "feat"] = _infer(sliced_x, processor, model, args)
        else:
            data = {}
            for path in tqdm(df.audio_path.unique()):
                x, _ = librosa.load(path, sr=args.sr, mono=True)
                data[path] = _infer(x, processor, model, args)
            df["feat"] = df.apply(functools.partial(_slice_feats, feats=data, stride_size=_get_stride_size(args.model), sr=args.sr), axis=1)

    df["feat"] = df.apply(functools.partial(_pool_feats, pool=args.pool), axis=1)

    if args.pool == "random":
        df["feat_index"] = df["feat"].apply(lambda x: x[1])
        df["feat_length"] = df["feat"].apply(lambda x: x[2])
        df["feat"] = df["feat"].apply(lambda x: x[0])

    df.to_pickle(args.output_path)
