import random
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm


class ZCA:
    def __init__(self, data, epsilon=1e-9):
        # data: (num_samples, feature_size)
        assert len(data.shape) == 2
        # assert data.shape[0] < data.shape[1]
        self.mu = data.mean(0)
        cov = np.cov(data - self.mu, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # numerical stability
        invsqrt_eigvals = np.sqrt(1.0 / np.clip(eigvals, epsilon, None))
        self.W = eigvecs @ np.diag(invsqrt_eigvals) @ eigvecs.T

    def transform(self, X):
        return (X - self.mu) @ self.W


def _get_zca(audios, processor, model, device, sr):
    data = []
    for path in tqdm(train_audios):
        x, _ = librosa.load(path, sr=16000, mono=True)
        x = processor(raw_speech=[x], sampling_rate=sr, padding=False, return_tensors="pt")
        inputs = {k: t.to(device) for k, t in x.items()}
        with torch.no_grad():
            outputs = model(
                output_hidden_states=True,
                **inputs,
            )
        hss = np.array([
            hs.cpu().detach().numpy()[0]
            for hs in outputs.hidden_states
        ])
        data.append(hss)
    data = np.concatenate(data, axis=1)
    zcas = [ZCA(x) for x in tqdm(data, desc="ZCA initialization")]
    return zcas


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/wavlm-large", help="Huggingface model name or spectral feature name")
    parser.add_argument("--dataset_csv", type=Path, help="Dataset to extract features")
    parser.add_argument("--output_path", type=Path, help="Output pkl path")
    parser.add_argument("--device", default="cpu", help="Device to infer, cpu or cuda:0 (gpu)")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (default: 16000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _get_args()

    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)

    df = pd.read_csv(args.dataset_csv)
    random.seed(args.seed)

    train_audios = df.loc[df.split == "train", "audio_path"].drop_duplicates().sample(100, random_state=args.seed)
    test_audios = df[df.split == "test"].audio_path.unique()

    # Gather statistics from training set
    zcas = _get_zca(train_audios, processor, model, args.device, args.sr)

    # Compute masking similarity on test set
    results = []
    for i, path in enumerate(tqdm(test_audios)):
        x, _ = librosa.load(path, sr=args.sr, mono=True)
        x = processor(raw_speech=[x], sampling_rate=args.sr, padding=False, return_tensors="pt")
        inputs = {k: t.to(args.device) for k, t in x.items()}
        with torch.no_grad():
            outputs = model(
                output_hidden_states=True,
                **inputs,
            )

        t = outputs.hidden_states[0].shape[1]
        m = model.config.mask_feature_length
        start = random.randint(0, t - m)

        mask_time_indices = torch.zeros((1, t), dtype=torch.bool)
        mask_time_indices[0, start:start+m] = True
        mask_time_indices = mask_time_indices.to(args.device)

        with torch.no_grad():
            masked_outputs = model(
                output_hidden_states=True,
                mask_time_indices=mask_time_indices,
                **inputs,
            )
        mask_time_indices = mask_time_indices.cpu()

        hss = [
            hs.cpu().detach().numpy()[0]
            for hs in outputs.hidden_states
        ]
        masked_hss = [
            hs.cpu().detach().numpy()[0]
            for hs in masked_outputs.hidden_states
        ]

        # Only looking at masked region
        hss = np.array([
            hs[mask_time_indices[0]]
            for hs in hss
        ])
        masked_hss = np.array([
            hs[mask_time_indices[0]]
            for hs in masked_hss
        ])

        # ZCA normalization
        hss = np.array([
            zca.transform(hs)
            for zca, hs in zip(zcas, hss)
        ])
        masked_hss = np.array([
            zca.transform(hs)
            for zca, hs in zip(zcas, masked_hss)
        ])

        cos_sims = [
            (hs1 * hs2).sum(1) / (np.linalg.norm(hs1, axis=1) * np.linalg.norm(hs2, axis=1))
            for hs1, hs2 in zip(hss, masked_hss)
        ]
        results.append({
            "audio_path": path,
            "cos": cos_sims,
            "start": start,
            "end": start + m,
        })

    pd.DataFrame(results).reset_index(drop=True).to_pickle(args.output_path)
