import pandas as pd
import numpy as np
import scipy.stats as st
from tqdm import tqdm
from functools import partial
from collections import defaultdict
import pickle
import multiprocessing as mp
import argparse
import panphon


def filter_phones(df, cutoff=100):
    ft = panphon.FeatureTable()
    df = df[df.split == "test"]
    phones = df[~df.ipa.isna()].ipa.unique().tolist()
    for phone, count in df.ipa.value_counts().items():
        if count < cutoff:
            print(f"{phone} has only {count} samples, removing.")
            phones.remove(phone)
        elif not ft.seg_known(phone):
            print(f"{phone} is not known in panphon, removing.")
            phones.remove(phone)
    return phones


def get_quadruples(phones):
    """Written by David Mortensen, modified by Kwanghee Choi."""

    ft = panphon.FeatureTable()
    assert all(ft.seg_known(p) for p in phones), "Some phones not in panphon"

    X = np.stack([ft.fts(p).numeric() for p in phones], axis=0)

    # Expand positive and negative features to binarized representation
    X_pos = (X == 1).astype(int)
    X_neg = (X == -1).astype(int)
    X = np.concatenate([X_pos, X_neg], axis=1)

    n, d = X.shape
    # Y = np.bitwise_xor(X[:, None, :], X[None, :, :])  # (n, n, d)
    Y = X[:, None, :] - X[None, :, :]  # (n, n, d)

    # Reshape for easier vectorized comparison
    Y_flat = Y.reshape(n * n, d)

    # For all (i,j) and (k,l), check analogy conditions
    # mask: Y[i,j] == Y[k,l], i.e., i-j = k-l
    mask = (Y_flat[:, None, :] == Y_flat[None, :, :]).all(-1)
    indices = np.argwhere(mask)

    # Convert back to 4D indices
    i, j = np.divmod(indices[:, 0], n)
    k, l = np.divmod(indices[:, 1], n)
    quadruples = np.stack([i, j, k, l], axis=1)

    # Remove rows with duplicate indices
    mask_unique = np.all(np.diff(np.sort(quadruples, axis=1), axis=1) != 0, axis=1)
    quadruples = quadruples[mask_unique]

    # Remove symmetric analogies
    normalized = quadruples.copy()
    normalized[:, 1:3] = np.sort(normalized[:, 1:3], axis=1)
    _, idx = np.unique(normalized, axis=0, return_index=True)
    quadruples = quadruples[idx]

    # Map to phones
    result = [tuple(phones[idx] for idx in row) for row in quadruples]

    # [ [0] = [1] + [2] - [3] ]
    return sorted(result)


def normalize(emb):
    if len(emb.shape) == 2:
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)
    elif len(emb.shape) == 1:
        return emb / np.linalg.norm(emb)
    else:
        raise ValueError("Invalid embedding shape")


def cos_montecarlo(X, Y, N=1000, seed=0):
    rng = np.random.default_rng(seed=seed)

    X_ids = rng.integers(0, X.shape[0], size=N)
    Y_ids = rng.integers(0, Y.shape[0], size=N)

    return (X[X_ids] * Y[Y_ids]).sum(1).mean(0)


def arith_montecarlo(A, B, C, D, N=1000, seed=0):
    rng = np.random.default_rng(seed=seed)

    A_ids = rng.integers(0, A.shape[0], size=N)
    B_ids = rng.integers(0, B.shape[0], size=N)
    C_ids = rng.integers(0, C.shape[0], size=N)
    D_ids = rng.integers(0, D.shape[0], size=N)

    Aprime = normalize(B[B_ids] + C[C_ids] - D[D_ids])
    return (A[A_ids] * Aprime).sum(1).mean(0)


def get_ci(func, p=0.99, N=10):
    trials = [func(seed=i) for i in range(N)]
    mean = np.mean(trials)
    se = np.std(trials, ddof=1) / np.sqrt(N)
    # Student's t
    ci = st.t.interval(p, N-1, loc=mean, scale=se)
    return (mean, ci[0], ci[1])


def process_quadruplet(df, quadruplet):
    a, b, c, d = quadruplet  # a = b + c - d

    A = np.array(df[df.ipa == a].feat.tolist())
    B = np.array(df[df.ipa == b].feat.tolist())
    C = np.array(df[df.ipa == c].feat.tolist())
    D = np.array(df[df.ipa == d].feat.tolist())
    notA = np.array(df[df.ipa != a].feat.tolist())

    return quadruplet, {
        "different": get_ci(partial(cos_montecarlo, A, notA)),
        "same": get_ci(partial(cos_montecarlo, A, A)),
        "arithmetic": get_ci(partial(arith_montecarlo, A, B, C, D)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compute synthesis distances over layers")
    parser.add_argument(
        "--dataset",
        choices=["timit", "voxangeles", ],
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=["wavlm-large", "hubert-large", "w2v2-large", ],
        required=True,
    )
    parser.add_argument(
        "--slice",
        choices=["audioslice", "featslice"],
        required=True,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    phones = filter_phones(pd.read_csv(f"feats/{args.dataset}.csv"))
    quadruples = get_quadruples(phones)

    results = {q: [] for q in quadruples}
    for layer in tqdm(range(25)):
        inpath = f"feats/{args.dataset}-{args.model}-{layer}-{args.slice}.pkl"
        df = pd.read_pickle(inpath)
        df.feat = df.feat.apply(normalize)
        df = df[df.ipa.isin(phones) & (df.split == "test")].reset_index(drop=True).copy()

        _process = partial(process_quadruplet, df)
        ctx = mp.get_context("fork")
        with ctx.Pool() as pool:
            for q, result_dict in pool.imap_unordered(_process, quadruples):
                results[q].append(result_dict)

    outname = f"feats/similarities-{args.dataset}-{args.model}-{args.slice}.pkl"
    with open(outname, "wb") as f:
        pickle.dump(results, f)
