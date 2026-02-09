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
from pathlib import Path


def filter_phones(df, cutoff=50):
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


def split_df_per_position(
    df,
    columns,
    quadruples,
    bins,
    stride=320,
    window=400,
):
    if window == 0:
        # mfcc and melspec cases; it has padding
        df["relative_position"] = (df["feat_index"] + 0.5) / df["feat_length"]
    else:
        # s3m cases
        total = df["feat_length"] * stride + (window - stride)
        position = df["feat_index"] * stride + (window // 2)
        df["relative_position"] = position / total

    dfs = [
        df[(df.relative_position >= lo) & (df.relative_position < hi)].copy()
        for lo, hi in zip(bins[:-1], bins[1:])
    ]
    print("Split into %d bins" % len(dfs))
    print("Sample counts:", [len(_df) for _df in dfs])
    print("Removed samples:")
    print("left:", len(df[df.relative_position < bins[0]]))
    print("right:", len(df[df.relative_position >= bins[-1]]))

    # Sanity check
    filtered_quadruples = []
    for quad in quadruples:
        exist = True
        for _df in dfs:
            for column in columns:
                if not all((_df[column] == p).any() for p in quad):
                    exist = False
                    break
            if not exist:
                break
        if exist:
            filtered_quadruples.append(quad)
    
    if len(filtered_quadruples) < len(quadruples):
        print("WARNING: Some quadruples were not found in some bins, %d < %d" % (len(filtered_quadruples), len(quadruples)))

    return dfs, filtered_quadruples


def parse_args():
    parser = argparse.ArgumentParser(description="Compute synthesis distances over layers")
    parser.add_argument(
        "--dataset_name",
        default=None,
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--model",
        choices=["wavlm-large", "hubert-large", "w2v2-large", "xlsr-53", "w2v2-phoneme", "w2v2-multipa", "melspec", "mfcc"],
        required=True,
    )
    parser.add_argument(
        "--slice",
    )
    parser.add_argument(
        "--target_col",
        default="ipa",
    )
    parser.add_argument(
        "--is_random",
        action="store_true",
    )
    args = parser.parse_args()

    if args.dataset_name is None:
        args.dataset_name = args.dataset_path.stem

    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)

    if args.is_random:
        assert args.dataset_path.suffix == ".pkl", "Dataset path must be a pickle file"
        df = pd.read_pickle(args.dataset_path)

        phones = filter_phones(df)
        quadruples = get_quadruples(phones)
        df = df[df.ipa.isin(phones) & (df.split == "test")].reset_index(drop=True).copy()
        df.feat = df.feat.apply(normalize)

        columns = ["l_3", "l_2", "l_1", "ipa", "r_1", "r_2", "r_3"]
        # columns = ["l_1", "ipa", "r_1"]
        bins = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        stride, window = (512, 0) if args.model in ("melspec", "mfcc") else (320, 400)
        dfs, filtered_quadruples = split_df_per_position(df, columns, quadruples, bins, stride, window)

        results = dict()
        results["bins"] = bins

        for column in tqdm(columns):
            results[column] = list()
            for binned_df in tqdm(dfs):
                _df = binned_df.copy()
                _df["ipa"] = binned_df[column]
                rates = {q: [] for q in filtered_quadruples}

                ctx = mp.get_context("fork")
                _process = partial(process_quadruplet, _df)
                with ctx.Pool() as pool:
                    for q, result_dict in pool.imap_unordered(_process, filtered_quadruples):
                        rates[q].append(result_dict)
                results[column].append(rates)

        outname = f"feats/similarities-{args.dataset_path.stem}.pkl"

    else:
        assert args.dataset_path.suffix == ".csv", "Dataset path must be a CSV file"
        df = pd.read_csv(args.dataset_path)
        if args.target_col != "ipa":
            df["ipa"] = df[args.target_col]

        phones = filter_phones(df)
        quadruples = get_quadruples(phones)

        results = {q: [] for q in quadruples}
        for layer in tqdm(range(25)):
            inpath = f"feats/{args.dataset_name}-{args.model}-{layer}-{args.slice}.pkl"
            if not Path(inpath).exists():
                print(f"{inpath} does not exist, breaking.")
                break

            df = pd.read_pickle(inpath)
            if args.target_col != "ipa":
                df["ipa"] = df[args.target_col]
            df = df[df.ipa.isin(phones) & (df.split == "test")].reset_index(drop=True).copy()
            df.feat = df.feat.apply(normalize)

            _process = partial(process_quadruplet, df)
            ctx = mp.get_context("fork")
            with ctx.Pool() as pool:
                for q, result_dict in pool.imap_unordered(_process, quadruples):
                    results[q].append(result_dict)

        outname = f"feats/similarities-{args.dataset_name}-{args.model}-{args.slice}.pkl"
        if args.target_col != "ipa":
            outname = outname.replace(".pkl", f"-{args.target_col}.pkl")

    with open(outname, "wb") as f:
        pickle.dump(results, f)
