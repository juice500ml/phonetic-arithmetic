import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from estimate_similarity import filter_phones, get_quadruples


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def merge_equivalences(pairs):
    uf = UnionFind()

    for a, b in pairs:
        uf.union(a, b)

    groups = defaultdict(set)
    for x in uf.parent:
        groups[uf.find(x)].add(x)

    return list(groups.values())


def _pairwise_cos(cache, pairs, eps=1e-12):
    X = []
    for p1, p2 in pairs:
        X.append((cache[p1] - cache[p2]))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, eps)
    S = Xn @ Xn.T
    iu = np.triu_indices(S.shape[0], k=1)
    return S[iu]


def pcs_row(union, df, eps=1e-12):
    union = np.array(list(union))
    _feat_cache = dict()
    for p in union.flatten():
        _feat_cache[p] = np.array(df[df.ipa == p].feat.tolist())
        _feat_cache[p] /= np.linalg.norm(_feat_cache[p], axis=1, keepdims=True)
        _feat_cache[p] = _feat_cache[p].mean(0)

    non_union = np.array([
        (union[i][0], union[j][1])
        for i in range(len(union))
        for j in range(len(union))
        if i != j
    ])

    union_cos = _pairwise_cos(_feat_cache, union)
    non_union_cos = _pairwise_cos(_feat_cache, non_union)

    return roc_auc_score(
        np.array([1] * len(union_cos) + [0] * len(non_union_cos)),
        np.concatenate([union_cos, non_union_cos]),
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="timit")
    parser.add_argument("--model", type=str, default="wavlm-large")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = get_args()
    
    phones = filter_phones(pd.read_csv(f"feats/{args.dataset}.csv"))
    quadruples = get_quadruples(phones)
    pairs = [((p1, p2), (p3, p4)) for p1, p2, p3, p4 in quadruples] + [((p1, p3), (p2, p4)) for p1, p2, p3, p4 in quadruples]
    unions = merge_equivalences(pairs)
    unions = [union for union in unions if len(union) > 2]

    results = []
    for i in tqdm(range(24)):
        path = f"feats/{args.dataset}-{args.model}-{i}-featslice.pkl"
        if not Path(path).exists():
            print(f"{path} does not exist, stopping.")
            break
        df = pd.read_pickle(path)
        df = df[df.split == "test"]
        for union in unions:
            results.append({
                "layer": i,
                "union": union,
                "pcs": pcs_row(union, df),
            })

    df = pd.DataFrame(results)
    df.reset_index(drop=True).to_pickle(f"feats/pcs-{args.dataset}-{args.model}.pkl")

    print(df.groupby("layer").agg({"pcs": "mean"}))
