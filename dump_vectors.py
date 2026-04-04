import argparse
import pickle
from pathlib import Path

import pandas as pd
import panphon

from plot_everything import _calculate_contextual_vectors
from analyze_synth import _split_train_test, get_phonological_vector, separate_phones
from estimate_similarity import filter_phones


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-path", type=str, help="Path to the features")
    parser.add_argument("--output-path", type=str, help="Path to the output file")
    parser.add_argument("--vector", choices=("phn", "ctx"))
    parser.add_argument("--vector-type", choices=("original", "full", "extended"))
    args = parser.parse_args()
    return args


def separate_phones_sign(phones, target_name, sign):
    pos_phones = []
    neg_phones = []

    ft = panphon.FeatureTable()
    for phone in phones:
        feat = ft.fts(phone)
        if feat[target_name] == sign:
            pos_phones.append(phone)
        else:
            neg_phones.append(phone)

    return set(pos_phones), set(neg_phones)


if __name__ == "__main__":
    args = _get_args()
    print(args)

    df = pd.read_pickle(args.feat_path)
    df_train, _ = _split_train_test(df)

    if args.vector_type == "original" and args.vector == "phn":
        phones = filter_phones(df)
        df = df_train
    else:
        phones = filter_phones(df, cutoff=0)

    ft = panphon.FeatureTable()
    features = ft.fts("a").names

    vectors = {}
    if args.vector_type == "original":
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
        if args.vector == "phn":
            for target, fixed in cases:
                pos_phones, neg_phones = separate_phones(phones, target, fixed)
                vectors[target] = get_phonological_vector(pos_phones, neg_phones, df)
        elif args.vector == "ctx":
            vectors = _calculate_contextual_vectors(df, phones, ["l_2", "l_1", "ipa", "r_1", "r_2"])
    elif args.vector_type == "full":
        if args.vector == "phn":
            for target in features:
                pos_phones, neg_phones = separate_phones(phones, target, [])
                if len(pos_phones) == 0 or len(neg_phones) == 0:
                    continue
                vectors[target] = get_phonological_vector(pos_phones, neg_phones, df)
        elif args.vector == "ctx":
            pos_neg_phones = {feat: separate_phones(phones, feat, []) for feat in features}
            pos_neg_phones = {feat: (pos_phones, neg_phones) for feat, (pos_phones, neg_phones) in pos_neg_phones.items() if len(pos_phones) > 0 and len(neg_phones) > 0}
            vectors = _calculate_contextual_vectors(df, phones, ["l_4", "l_3", "l_2", "l_1", "ipa", "r_1", "r_2", "r_3", "r_4"], pos_neg_phones=pos_neg_phones)
    elif args.vector_type == "extended":
        if args.vector == "phn":
            for target in features:
                for sign in (1, -1):
                    pos_phones, neg_phones = separate_phones_sign(phones, target, sign)
                    if len(pos_phones) == 0 or len(neg_phones) == 0:
                        continue
                    vectors[f"{target}{'+' if sign == 1 else '-'}"] = get_phonological_vector(pos_phones, neg_phones, df)
        elif args.vector == "ctx":
            pos_neg_phones = {f"{feat}{'+' if sign == 1 else '-'}": separate_phones_sign(phones, feat, sign) for feat in features for sign in (1, -1)}
            pos_neg_phones = {feat: (pos_phones, neg_phones) for feat, (pos_phones, neg_phones) in pos_neg_phones.items() if len(pos_phones) > 0 and len(neg_phones) > 0}
            vectors = _calculate_contextual_vectors(df, phones, ["l_4", "l_3", "l_2", "l_1", "ipa", "r_1", "r_2", "r_3", "r_4"], pos_neg_phones=pos_neg_phones)

    results = vars(args)
    results["vectors"] = vectors
    with open(args.output_path, "wb") as f:
        pickle.dump(results, f)
