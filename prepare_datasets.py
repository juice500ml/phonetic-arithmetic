import glob
from pathlib import Path
from tqdm import tqdm
import argparse
import pandas as pd
import praatio.textgrid
import numpy as np
import librosa
import os
import re
from datasets import load_dataset


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset")
    parser.add_argument("--dataset_type", type=str, choices=["timit", "voxangeles"])
    parser.add_argument("--output_path", type=Path, help="Output csv folder")
    return parser.parse_args()


def _prepare_timit(timit_path: Path):
    # adapted from https://github.com/juice500ml/acoustic-units-for-ood/blob/main/dataset_prep.py
    timit = load_dataset("timit_asr", data_dir=timit_path, trust_remote_code=True)

    rows = []
    for split in ["train", "test"]:
        for utterance in tqdm(timit[split]):
            audio_path = utterance["audio"]["path"]
            speaker = utterance["speaker_id"]
            alignment = utterance["phonetic_detail"]
            for phn, start, stop in zip(alignment["utterance"], alignment["start"], alignment["stop"]):
                ipa = {
                    # Stops
                    "b": "b",
                    "d": "d",
                    "g": "ɡ",
                    "p": "p",
                    "t": "t",
                    "k": "k",
                    "dx": "ɾ",
                    "q": "ʔ",

                    # Affricates
                    "jh": "d͡ʒ",
                    "ch": "t͡ʃ",

                    # Fricatives
                    "s": "s",
                    "sh": "ʃ",
                    "z": "z",
                    "zh": "ʒ",
                    "f": "f",
                    "th": "θ",
                    "v": "v",
                    "dh": "ð",

                    # Nasals
                    "m": "m",
                    "n": "n",
                    "ng": "ŋ",
                    "em": "m̩",
                    "en": "n̩",
                    "eng": "ŋ̍",
                    "nx": "ɾ̃",

                    # Semivowels and Glides
                    "l": "l",
                    "r": "ɹ",
                    "w": "w",
                    "y": "j",
                    "hh": "h",
                    "hv": "ɦ",
                    "el": "l̩",

                    # Vowels
                    "iy": "i",
                    "ih": "ɪ",
                    "eh": "ɛ",
                    "ae": "æ",
                    "aa": "ɑ",
                    "ah": "ʌ",
                    "ao": "ɔ",
                    "uh": "ʊ",
                    "uw": "u",
                    "ux": "ʉ",
                    "er": "ɝ",
                    "ax": "ə",
                    "ix": "ɨ",
                    "axr": "ɚ",
                    "ax-h": "ə̯",

                    # Diphthongs
                    # These are ignored for simplicity
                    "ey": None,
                    "aw": None,
                    "ay": None,
                    "oy": None,
                    "ow": None,

                    # Stops closures
                    # These are attached to their succeeding stop
                    "bcl": None,
                    "dcl": None,
                    "gcl": None,
                    "pcl": None,
                    "tcl": None,
                    "kcl": None,

                    # Non-speech
                    "pau": None,
                    "epi": None,
                    "h#": None,
                }[phn]

                # Closure attached to their previous phone
                if phn in ("b", "d", "g", "p", "t", "k", "jh", "ch"):
                    closure = {"b": "bcl", "d": "dcl", "g": "gcl", "p": "pcl", "t": "tcl", "k": "kcl", "jh": "dcl", "ch": "tcl"}[phn]
                    if rows[-1]["timit_phn"] == closure:
                        start = rows[-1]["min"] * 16000

                rows.append({
                    "audio_path": audio_path,
                    "speaker": speaker,
                    "min": start / 16000,
                    "max": stop / 16000,
                    "timit_phn": phn,
                    "ipa": ipa,
                    "split": split
                })

    return pd.DataFrame(rows)


def _prepare_voxangeles(root_path: Path):
    rows = []

    for path in (root_path / "data/audited_aligned").glob("**/*.TextGrid"):
        grid = praatio.textgrid.openTextgrid(path, includeEmptyIntervals=False)
        tier_name = next((x for x in grid.tierNames if x in ("phone", "phones", "Narrow")))
        for entry in grid.getTier(tier_name).entries:
            rows.append({
                "audio_path": str(path.with_suffix(".wav")),
                "min": entry.start,
                "max": entry.end,
                "voxangeles_phn": entry.label,
                "split": "test",
                "language": path.parent.name,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = _get_args()
    print(args)

    _prepare = {
        "timit": _prepare_timit,
        "voxangeles": _prepare_voxangeles,
    }[args.dataset_type]
    df = _prepare(args.dataset_path)

    os.makedirs(args.output_path, exist_ok=True)
    csv_path = args.output_path / f"{args.dataset_type}.csv"
    df.to_csv(str(csv_path), index=False)
    print("Stored to", csv_path)
