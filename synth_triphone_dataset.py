import argparse
from itertools import product
from pathlib import Path
from tqdm import tqdm

import numpy as np
import soundfile as sf
import pandas as pd
import torch
from datasets import Dataset, Audio
from TTS.api import TTS


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="tts_models/en/vits/vits")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_path", type=Path, default=Path("./triphones"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export_to_datasets", action="store_true")

    args = parser.parse_args()
    print(args)

    audio_path = args.output_path / "audio"
    if not audio_path.exists():
        audio_path.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == "__main__":
    args = _get_args()
    tts = TTS(args.model_name).to(args.device)

    # Customize TTS model
    tts.synthesizer.tts_model.tokenizer.use_phonemes = False
    tts.synthesizer.tts_model.tokenizer.text_cleaner = None

    # Generate words
    # vowels = ['i', 'ɪ', 'ʊ', 'u', 'ɛ', 'ə', 'æ', 'ɑ', 'ɔ']
    # consonants = ['ɹ', 'ɾ', 'm', 'n', 'ŋ', 'p', 't', 'ʧ', 'k', 'b', 'd', 'ʤ', 'ɡ' ,'f', 'θ', 's', 'ʃ', 'v', 'ð', 'z', 'ʒ', 'l', 'r']
    vowels = ['i', 'ɪ', ]
    consonants = ['ɹ', 'ɾ', 'm', ]

    vocab = tts.synthesizer.tts_model.tokenizer.characters.vocab
    assert all(p in vocab for p in vowels + consonants)

    words = [
        ("cvc", p1, p2, p3)
        for p1 in consonants
        for p2 in vowels
        for p3 in consonants
    ] + [
        ("vcv", p1, p2, p3)
        for p1 in vowels
        for p2 in consonants
        for p3 in vowels
    ]
    print("Total words:", len(words))
    print("Total speakers:", len(tts.speakers))
    print("Total synthesized words:", len(words) * len(tts.speakers))

    test_speaker_indices = np.random.default_rng(args.seed).choice(len(tts.speakers), size=len(tts.speakers)//2, replace=False)
    test_speakers = tts.speakers[test_speaker_indices]
    print(f"Test speakers ({len(test_speakers)}):", test_speakers)

    # Synthesize words
    metadata = []
    for speaker, (word_type, start_phone, middle_phone, end_phone) in tqdm(product(tts.speakers, words), total=len(words) * len(tts.speakers)):
        # Customize word a bit for better synthesis quality
        if word_type == "cvc":
            word = f"{start_phone}|ˈ{middle_phone}ː|{end_phone}"
            word_len = 5
            start_phone_len, end_phone_len = 1, 1
        elif word_type == "vcv":
            word = f"{start_phone}ː|{middle_phone}|ˈ{end_phone}ː"
            word_len = 6
            start_phone_len, end_phone_len = 2, 3
        else:
            raise ValueError(f"Invalid word type: {word_type}")

        # Give context to word for better synthesis quality
        text = f"aɪ s|ˈeɪ, {word}, ɐ|ɡ|ˈɛ|n"
        start_token_len, end_token_len = 17, 11

        # Feed to TTS model
        torch.manual_seed(args.seed)
        output = tts.synthesizer.tts_model.synthesize(text=text, speaker=speaker, use_cuda=args.device != "cpu")

        wav = output["wav"]
        tokens = output["text_inputs"].cpu().flatten()
        alignments = output["alignments"][0].cpu().bool()

        token_len, frame_len = alignments.shape
        stride_size = 256
        sample_rate = tts.synthesizer.output_sample_rate
        assert (stride_size * frame_len) == len(wav)
        assert (len(tokens) == token_len) and all(alignments.sum(0) == 1) and (alignments.sum() == frame_len)

        # Slice only the word of interest
        start_frame = stride_size * torch.nonzero(alignments[start_token_len])[0]
        end_frame = stride_size * torch.nonzero(alignments[-end_token_len])[0]
        wav = wav[start_frame:end_frame]

        # Obtain the time alignment of center phone
        start_phone_token_len = (start_phone_len + 1) * 2
        end_phone_token_len = (end_phone_len + 1) * 2 + 1
        middle_phone_alignment = alignments[start_token_len+start_phone_token_len:-end_token_len-end_phone_token_len].sum(0)
        center_min = stride_size * torch.nonzero(middle_phone_alignment)[0] - start_frame
        center_max = stride_size * torch.nonzero(middle_phone_alignment)[-1] - start_frame

        # Save audio
        audio_path = "audio" / f"{speaker}_{args.seed}_{start_phone}{middle_phone}{end_phone}.wav"
        sf.write(str(args.output_path / audio_path), wav, sample_rate)

        metadata.append({
            "speaker": speaker,
            "word": word,
            "word_type": word_type,
            "start_phone": start_phone,
            "middle_phone": middle_phone,
            "end_phone": end_phone,
            "full_context": text,
            "min": center_min / sample_rate,
            "max": center_max / sample_rate,
            "audio_path": str(audio_path),
            "split": "test" if speaker in test_speakers else "train",
        })

    df = pd.DataFrame(metadata)
    df.to_csv(args.output_path / "metadata.csv", index=False)

    if args.export_to_datasets:
        ds = Dataset.from_pandas(df)
        ds = ds.add_column("audio", df["audio_path"].tolist())
        ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
        ds.push_to_hub("juice500/triphone-vits-en", private=True)
