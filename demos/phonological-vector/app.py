import pickle
from pathlib import Path

import librosa
import numpy as np
import gradio as gr

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from specplotter import SpecPlotter

from vocos import Vocos
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch


def _read_alignment(fname):
    data = []
    with open(fname, "r") as f:
        for line in f:
            start, end, text = line.strip().split()
            data.append({
                "start": int(start),
                "end": int(end),
                "text": text,
            })
    return data

def _read_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)["vectors"]

def _audio_to_int16(x):
    x = np.clip(x, -1.0, 1.0)
    x = (x * 32767).astype(np.int16)
    return x

def _audio_to_float32(x):
    x = x.astype(np.float32) / 32767.0
    return x

def _read_audio(path):
    x, _ = librosa.load(path, sr=16000, mono=True)
    return 16000, _audio_to_int16(x)


class ModifyPhone:
    def __init__(self, model, synth_model, device="cpu"):
        self.synth = Vocos.from_pretrained(synth_model).to(device)
        self.device = device
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model)
        self.ssl = AutoModel.from_pretrained(model).to(device)
        self.sr = 16000
        self.stride = 320

    def extract_feats(self, audio):
        inputs = self.processor(
            raw_speech=[audio],
            sampling_rate=self.sr,
            padding=False,
            return_tensors="pt",
        )
        out = self.ssl(**{k: t.to(self.device) for k, t in inputs.items()})
        feats = out.last_hidden_state

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


def run_speech_edit(audio, audio_dropdown, start: float, end: float, vector_type: str, vector: str, weight: float, margin=400):
    if audio_dropdown in ("upload", "record"):
        sr, signal = audio
        x = _audio_to_float32(signal)
        if sr != ENGINE.sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=ENGINE.sr)
        start = np.clip(start, 0, len(x) / ENGINE.sr)
        end = np.clip(end, start, len(x) / ENGINE.sr)
        vec = PHON_VECTORS[vector_type][vector] * weight
        return ENGINE.sr, _audio_to_int16(ENGINE.modify(x, vec, start, end))
    else:
        x = ENGINE.load_audio(EXAMPLE_AUDIO)
        row = [w for w in EXAMPLE_WRD if w["text"] == audio_dropdown][0]
        s = max(0, row["start"] - margin)
        e = min(len(x), row["end"] + margin)

        start = int(start * ENGINE.sr) + s
        end = int(end * ENGINE.sr) + s
        start = np.clip(start, s, e)
        end = np.clip(end, start, e)

        vec = PHON_VECTORS[vector_type][vector] * weight
        signal = ENGINE.modify(x, vec, start / ENGINE.sr, end / ENGINE.sr)

        return ENGINE.sr, _audio_to_int16(signal[s:e])


def plot_spectrogram_edited(audio, start, stop):
    if audio is None:
        return None
    sr, signal = audio
    if sr != 16000:
        signal = _audio_to_float32(signal)
        signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
        sr = 16000

    start = np.clip(start, 0, len(signal) / sr)
    stop = np.clip(stop, start, len(signal) / sr)

    fig, ax = plt.subplots(figsize=(int(len(signal) / sr * 20), 4))
    plotter = SpecPlotter()
    plotter.plot_spectrogram(signal, ax=ax, show_annotation=False)
    ax.axvline(start, color="black", linewidth=1.5, linestyle="-", alpha=0.7)
    ax.axvline(stop, color="black", linewidth=1.5, linestyle="-", alpha=0.7)
    ax.add_patch(
        plt.Rectangle(
            (start, 7),
            stop - start,
            1,
            color="black",
            alpha=0.4,
            clip_on=False
        )
    )
    ax.text(
        (start + stop) / 2,
        7.5,
        "Selected",
        ha="center",
        va="center",
        color="white",
        fontsize=9
    )

    return fig


print("Loading phonological vectors...")
PHON_VECTORS = {
    "TIMIT (original)": _read_pkl("examples/original-timit.pkl"),
    "TIMIT (unconstrained)": _read_pkl("examples/unconstrained-timit.pkl"),
    "TIMIT (extended)": _read_pkl("examples/extended-timit.pkl"),
    "VoxAngeles (original)": _read_pkl("examples/original-voxangeles.pkl"),
    "VoxAngeles (unconstrained)": _read_pkl("examples/unconstrained-voxangeles.pkl"),
    "VoxAngeles (extended)": _read_pkl("examples/extended-voxangeles.pkl"),
}
print("Phonological vectors loaded!")

print("Loading models...")
DEVICE = "cpu"
ENGINE = ModifyPhone(
    model="microsoft/wavlm-large",
    synth_model="juice500/vocos-wavlm-libritts",
    device=DEVICE,
)
VOCOS = {
    "LibriTTS": ENGINE.synth,
    "FLEURS-R": Vocos.from_pretrained("juice500/vocos-wavlm-fleursr").to(DEVICE),
}
print("Models loaded!")


EXAMPLE_AUDIO = "examples/LDC93S1.wav"
EXAMPLE_PHN = _read_alignment("examples/LDC93S1.phn")
EXAMPLE_WRD = _read_alignment("examples/LDC93S1.wrd")
EXAMPLE_WRD.insert(0, {
    "start": 0,
    "end": EXAMPLE_WRD[-1]["end"],
    "text": "Full sentence",
})

def _read_partial_audio(audio_input, audio_dropdown, trigger_source, margin=400):
    if audio_dropdown in ("record", "upload"):
        return audio_input

    sr, signal = _read_audio(EXAMPLE_AUDIO)
    row = [w for w in EXAMPLE_WRD if w["text"] == audio_dropdown][0]
    start, end = row["start"], row["end"]
    start = max(0, start - margin)
    end = min(len(signal), end + margin)
    return sr, signal[start:end]

def plot_spectrogram_original(audio, audio_dropdown, margin=400):
    if audio is None:
        return None
    sr, signal = audio

    if audio_dropdown in ("record", "upload"):
        if sr != 16000:
            signal = _audio_to_float32(signal)
            signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
            sr = 16000

        fig, ax = plt.subplots(figsize=(int(len(signal) / sr * 20), 4))
        plotter = SpecPlotter()
        plotter.plot_spectrogram(signal, ax=ax, show_annotation=False)
        return fig

    sr, signal = _read_audio(EXAMPLE_AUDIO)
    row = [w for w in EXAMPLE_WRD if w["text"] == audio_dropdown][0]
    start, end = row["start"], row["end"]
    start = max(0, start - margin)
    end = min(len(signal), end + margin)
    signal = signal[start:end]

    fig, ax = plt.subplots(figsize=(int(len(signal) / sr * 20), 4))
    plotter = SpecPlotter()
    plotter.plot_spectrogram(signal, ax=ax, show_annotation=False)

    for p in EXAMPLE_PHN:
        if p["end"] >= start and p["start"] <= end:
            s = max(0, p["start"] - start) / sr
            e = min(len(signal), p["end"] - start) / sr

            ax.axvline(s, color="black", linewidth=1.5, linestyle="-", alpha=0.4)
            ax.axvline(e, color="black", linewidth=1.5, linestyle="-", alpha=0.4)
            ax.add_patch(
                plt.Rectangle(
                    (s, 7),
                    e - s,
                    1,
                    color="black",
                    alpha=0.4,
                    clip_on=False
                )
            )
            ax.text(
                (s + e) / 2,
                7.5,
                p["text"],
                ha="center",
                va="center",
                color="white",
                fontsize=9
            )

    return fig

def swap_synth(model_name):
    ENGINE.synth = VOCOS[model_name]


with gr.Blocks(title="Phonological Vector-based Speech Editing Demo") as demo:
    with gr.Row():
        gr.Markdown("""
## 🎙️ Phonological Vector-based Speech Editing Demo

Demonstration for the paper [[b]=[d]-[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](https://arxiv.org/abs/2602.18899).
This demo reproduces Experiment 2: Scale of Phonological Vectors, illustrating the controllability of speech editing by phonological vectors.

**Upload, record, or use the example audio (or word). Then, inspect the spectrogram, select the time window, choose a phonological vector to apply, then hit Run.**
(For the example words, we gave 0.25s margin to the start and end of the word.)""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
### Hyperparameters
- **Start / Stop (s)**: Time range (in seconds) over which the phonological vector is applied. Use the input spectrogram to identify the target phone's boundaries.
- **Lambda**: Strength of the phonological vector. Positive values strengthen the selected feature; negative values strengthens the opposite feature.
- **Vocos training dataset**: Training corpus used for the vocoder (Vocos) that resynthesizes the modified representation back to audio.
- **Vector extraction method**: How phonological vectors are estimated from S3M representations. Different options correspond to different training dataset/calculating the vectors.
- **Phonological feature**: The phonological vector to add into the selected time window.
""")

        with gr.Column(scale=1):
            gr.Markdown("""### Hyperparameters""")
            with gr.Row():
                start_time = gr.Number(label="Start (s)", value=0.0, precision=3, scale=1, interactive=True)
                stop_time = gr.Number(label="Stop (s)", value=1.0, precision=3, scale=1, interactive=True)
                vector_lambda = gr.Slider(label="Lambda",  value=0.0, minimum=-5, maximum=5, step=0.1, interactive=True)

                model_dropdown = gr.Dropdown(
                    label="Vocos training dataset",
                    choices=list(VOCOS.keys()),
                    value=next(iter(VOCOS.keys())),
                    interactive=True,
                )
                model_dropdown.change(
                    fn=swap_synth,
                    inputs=model_dropdown,
                )

                vector_type_dropdown = gr.Dropdown(
                    label="Vector extraction method",
                    choices=list(PHON_VECTORS.keys()),
                    value=next(iter(PHON_VECTORS.keys())),
                    interactive=True,
                )

                vector_dropdown = gr.Dropdown(
                    label="Phonological feature",
                    choices=list(next(iter(PHON_VECTORS.values())).keys()),
                    value=next(iter(next(iter(PHON_VECTORS.values())).keys())),
                    interactive=True,
                )
                vector_type_dropdown.change(
                    fn=lambda key: gr.Dropdown(choices=list(PHON_VECTORS[key].keys())),
                    inputs=vector_type_dropdown,
                    outputs=vector_dropdown,
                )
                run_btn = gr.Button("▶ Run", variant="primary", scale=1)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input audio")
            audio_dropdown = gr.Dropdown(
                choices=[w["text"] for w in EXAMPLE_WRD],
                label="Choose a word to modify (or record your own below)",
                value=None,
                interactive=True,
            )
            audio_input = gr.Audio(
                type="numpy",
                sources=["upload", "microphone"],
                recording=True,
                value=None,
            )
        with gr.Column(scale=1):
            gr.Markdown("### Output audio")
            audio_output = gr.Audio(type="numpy", interactive=False)


    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input spectrogram")
            trigger_source = gr.State(value=None)
            audio_dropdown.change(fn=lambda x: x, inputs=[audio_dropdown], outputs=[trigger_source])
            audio_input.upload(fn=lambda: "upload", inputs=[], outputs=[trigger_source])
            audio_input.stop_recording(fn=lambda: "record", inputs=[], outputs=[trigger_source])

            input_audio_plot = gr.Plot(
                show_label=True,
                elem_id="input-spectrogram-plot",
            )
            trigger_source.change(
                fn=_read_partial_audio,
                inputs=[audio_input, trigger_source],
                outputs=audio_input,
            ).then(
                fn=plot_spectrogram_original,
                inputs=[audio_input, trigger_source],
                outputs=input_audio_plot,
            )

        with gr.Column(scale=1):
            gr.Markdown("### Output spectrogram")
            output_audio_plot = gr.Plot(show_label=True)

            run_btn.click(
                fn=run_speech_edit,
                inputs=[audio_input, trigger_source, start_time, stop_time, vector_type_dropdown, vector_dropdown, vector_lambda],
                outputs=audio_output,
            )
            audio_output.change(
                fn=plot_spectrogram_edited,
                inputs=[audio_output, start_time, stop_time],
                outputs=output_audio_plot,
            )

if __name__ == "__main__":
    demo.launch()
