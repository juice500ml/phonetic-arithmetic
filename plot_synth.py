import pandas as pd
import argparse
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.misc
from pathlib import Path
import soundfile as sf
plt.style.use("tableau-colorblind10")

from analyze_synth import ModifyPhone, filter_phones, separate_phones, get_phonological_vector


# Written by Prof. David Harwath
class SpecPlotter(object):

    def __init__(self):
        # signal processing stuff
        self.fnotch = 60
        self.notchQ = 30
        self.assumed_rate = 16000
        self.coeff = 0.97
        self.window_size = 0.004 # 0.004 for wideband, 0.025 for narrowband
        self.window_stride = 0.001 #0.001 for wideband, 0.01 for narrowband
        self.hop_length = int(self.assumed_rate * self.window_stride)
        self.n_fft = 1024
        self.win_length = int(self.assumed_rate * self.window_size)
        self.window = scipy.signal.windows.hamming
        self.db_spread = 60
        self.db_cutoff = 3
        self.fig_height=10
        self.inches_per_sec = 10

    def compute_spectrogram(self, signal):
        g = scipy.signal.windows.gaussian(41, std=6)
        g = g / g.sum()
        y = signal - signal.mean()
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.win_length,
                                                 hop_length=self.hop_length)
        zcr = np.convolve(zcr[0], g, mode='same')
        zcr = zcr - zcr.min()
        b, a = scipy.signal.iirnotch(self.fnotch, self.notchQ, self.assumed_rate)
        y = scipy.signal.lfilter(b, a, y)
        y = np.append(y[0],y[1:]-self.coeff*y[:-1])
#        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.win_length,
#                                                 hop_length=self.hop_length)
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.win_length, window=self.window)
        power_spec = np.abs(stft)**2
        total_energy = 10*np.log10(np.sum(power_spec, axis=0))
        total_energy = total_energy - total_energy.max()
        total_energy = np.clip(total_energy, -1 * self.db_spread, 0)
        total_energy = total_energy - total_energy.min()
        total_energy = np.convolve(total_energy, g, mode='same')
        total_energy = total_energy - total_energy.max()
        f0 = int(np.round((125 / self.assumed_rate * .5) * power_spec.shape[0]))
        f1 = int(np.round((750 / self.assumed_rate * .5) * power_spec.shape[0]))
        lowfreq_energy = 10*np.log10(np.sum(power_spec[f0:f1,:], axis=0))
        lowfreq_energy = lowfreq_energy - lowfreq_energy.max()
        lowfreq_energy = np.clip(lowfreq_energy, -1 * self.db_spread, 0)
        lowfreq_energy = lowfreq_energy - lowfreq_energy.min()
        lowfreq_energy = np.convolve(lowfreq_energy, g, mode='same')
        lowfreq_energy = lowfreq_energy - lowfreq_energy.max()
        #mel_basis = librosa.filters.mel(self.assumed_rate, self.n_fft, n_mels=80,
        #                                fmin=20)
        #power_spec = np.dot(mel_basis, power_spec)
        logspec = librosa.power_to_db(power_spec, ref=np.max)
        logspec = np.flipud(logspec)
        clipped_logspec = np.clip(logspec, -1*self.db_spread, -1*self.db_cutoff)
        
        return y, clipped_logspec, zcr, total_energy, lowfreq_energy
    
    def plot_spectrogram(self, signal, outfile=None):
        y, spec, zcr, te, lfe  = self.compute_spectrogram(signal)
        extent= [0, signal.shape[0] / self.assumed_rate, 0,
                self.assumed_rate / 2000] # Convert x to seconds, y to kHz
        n_sec = signal.shape[0] / self.assumed_rate
        figure = plt.figure(figsize=(n_sec * self.inches_per_sec, self.fig_height))
        gs = figure.add_gridspec(nrows=5, ncols=1, height_ratios=[1,1,1,12,1], hspace=0.05)
        ax = figure.add_subplot(gs[0,0])
        # plot zcr
        plt.fill_between(np.arange(len(zcr)), zcr, y2=zcr.min(), color='gray')
        plt.margins(0, 0)
        ax.set_ylim(0, 1)
        plt.xticks([], [])
        plt.annotate('Zero Crossing Rate', (10, 0.6))
        plt.ylabel('kHz')
        plt.yticks([], [])
        ax = figure.add_subplot(gs[1,0])
        # plot total energy
        plt.fill_between(np.arange(len(te)), te, y2=te.min(), color='gray')
        plt.margins(0, 0)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.ylabel('dB')
        plt.annotate('Total Energy', (10, (-15/40)*np.abs(te.min())))
        ax = figure.add_subplot(gs[2,0])
        # plot low freq energy
        plt.fill_between(np.arange(len(lfe)), lfe, y2=lfe.min(), color='gray')
        plt.margins(0, 0)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.ylabel('dB')
        plt.annotate('Energy: 125 to 750 Hz', (10, -(15/40)*np.abs(lfe.min())))
        ax = figure.add_subplot(gs[3,0])
        # plot spec
        plt.imshow(spec, cmap='gist_gray_r', extent=extent, aspect='auto')
        #plt.xlabel('Time (s)')
        plt.ylabel('Frequency (kHz)')
        plt.xticks(np.arange(0, n_sec, 0.1))
        ax.tick_params(labelbottom=False, labelleft=True, labelright=True)
        #ax = plt.gca()
        ax.grid(color='k', linestyle='dotted', linewidth=0.5)
        plt.annotate('Wide Band Spectrogram', (0.01, 7.7))
        ax = figure.add_subplot(gs[4,0])
        # plot waveform
        plt.plot(y, linewidth=0.25, color='k')
        plt.margins(0, 0)
        ticks = np.arange(0, n_sec, 0.1)
        ticklabs = ["%.1f" % z for z in ticks]
        plt.xticks(ticks=self.assumed_rate*ticks, labels=ticklabs)
        plt.yticks([], [])
        plt.xlabel('Time (seconds)')
        plt.annotate('Waveform', (200, 0.3 * y.max()))
        if outfile:
            plt.savefig(outfile, bbox_inches='tight')
        else:
            plt.show()
        return


def _get_arrow_xy(xy, direction):
    dx, dy = {
        "<": (+0.05, -0.2),
        ">": (-0.05, -0.2),
        "^": (0, 2),
        "v": (-0.002, 2.5),
        "`": (-0.035, 1.4),
    }[direction]
    x, y = xy
    return (x + dx, y + dy)


if __name__ == "__main__":
    df = pd.read_pickle("feats/timit-wavlm-large-24-featslice.pkl")
    df_train = df[df.split == "train"]
    df_test = df[df.split == "test"]

    mp = ModifyPhone(model="microsoft/wavlm-large", synth_model="juice500/vocos-wavlm-libritts", device="cuda:0")
    phones = filter_phones(df_test)

    settings = [
        {
            "feature": "hi",
            "fixed_features": [("cons", -1)],
            "filename": "TIMIT/TEST/DR1/FAKS0/SA1.WAV",
            "phone": "i",
            "arrows": [[], [], [], [], [], [], [], []],
        },
        {
            "feature": "round",
            "fixed_features": [("cons", -1)],
            "filename": "TIMIT/TEST/DR1/FAKS0/SA1.WAV",
            "phone": "i",
            "arrows": [
                [],
                [],
                [],
                [{"c": "C1", "xy": (0.12, 2.344), "d": "<"}, {"c": "C4", "xy": (0.12, 2.864), "d": "<"}],
                [{"c": "C1", "xy": (0.12, 2.284), "d": "<"}, {"c": "C4", "xy": (0.12, 2.688), "d": "<"}],
                [{"c": "C1", "xy": (0.12, 2.006), "d": "<"}, {"c": "C4", "xy": (0.12, 2.322), "d": "<"}],
                [{"c": "C1", "xy": (0.12, 1.145), "d": "<"}, {"c": "C4", "xy": (0.12, 2.136), "d": "<"}],
            ]
        },
        {
            "feature": "voi",
            "fixed_features": [("cons", +1)],
            "filename": "TIMIT/TEST/DR1/FAKS0/SI1573.WAV",
            "phone": "b",
            "arrows": [
                [{"c": "C1", "xy": (0.14, 0.38), "d": "`"}],
                [{"c": "C1", "xy": (0.125,0.38), "d": "`"}],
                [{"c": "C1", "xy": (0.125,0.38), "d": "`"}],
                [{"c": "C1", "xy": (0.11, 0.38), "d": "`"}],
                [{"c": "C1", "xy": (0.10, 0.38), "d": "`"}],
                [{"c": "C1", "xy": (0.06, 0.38), "d": "`"}],
                [{"c": "C1", "xy": (0.055, 0.38), "d": "`"}],
            ],
        },
        {
            "feature": "strid",
            "fixed_features": [("cons", +1)],
            "filename": "TIMIT/TEST/DR1/FAKS0/SI1573.WAV",
            "phone": "b",
            "arrows": [
                [],
                [{"c": "C1", "xy": (0.114, 6), "d": "`"}, ],
                [{"c": "C1", "xy": (0.114, 6), "d": "`"}, ],
                [{"c": "C1", "xy": (0.114, 6), "d": "`"}, ],
                [{"c": "C4", "xy": (0.08, 6), "d": ">"}, ],
                [{"c": "C4", "xy": (0.08, 6), "d": ">"}, ],
                [{"c": "C4", "xy": (0.08, 6), "d": ">"}, ],
            ],
        },
        {
            "feature": "nas",
            "fixed_features": [("cons", +1)],
            "filename": "TIMIT/TEST/DR1/FAKS0/SI1573.WAV",
            "phone": "b",
            "arrows": [
                [],
                [],
                [],
                [{"c": "C1", "xy": (0.114, 6), "d": "`"}, ],
                [{"c": "C1", "xy": (0.114, 6), "d": "`"}, {"c": "C4", "xy": (0.1, 0.5), "d": "`"}, ],
                [{"c": "C4", "xy": (0.06, 0.5), "d": "`"}, ],
                [{"c": "C4", "xy": (0.08, 2), "d": ">"}, ],
            ],
        },
    ]

    Path("plots/main").mkdir(parents=True, exist_ok=True)
    Path("examples").mkdir(parents=True, exist_ok=True)

    for setting in settings:
        pos_phones, neg_phones = separate_phones(phones, setting["feature"], setting["fixed_features"])
        vec = get_phonological_vector(pos_phones, neg_phones, df_train)
        row = df_test[(df_test.ipa == setting["phone"]) & df_test.audio_path.str.contains(setting["filename"])].iloc[0]

        audio = mp.load_audio(row["audio_path"])
        fig, axes = plt.subplots(1, 7, figsize=(14, 2), constrained_layout=True)

        for i, (weight, arrows) in enumerate(zip((-5, -2, -1, 0, 1, 2, 5), setting["arrows"])):
            modified_audio = mp.modify(audio, vec * weight, row["min"], row["max"])
            signal = modified_audio[int(row["min"]*16000 - 800):int(row["max"]*16000 + 800)]
            sf.write(f"examples/synth-{setting['feature']}-{setting['phone']}-{weight}.wav", signal, 16000)

            _, spec, _, _, _ = SpecPlotter().compute_spectrogram(signal)
            extent = [0, signal.shape[0] / 16000, 0, 16000 / 2000]
            axes[i].imshow(spec, cmap='gist_gray_r', extent=extent, aspect='auto')
            axes[i].axvline(800/16000, c="C0")
            axes[i].axvline(800/16000 + (row["max"] - row["min"]), c="C0")
            axes[i].set_title(f"$\\lambda={weight}$")
            if i == 0:
                axes[i].set_ylabel("Frequency (kHz)")

            for arrow in arrows:
                axes[i].annotate(
                    ' ', color=arrow["c"], xy=arrow["xy"], xytext=_get_arrow_xy(arrow["xy"], arrow["d"]),
                    arrowprops=dict(arrowstyle='-|>', color=arrow["c"], lw=3, mutation_scale=20),
                )

        plt.savefig(f"plots/main/synth-{setting['feature']}-{setting['phone']}.pdf")
        plt.savefig(f"examples/synth-{setting['feature']}-{setting['phone']}.png")
