# save as wav_to_melspec_hz.py
import os, sys, argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# -------- Import your training module (must be import-safe) --------
TRAIN_MOD = os.getenv("TRAIN_MOD", "morse_ctc_tpu")

def _import_training_module():
    try:
        return __import__(TRAIN_MOD, fromlist=["*"])
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent
        sys.path.insert(0, str(repo_root))
        return __import__(TRAIN_MOD, fromlist=["*"])

training = _import_training_module()

# Pull feature function & constants from training
wav_to_logmels = getattr(training, "wav_to_logmels")
SR             = int(getattr(training, "SR", 16000))
N_FFT          = int(getattr(training, "N_FFT", 1024))
N_MELS_DEFAULT = int(getattr(training, "N_MELS", 32))
LOWER_HZ_DEFAULT = float(getattr(training, "LOWER_EDGE_HZ", 20.0))
UPPER_HZ_DEFAULT = float(getattr(training, "UPPER_EDGE_HZ", SR / 2.0))

def _resample_1d_tf(wav, src_sr, dst_sr):
    """Lightweight mono resample using TF's bilinear resize."""
    if src_sr == dst_sr:
        return wav
    T = tf.shape(wav)[0]
    new_T = tf.cast(tf.math.round(tf.cast(T, tf.float32) * (dst_sr / float(src_sr))), tf.int32)
    x = tf.reshape(wav, [1, 1, T, 1])
    x = tf.image.resize(x, size=(1, new_T), method="bilinear")
    return tf.reshape(x, [new_T])

def load_mono_wav(path):
    audio_bytes = tf.io.read_file(path)
    wav, sr = tf.audio.decode_wav(audio_bytes, desired_channels=1)  # float32 [-1,1]
    wav = tf.squeeze(wav, axis=-1)
    sr = int(sr.numpy())
    if sr != SR:
        wav = _resample_1d_tf(wav, sr, SR)
    return wav

def compute_logmel(wav_f32_1d):
    feats = wav_to_logmels(tf.convert_to_tensor(wav_f32_1d, dtype=tf.float32))
    return feats.numpy()  # [T, n_mels]

def mel_bin_centers_hz(n_mels, n_fft, sr, lower_hz=20.0, upper_hz=None):
    """
    Use the actual TF mel weight matrix and take the FFT bin with the
    maximum weight for each mel filter as the 'center frequency'.
    """
    if upper_hz is None:
        upper_hz = sr / 2.0
    mel_fb = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_fft // 2 + 1,
        sample_rate=sr,
        lower_edge_hertz=lower_hz,
        upper_edge_hertz=upper_hz
    ).numpy()  # [num_spectrogram_bins, n_mels]
    k_max = np.argmax(mel_fb, axis=0)  # [n_mels]
    f_hz = (k_max.astype(np.float64) * sr) / float(n_fft)  # center freq per mel bin
    return f_hz  # [n_mels]

def make_secondary_freq_axis(ax, n_mels, f_hz,
                             fine_hz_max=2000.0,
                             major_step_low=250.0,
                             minor_step_low=50.0,
                             major_step_high=1000.0):
    """
    Add a right-side y-axis with frequency in Hz.
    Denser ticks from 0..fine_hz_max (e.g., 0..2000 Hz).
    """
    y_bins = np.arange(n_mels, dtype=np.float64)

    # Ensure strictly increasing (monotonic for interpolation)
    # (TF mel matrix yields monotonic centers already, but guard anyway)
    order = np.argsort(f_hz)
    f_hz_sorted = f_hz[order]
    y_bins_sorted = y_bins[order]

    # forward: bin_index -> Hz
    def bins_to_hz(y):
        return np.interp(y, y_bins_sorted, f_hz_sorted)

    # inverse: Hz -> bin_index
    def hz_to_bins(y_hz):
        return np.interp(y_hz, f_hz_sorted, y_bins_sorted)

    secax = ax.secondary_yaxis('right', functions=(bins_to_hz, hz_to_bins))
    secax.set_ylabel("Frequency (Hz)")

    # Build tick sets
    hz_min, hz_max = float(f_hz_sorted[0]), float(f_hz_sorted[-1])
    low_max = min(fine_hz_max, hz_max)

    # Major ticks: dense below fine_hz_max, coarse above
    major_low = np.arange(0.0, low_max + 1e-6, major_step_low)
    major_high_start = max(low_max, major_step_high)
    major_high = np.arange(major_high_start, hz_max + 1e-6, major_step_high)
    major_ticks = np.unique(np.concatenate([major_low, major_high]))
    major_ticks = major_ticks[(major_ticks >= hz_min) & (major_ticks <= hz_max)]

    # Minor ticks: dense only in the low region
    minor_ticks = np.arange(0.0, low_max + 1e-6, minor_step_low)
    minor_ticks = minor_ticks[(minor_ticks >= hz_min) & (minor_ticks <= hz_max)]

    # Apply ticks in secondary-axis units (Hz)
    secax.set_yticks(major_ticks)
    secax.set_yticks(minor_ticks, minor=True)
    secax.set_yticklabels([f"{int(tick):d}" for tick in major_ticks])

    # Optional: subtle grid aligned with major ticks on the right axis
    secax.grid(which="major", axis="y", linestyle="--", alpha=0.25)

    return secax

def save_melspec_png(logmel, out_png, cmap="magma", dpi=150,
                     add_colorbar=False, lower_hz=LOWER_HZ_DEFAULT, upper_hz=UPPER_HZ_DEFAULT,
                     n_fft=N_FFT, sr=SR,
                     fine_hz_max=2000.0, major_step_low=250.0, minor_step_low=50.0, major_step_high=1000.0):
    """
    logmel: [T, n_mels]
    """
    n_frames, n_mels = logmel.shape[0], logmel.shape[1]
    f_hz = mel_bin_centers_hz(n_mels, n_fft, sr, lower_hz, upper_hz)

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(logmel.T, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bin")

    if add_colorbar:
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    # Right-hand frequency axis in Hz with denser ticks below 2 kHz
    make_secondary_freq_axis(
        ax, n_mels, f_hz,
        fine_hz_max=fine_hz_max,
        major_step_low=major_step_low,
        minor_step_low=minor_step_low,
        major_step_high=major_step_high
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Render a log-mel spectrogram PNG with right-side Hz axis (dense 0–2 kHz).")
    ap.add_argument("wav", help="Input WAV (any sample rate; resampled to training SR).")
    ap.add_argument("--out", help="Output PNG path (default: <wav_basename>.png).")
    ap.add_argument("--cmap", default="magma", help="Matplotlib colormap (default: magma).")
    ap.add_argument("--dpi", type=int, default=150, help="Output PNG DPI (default: 150).")
    ap.add_argument("--colorbar", action="store_true", help="Add a colorbar.")
    ap.add_argument("--lower-hz", type=float, default=LOWER_HZ_DEFAULT, help="Lower mel edge (Hz).")
    ap.add_argument("--upper-hz", type=float, default=UPPER_HZ_DEFAULT, help="Upper mel edge (Hz).")
    ap.add_argument("--fine-max", type=float, default=2000.0, help="Dense tick region upper bound (Hz).")
    ap.add_argument("--major-low", type=float, default=250.0, help="Major tick step (Hz) in dense region.")
    ap.add_argument("--minor-low", type=float, default=50.0, help="Minor tick step (Hz) in dense region.")
    ap.add_argument("--major-high", type=float, default=1000.0, help="Major tick step (Hz) above dense region.")
    args = ap.parse_args()

    in_path = Path(args.wav).resolve()
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    out_path = Path(args.out).resolve() if args.out else in_path.with_suffix(".png")

    wav = load_mono_wav(str(in_path))
    logmel = compute_logmel(wav)
    save_melspec_png(
        logmel, str(out_path),
        cmap=args.cmap, dpi=args.dpi,
        add_colorbar=args.colorbar,
        lower_hz=args.lower_hz, upper_hz=args.upper_hz,
        n_fft=N_FFT, sr=SR,
        fine_hz_max=args.fine_max,
        major_step_low=args.major_low,
        minor_step_low=args.minor_low,
        major_step_high=args.major_high
    )

    dur_sec = float(tf.shape(wav)[0].numpy()) / SR
    print(f"Wrote {out_path}")
    print(f"SR: {SR} Hz | Duration: {dur_sec:.2f}s | Frames: {logmel.shape[0]} | Mel bins: {logmel.shape[1]}")

if __name__ == "__main__":
    main()
