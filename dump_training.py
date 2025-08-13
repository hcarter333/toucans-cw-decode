# dump_training.py
import os, csv, random, argparse, re, sys
from pathlib import Path
import numpy as np

# Use a non-interactive backend so saving PNGs works everywhere
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Locate & import your training module ----------
TRAIN_MOD = os.getenv("TRAIN_MOD", "morse_ctc_tpu")

def _import_training_module():
    try:
        return __import__(TRAIN_MOD, fromlist=["*"])
    except ModuleNotFoundError:
        # If this file lives in a subfolder (e.g., training_samples/), add repo root
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        return __import__(TRAIN_MOD, fromlist=["*"])

training = _import_training_module()

# Pull what we need, with safe fallbacks
synth_morse_audio_from_text = getattr(training, "synth_morse_audio_from_text")
random_ham_text = getattr(training, "random_ham_text", None)    # phrase-aware (preferred)
random_text     = getattr(training, "random_text", None)        # fallback
wav_to_logmels  = getattr(training, "wav_to_logmels")
SR              = getattr(training, "SR", 16000)
F_MIN           = getattr(training, "F_MIN", 400.0)
F_MAX           = getattr(training, "F_MAX", 900.0)

# ---------- Helpers ----------
def save_wav_int16(path, wav_f32, sr=SR):
    """Save mono float32 [-1,1] array as 16-bit PCM WAV."""
    import wave
    wav = np.asarray(wav_f32, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    pcm16 = (wav * 32767.0).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

def save_logmel_png(path, wav_f32, cmap="magma"):
    """Compute log-mel with the SAME routine as training and write a PNG."""
    import tensorflow as tf
    x = tf.convert_to_tensor(wav_f32, dtype=tf.float32)
    logmel = wav_to_logmels(x).numpy()          # [T, N_MELS]
    # Plot as mel (y) x time (x)
    plt.figure(figsize=(8, 3))
    # transpose so mel bins go on Y axis; origin='lower' puts low freqs at bottom
    plt.imshow(logmel.T, origin="lower", aspect="auto", cmap=cmap)
    plt.xlabel("Frames")
    plt.ylabel("Mel bin")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

def slugify(s, maxlen=48):
    return re.sub(r"[^A-Za-z0-9]+", "_", s.strip()).strip("_")[:maxlen] or "sample"

def pick_text(min_len, max_len, use_phrase_sampler=True):
    if use_phrase_sampler and random_ham_text is not None:
        return random_ham_text(min_len=min_len, max_len=max_len, p_phrase=0.35)
    if random_text is not None:
        return random_text(min_len=min_len, max_len=max_len)
    # last-resort fallback: letters/numbers and spaces
    import string as _s
    alphabet = _s.ascii_uppercase + "0123456789 "
    L = random.randint(min_len, max_len)
    return "".join(random.choices(alphabet, k=L)).strip()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Dump synthetic Morse training WAVs + log-mel PNGs.")
    ap.add_argument("--out", default="artifacts/sanity_wavs", help="Output directory")
    ap.add_argument("--num", type=int, default=10, help="How many samples to generate")
    ap.add_argument("--fixed-text", default=None,
                    help="If set, use this exact text for every sample (else generate)")
    ap.add_argument("--min-len", type=int, default=5, help="Min random text length")
    ap.add_argument("--max-len", type=int, default=24, help="Max random text length")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    ap.add_argument("--no-phrases", action="store_true",
                    help="Disable phrase-aware sampler; use plain random text")
    ap.add_argument("--cmap", default="magma", help="Matplotlib colormap for spectrogram")
    args = ap.parse_args()

    #random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    manifest_path = os.path.join(args.out, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["filename_wav", "filename_png", "text", "wpm", "jitter",
                    "fwobble", "snr_db", "base_freq_hz", "samples", "sr"])

        for i in range(1, args.num + 1):
            # Match your training-time augmentation ranges
            wpm     = random.uniform(17, 20)
            fwobble = random.uniform(0.0, 0.02)
            jitter  = random.uniform(0.05, 0.15)
            snr_db  = random.uniform(0.0, 24.0)
            basef   = random.uniform(F_MIN, F_MAX)

            text = args.fixed_text if args.fixed_text else pick_text(
                args.min_len, args.max_len, use_phrase_sampler=(not args.no_phrases)
            )

            wav  = synth_morse_audio_from_text(
                text, wpm=wpm, fwobble=fwobble, jitter=jitter, snr_db=snr_db, base_freq=basef
            )

            base = f"{i:03d}_{slugify(text)}_wpm{int(round(wpm))}_snr{int(round(snr_db))}"
            wav_path = os.path.join(args.out, base + ".wav")
            png_path = os.path.join(args.out, base + ".png")

            save_wav_int16(wav_path, wav, sr=SR)
            save_logmel_png(png_path, wav, cmap=args.cmap)

            w.writerow([os.path.basename(wav_path), os.path.basename(png_path), text,
                        f"{wpm:.3f}", f"{jitter:.3f}", f"{fwobble:.3f}",
                        f"{snr_db:.3f}", f"{basef:.1f}", len(wav), SR])

            print(f"Wrote {wav_path} and {png_path}")

    print(f"\nManifest: {manifest_path}")

if __name__ == "__main__":
    main()
