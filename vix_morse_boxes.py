import os, sys, argparse, importlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import tensorflow as tf

def load_train_mod(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        # if this viz script lives in a subdir, try repo root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, repo_root)
        return importlib.import_module(name)

def rle_segments(ids, blank_id=0, min_len=3):
    """
    Run-length segments of non-blank classes.
    Returns list of (start_idx, end_idx_inclusive, class_id).
    """
    segs = []
    if len(ids) == 0: 
        return segs
    prev = ids[0]
    start = 0
    for i in range(1, len(ids)):
        if ids[i] != prev:
            if prev != blank_id:
                if i - 1 - start + 1 >= min_len:
                    segs.append((start, i - 1, int(prev)))
            start = i
            prev = ids[i]
    # tail
    if prev != blank_id and (len(ids) - 1 - start + 1) >= min_len:
        segs.append((start, len(ids) - 1, int(prev)))
    return segs

def dominant_band(logmel, t0, t1, pad=2):
    """
    Find dominant mel bin in frames [t0,t1] and return (y0,height) for a tight band box.
    """
    # logmel: [T, M]
    M = logmel.shape[1]
    slab = logmel[t0:t1+1, :]          # [W, M]
    prof = slab.mean(axis=0)           # [M]
    k = int(np.argmax(prof))
    y0 = max(0, k - pad)
    y1 = min(M - 1, k + pad)
    return y0, (y1 - y0 + 1)

def main():
    ap = argparse.ArgumentParser(
        description="Draw CTC-derived 'bounding boxes' for decoded Morse on a log-mel spectrogram."
    )
    ap.add_argument("wav", help="Input 16 kHz mono WAV")
    ap.add_argument("--model", default="artifacts/morse_ctc_model.keras", help="Keras model path")
    ap.add_argument("--train-mod", default="morse_ctc_tpu",
                    help="Python module name of your training script (to reuse wav_to_logmels, constants, ID2TOK)")
    ap.add_argument("--out", default=None, help="Output PNG path (default: <wav>.boxes.png)")
    ap.add_argument("--min-seg-frames", type=int, default=3, help="Ignore tiny segments shorter than this")
    ap.add_argument("--full-height", action="store_true",
                    help="Draw boxes full-height instead of a tight tone band")
    args = ap.parse_args()

    # --- load your training module for consistent preprocessing ---
    T = load_train_mod(args.train_mod)

    wav_to_logmels = getattr(T, "wav_to_logmels")
    SR    = getattr(T, "SR", 16000)
    HOP   = getattr(T, "HOP", 256)
    N_MELS= getattr(T, "N_MELS", 64)
    ID2TOK= getattr(T, "ID2TOK", None)
    BLANK = getattr(T, "BLANK_TOKEN", 0)

    # --- load audio ---
    audio_bytes = tf.io.read_file(args.wav)
    audio, sr = tf.audio.decode_wav(audio_bytes)  # [N,1]
    audio = tf.squeeze(audio, -1)                 # [N]
    if int(sr.numpy()) != SR:
        raise SystemExit(f"Expected {SR} Hz, got {int(sr.numpy())} Hz. Please resample.")

    # --- compute log-mel (un-normalized) for plotting ---
    logmel = wav_to_logmels(audio).numpy()        # [T, M]
    Tframes, Mbins = logmel.shape

    # --- model features (z-norm like training) ---
    feats = tf.convert_to_tensor(logmel, tf.float32)
    m = tf.reduce_mean(feats); s = tf.math.reduce_std(feats) + 1e-6
    feats = (feats - m) / s
    feats = feats[None, ...]                      # [1, T, M]

    # --- load model and run ---
    model = tf.keras.models.load_model(args.model, compile=False)
    logits = model(feats, training=False).numpy()[0]  # [T, C]
    # framewise class & prob
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    cls   = probs.argmax(axis=-1)                  # [T]

    # --- merge into time segments for non-blank IDs ---
    segs = rle_segments(cls, blank_id=BLANK, min_len=args.min_seg_frames)

    # --- plot spectrogram ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(logmel.T, origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bin")

    # Optional right Y axis in Hz like your previous viz
    def mel_y_ticks_to_hz(ax_right):
        # crude reference labels; same as earlier helper you used
        ticks = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60]   # mel bins
        # map bins to linear freq approx: project mel bin centers back to Hz using SR/4 upper edge
        # Here we’ll just annotate helpful values (visual guide)
        ax_right.set_yticks(ticks)
        labels = [""]*len(ticks)
        for i, b in enumerate(ticks):
            labels[i] = ""  # keep empty unless you want to compute exact inversion
        ax_right.set_yticklabels(labels)
        ax_right.set_ylabel("Frequency (Hz)")
        ax_right.grid(False)

    # --- draw boxes ---
    for (t0, t1, cid) in segs:
        char = ID2TOK.get(int(cid), str(int(cid))) if ID2TOK else str(int(cid))
        conf = probs[t0:t1+1, cid].mean()

        if args.full_height:
            y0, h = 0, Mbins
        else:
            y0, h = dominant_band(logmel, t0, t1, pad=2)

        rect = patches.Rectangle(
            (t0, y0), (t1 - t0 + 1), h,
            linewidth=1.5, edgecolor="cyan", facecolor="none", alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(t0 + 1, y0 + h + 1, f"{char} ({conf:.2f})",
                color="cyan", fontsize=9, va="bottom", ha="left")

    plt.tight_layout()
    out_png = args.out or (os.path.splitext(args.wav)[0] + ".boxes.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Wrote {out_png}")
    print(f"Segments: {[(t0, t1, (ID2TOK.get(c, str(c)) if ID2TOK else str(c))) for t0,t1,c in segs]}")
    print("NOTE: These are CTC-aligned character spans, not object-detection boxes.")
    
if __name__ == "__main__":
    main()
