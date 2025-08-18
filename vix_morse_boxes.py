# viz_greedy_spans.py
import argparse, os, sys, re
from pathlib import Path
import numpy as np
import tensorflow as tf

# Non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ------------------------------------------------------------
# Import your training module so we reuse EXACT constants & fns
# ------------------------------------------------------------
TRAIN_MOD = os.getenv("TRAIN_MOD", "morse_ctc_tpu")

def _import_training_module():
    try:
        return __import__(TRAIN_MOD, fromlist=["*"])
    except ModuleNotFoundError:
        # If this script lives in a subfolder, add repo root and retry
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        return __import__(TRAIN_MOD, fromlist=["*"])

training = _import_training_module()

# Pull what we need (use the module’s source of truth)
SR        = getattr(training, "SR")
N_FFT     = getattr(training, "N_FFT")
HOP       = getattr(training, "HOP")
N_MELS    = getattr(training, "N_MELS")
F_MIN     = getattr(training, "F_MIN")
F_MAX     = getattr(training, "F_MAX")
BLANK     = getattr(training, "BLANK_TOKEN")
ID2TOK    = getattr(training, "ID2TOK")
wav_to_logmels = getattr(training, "wav_to_logmels")
preprocess_wav_for_model = getattr(training, "preprocess_wav_for_model")
beam_ctc_decode = getattr(training, "beam_ctc_decode", None)
greedy_ctc_decode = getattr(training, "greedy_ctc_decode")
_resample_1d_tf = getattr(training, "_resample_1d_tf", None)

HOP_SEC = HOP / SR

# -----------------------
# I/O helpers
# -----------------------
def load_wav_mono_16k(path):
    audio_bytes = tf.io.read_file(path)
    audio, sr = tf.audio.decode_wav(audio_bytes, desired_channels=1)
    audio = tf.squeeze(audio, -1)  # [T]
    sr = int(sr.numpy())
    if sr != SR:
        if _resample_1d_tf is None:
            raise SystemExit(f"Expected {SR} Hz. Please resample: ffmpeg -i in.wav -ac 1 -ar {SR} out.wav")
        audio = _resample_1d_tf(audio, sr, SR)
    return audio.numpy()

# -----------------------
# Greedy framewise ids
# -----------------------
def greedy_decode_frames(logits_tf):
    """
    logits_tf: tf.Tensor [1, T, C]
    returns: ids [T], conf [T]
    """
    probs = tf.nn.softmax(logits_tf, axis=-1).numpy()[0]  # [T, C]
    ids   = probs.argmax(axis=-1).astype(np.int32)
    conf  = probs[np.arange(probs.shape[0]), ids]
    return ids, conf

def collapse_to_text_from_ids(ids):
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(ID2TOK.get(i, ""))
        prev = i
    return "".join(out)

def spans_from_ids(ids, min_len=1, prob=None, prob_thresh=0.0):
    """
    Build contiguous (start_frame, end_frame, char_id) spans for non-blank runs.
    Optional probability gating by avg conf within the span.
    """
    spans = []
    T = len(ids)
    prev = BLANK
    start = None
    for t in range(T):
        i = int(ids[t])
        if i != BLANK and (prev == BLANK or i != prev):
            start = t
        if (i == BLANK or i != prev) and prev != BLANK and start is not None:
            end = t
            if (end - start) >= min_len:
                if prob is not None and prob_thresh > 0.0:
                    if float(np.mean(prob[start:end])) >= prob_thresh:
                        spans.append((start, end, int(prev)))
                else:
                    spans.append((start, end, int(prev)))
            start = None
        if t == T - 1 and i != BLANK and start is not None:
            end = t + 1
            if (end - start) >= min_len:
                if prob is not None and prob_thresh > 0.0:
                    if float(np.mean(prob[start:end])) >= prob_thresh:
                        spans.append((start, end, int(i)))
                else:
                    spans.append((start, end, int(i)))
        prev = i
    return spans

# -----------------------
# Plotting
# -----------------------
def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def _mel_to_hz(m):
    return 700.0 * (10.0**(m / 2595.0) - 1.0)

def _hz_to_bin(hz, m_lo, m_hi, M):
    hz = np.clip(hz, F_MIN, F_MAX)
    m = _hz_to_mel(hz)
    return (m - m_lo) / (m_hi - m_lo) * (M - 1)

def draw_boxes_png(
    logmel_TxM, spans, out_png, ids_conf=None, title_txt="",
    label_pos="above", dpi=150, cmap="magma"
):
    """
    logmel_TxM: np.ndarray [T, M] (UN-normalized log-mel from training.wav_to_logmels)
    spans: list of (start_frame, end_frame, char_id)
    """
    T, M = logmel_TxM.shape
    extent = (0, T * HOP_SEC, 0, M)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.imshow(logmel_TxM.T, origin="lower", aspect="auto", extent=extent, cmap=cmap)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")

    # Boxes
    for (s, e, cid) in spans:
        x0 = s * HOP_SEC
        w  = (e - s) * HOP_SEC
        ax.add_patch(Rectangle((x0, 0), w, M, fill=False, linewidth=1.6, edgecolor="w"))

    # Labels
    for (s, e, cid) in spans:
        x0 = s * HOP_SEC
        w  = (e - s) * HOP_SEC
        label = ID2TOK.get(int(cid), "?")
        text  = label
        if ids_conf is not None and e > s:
            avgp = float(ids_conf[s:e].mean())
            text = f"{label} ({avgp:.2f})"

        if label_pos == "inside":
            ax.text(x0 + w/2, M * 0.95, text, ha="center", va="top", fontsize=9, color="w")
        elif label_pos == "below":
            ax.text(x0 + w/2, -0.10, text,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, clip_on=False,
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.5))
        else:  # "above"
            ax.text(x0 + w/2, 1.03, text,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, clip_on=False,
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.5))

    # Right-hand frequency axis (Hz) mapped to mel bins using the band [F_MIN, F_MAX]
    ax_r = ax.twinx()
    ax_r.set_ylim(0, M)
    lo_mel = _hz_to_mel(F_MIN)
    hi_mel = _hz_to_mel(F_MAX)

    # Dense ticks across 400–900 Hz (every 50 Hz)
    ticks_hz = np.arange(int(F_MIN), int(F_MAX) + 1, 50, dtype=int)
    tick_pos = [_hz_to_bin(h, lo_mel, hi_mel, M) for h in ticks_hz]
    ax_r.set_yticks(tick_pos)
    ax_r.set_yticklabels([str(h) for h in ticks_hz])
    ax_r.set_ylabel("Frequency (Hz)")

    # Title above everything
    if title_txt:
        fig.suptitle(title_txt, y=0.995)

    # Leave room for top labels/title if labels are above
    if label_pos == "above":
        fig.tight_layout(rect=[0, 0, 1, 0.84])
    elif label_pos == "below":
        fig.tight_layout(rect=[0, 0.10, 1, 1])
    else:
        fig.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Visualize CTC spans (greedy per-frame) on the log-mel spectrogram using the latest training model."
    )
    ap.add_argument("wav", help="Path to 16 kHz mono WAV (other rates will be resampled if training exposes _resample_1d_tf)")
    ap.add_argument("--model", default="artifacts/morse_ctc_model.keras", help="Model path")
    ap.add_argument("--out", default=None, help="Output PNG (default: artifacts/viz_boxes/<wav>.boxes.png)")
    ap.add_argument("--label-pos", choices=["above", "below", "inside"], default="above")
    ap.add_argument("--min-seg-frames", type=int, default=1, help="Minimum frames to keep a span")
    ap.add_argument("--prob-thresh", type=float, default=0.0, help="Drop spans with avg frame prob below this")
    ap.add_argument("--beam", action="store_true", help="Use lexicon-aware beam search for display text")
    ap.add_argument("--beam-width", type=int, default=12)
    ap.add_argument("--word-bonus", type=float, default=0.6)
    ap.add_argument("--callsign-bonus", type=float, default=0.9)
    args = ap.parse_args()

    # Load model & audio
    model = tf.keras.models.load_model(args.model, compile=False)
    wav = load_wav_mono_16k(args.wav)

    # Features for model inference
    feats = preprocess_wav_for_model(wav)       # [1, T, M] normalized
    logits = model(feats, training=False)       # [1, T, C]

    # Greedy per-frame ids for spans
    ids, conf = greedy_decode_frames(logits)
    spans = spans_from_ids(ids, min_len=args.min_seg_frames, prob=conf, prob_thresh=args.prob_thresh)

    # Text (beam or greedy-collapse)
    T = int(logits.shape[1])
    text = ""
    if args.beam and beam_ctc_decode is not None:
        # Use numpy logits for beam function
        decoded = beam_ctc_decode(logits.numpy(), [T],
                                  beam_width=args.beam_width,
                                  word_bonus=args.word_bonus,
                                  callsign_bonus=args.callsign_bonus)
        text = decoded[0] if decoded else ""
    else:
        # Greedy string from per-frame ids
        text = collapse_to_text_from_ids(ids)

    print("Decoded text:", text if text else "(empty)")
    print(f"Found {len(spans)} spans.")
    for (s, e, cid) in spans:
        print(f"[{s*HOP_SEC:7.3f}s – {(e)*HOP_SEC:7.3f}s] {ID2TOK.get(cid,'?')}  frames={e-s}")

    # Log-mel (UN-normalized) for plotting — use training’s exact function
    logmel = wav_to_logmels(tf.convert_to_tensor(wav, tf.float32)).numpy()  # [T, M]

    # Output
    if args.out:
        out_png = args.out
    else:
        base = os.path.splitext(os.path.basename(args.wav))[0]
        out_dir = os.path.join("artifacts", "viz_boxes")
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, base + ".boxes.png")

    title = f"{os.path.basename(args.wav)} | {text}"
    draw_boxes_png(logmel, spans, out_png, ids_conf=conf, title_txt=title, label_pos=args.label_pos)
    print("Wrote", out_png)
    print("NOTE: These are greedy CTC spans (time covers all mel bins), not object-detection boxes.")

if __name__ == "__main__":
    main()
