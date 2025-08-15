# save as viz_greedy_spans.py
import argparse, os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- same alphabet & constants as your infer_morse.py ---
MORSE_TABLE = {'A':'.-','B':'-...','C':'-.-.','D':'-..','E':'.','F':'..-.','G':'--.',
'H':'....','I':'..','J':'.---','K':'-.-','L':'.-..','M':'--','N':'-.','O':'---',
'P':'.--.','Q':'--.-','R':'.-.','S':'...','T':'-','U':'..-','V':'...-','W':'.--',
'X':'-..-','Y':'-.--','Z':'--..','0':'-----','1':'.----','2':'..---','3':'...--',
'4':'....-','5':'.....','6':'-....','7':'--...','8':'---..','9':'----.','.':' .-.-.-',
',':'--..--','?':'..--..','/':'-..-.','-':'-....-','(':'-.--. ',')':'-.--.-','@':'.--.-.',
':':'---...',"\'":'.----.','!':'-.-.--','&':'.-...',';':'-.-.-.','=':'-...-','+':'.-.-.'}
ALPHABET = sorted(set(list(MORSE_TABLE.keys()) + [' ']))
BLANK = 0
ID2TOK = {i+1: ch for i, ch in enumerate(ALPHABET)}  # 1..N

SR, N_FFT, HOP, N_MELS = 16000, 1024, 256, 64
HOP_SEC = HOP / SR

# Build mel filterbank once so it matches training/infer script
MEL_FB = tf.constant(tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=N_MELS, num_spectrogram_bins=N_FFT//2+1,
    sample_rate=SR, lower_edge_hertz=20.0, upper_edge_hertz=SR/2.0
))

def wav_to_logmels(wav):
    stft = tf.signal.stft(wav, frame_length=N_FFT, frame_step=HOP, fft_length=N_FFT)
    mag = tf.abs(stft)
    mel = tf.matmul(tf.square(mag), MEL_FB)
    return tf.math.log(mel + 1e-6)  # [T, M]

def preprocess(wav_f32):
    feats = wav_to_logmels(tf.convert_to_tensor(wav_f32, tf.float32))
    m, s = tf.reduce_mean(feats), tf.math.reduce_std(feats) + 1e-6
    return ((feats - m) / s)[None, ...]  # [1,T,M]

def load_wav(path):
    audio_bytes = tf.io.read_file(path)
    audio, sr = tf.audio.decode_wav(audio_bytes)
    audio = tf.squeeze(audio, -1)  # mono
    if int(sr.numpy()) != SR:
        raise SystemExit("Please resample to 16 kHz: ffmpeg -i in.wav -ac 1 -ar 16000 out.wav")
    return audio.numpy()

def greedy_decode_frames(logits):
    """
    logits: tf.Tensor [1, T, C]
    Returns:
      ids: np.ndarray [T] argmax class per frame
      probs: np.ndarray [T] max softmax prob per frame
    """
    logits_np = logits.numpy()
    # softmax for confidence (argmax would be same on logits)
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]  # [T, C]
    ids   = probs.argmax(axis=-1)                      # [T]
    conf  = probs[np.arange(probs.shape[0]), ids]      # [T]
    return ids, conf

def collapse_to_text(ids):
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(ID2TOK.get(i, ''))
        prev = i
    return ''.join(out)

def spans_from_ids(ids, min_len=1):
    """
    Build contiguous (start_frame, end_frame, char_id) for non-blank runs.
    min_len: minimum frames to keep a span.
    """
    spans = []
    T = len(ids)
    prev = BLANK
    start = None
    for t in range(T):
        i = int(ids[t])
        if i != BLANK and (prev == BLANK or i != prev):
            # start new span
            start = t
        if i == BLANK and prev != BLANK and start is not None:
            if (t - start) >= min_len:
                spans.append((start, t, int(prev)))
            start = None
        # handle last frame
        if t == T-1 and i != BLANK and start is not None:
            if (t+1 - start) >= min_len:
                spans.append((start, t+1, int(i)))
        prev = i
    return spans

def draw_boxes_png(logmel, spans, out_png, ids_conf=None, title_txt="", label_pos="above"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import os, numpy as np

    T, M = logmel.shape
    extent = (0, T*HOP_SEC, 0, M)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.imshow(logmel.T, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")

    # Boxes
    for (s, e, cid) in spans:
        x0 = s * HOP_SEC
        w  = (e - s) * HOP_SEC
        ax.add_patch(Rectangle((x0, 0), w, M, fill=False, linewidth=1.8))

    # Labels (above/below/inside)
    for (s, e, cid) in spans:
        x0 = s * HOP_SEC
        w  = (e - s) * HOP_SEC
        label = ID2TOK.get(int(cid), "?")
        text  = label
        if ids_conf is not None and e > s:
            avgp = float(ids_conf[s:e].mean())
            text = f"{label} ({avgp:.2f})"

        if label_pos == "inside":
            ax.text(x0 + w/2, M*0.95, text, ha="center", va="top", fontsize=9)
        elif label_pos == "below":
            ax.text(x0 + w/2, -0.08, text,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, clip_on=False,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
        else:  # "above"
            ax.text(x0 + w/2, 1.02, text,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, clip_on=False,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))

    # >>> Move title ABOVE the axes/labels <<<
    if title_txt:
        fig.suptitle(title_txt, y=0.99)  # higher than the axis area & labels

    # Leave space so labels/title aren’t cropped
    if label_pos == "above":
        fig.tight_layout(rect=[0, 0, 1, 0.86])   # more top room
    elif label_pos == "below":
        fig.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        fig.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize greedy CTC spans on the log-mel spectrogram.")
    ap.add_argument("--model", default="artifacts/morse_ctc_model.keras")
    ap.add_argument("--out", default=None, help="Output PNG path (default: WAV.png next to input)")
    ap.add_argument("--min-seg-frames", type=int, default=1, help="Minimum frames per character span")
    ap.add_argument("wav")
    args = ap.parse_args()

    # Load model and audio
    model = tf.keras.models.load_model(args.model, compile=False)
    wav = load_wav(args.wav)

    # Features for model
    feats = preprocess(wav)              # [1, T, M]
    logits = model(feats, training=False)  # [1, T, C]

    # Greedy per-frame ids (no prob threshold)
    ids, conf = greedy_decode_frames(logits)
    text = collapse_to_text(ids)
    print("Decoded text:\n", text)

    # Spans from ids
    spans = spans_from_ids(ids, min_len=args.min_seg_frames)
    print(f"Found {len(spans)} spans.")
    for (s,e,cid) in spans:
        t0, t1 = s*HOP_SEC, e*HOP_SEC
        print(f"[{t0:7.3f}s – {t1:7.3f}s] {ID2TOK.get(cid,'?')}  frames={e-s}")

    # Log-mel for plotting (un-normalized like infer script)
    feats_plot = wav_to_logmels(tf.convert_to_tensor(wav, tf.float32)).numpy()  # [T, M]

    # Output path
    if args.out:
        out_png = args.out
    else:
        base = os.path.splitext(os.path.basename(args.wav))[0]
        out_dir = os.path.join("artifacts", "viz_boxes")
        out_png = os.path.join(out_dir, base + ".boxes.png")

    title = f"{os.path.basename(args.wav)} | {text}"
    draw_boxes_png(feats_plot, spans, out_png, ids_conf=conf, title_txt=title)
    print("Wrote", out_png)

if __name__ == "__main__":
    main()
