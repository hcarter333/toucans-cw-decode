import math, random, string, os, io
import numpy as np
import tensorflow as tf

model = None
optimizer = None

PHRASES = [
    "CQ CQ", "CQ CQ CQ", "DE", "QRP", "QRO", "POTA", "SOTA", "5NN", "K", "BK", "KN", "SK"
]
KEEP_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")  

def random_callsign():
    import random, string
    # Very rough US-style: 1–2 prefix letters, a digit, 1–3 suffix letters
    prefix = "".join(random.choices(string.ascii_uppercase, k=random.choice([1,2])))
    digit  = random.choice("0123456789")
    suffix = "".join(random.choices(string.ascii_uppercase, k=random.choice([2,3])))
    return prefix + digit + suffix

def random_ham_text(min_len=5, max_len=24, p_phrase=0.35):
    if random.random() < p_phrase:
        # Sample a typical exchange
        cs1, cs2 = random_callsign(), random_callsign()
        templates = [
            f"CQ CQ DE {cs1} K",
            f"{cs1} DE {cs2} K",
            f"{cs1} DE {cs2} 5NN",
            f"{cs1} QRP", f"{cs1} POTA", f"{cs1} SOTA",
            random.choice(PHRASES)
        ]
        s = random.choice(templates)
    else:
        # fallback: uniform random over KEEP_CHARS
        L = random.randint(min_len, max_len)
        s = "".join(random.choices(KEEP_CHARS, k=L))
    # collapse excessive spaces and trim
    return " ".join(s.split())





# =========================
# 0) TPU / Strategy setup
# =========================
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # Detect TPU
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    STRATEGY = tf.distribute.TPUStrategy(tpu)
    print(">> Using TPU")
except Exception as e:
    STRATEGY = tf.distribute.get_strategy()
    print(">> Using default strategy (CPU/GPU).", str(e)[:120])

AUTOTUNE = tf.data.AUTOTUNE

# =========================
# 1) Morse tables & helpers
# =========================
MORSE_TABLE = {
    'A': '.-',     'B': '-...',   'C': '-.-.',  'D': '-..',  'E': '.',
    'F': '..-.',   'G': '--.',    'H': '....',  'I': '..',   'J': '.---',
    'K': '-.-',    'L': '.-..',   'M': '--',    'N': '-.',   'O': '---',
    'P': '.--.',   'Q': '--.-',   'R': '.-.',   'S': '...',  'T': '-',
    'U': '..-',    'V': '...-',   'W': '.--',   'X': '-..-', 'Y': '-.--',
    'Z': '--..',
    '0': '-----',  '1': '.----',  '2': '..---', '3': '...--', '4': '....-',
    '5': '.....',  '6': '-....',  '7': '--...', '8': '---..', '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', '/': '-..-.', '-': '-....-',
    '(': '-.--.',  ')': '-.--.-', '@': '.--.-.', ':': '---...', "'": '.----.',
    '!': '-.-.--', '&': '.-...',  ';': '-.-.-.', '=': '-...-', '+': '.-.-.'
}

# Allowed output alphabet (include blank for CTC)
ALPHABET = list(sorted(set(list(MORSE_TABLE.keys()) + [' '])))

# Map chars <-> ids (id 0 reserved for CTC blank)
BLANK_TOKEN = 0
TOK2ID = {ch: i+1 for i, ch in enumerate(ALPHABET)}  # 1..N
ID2TOK = {i+1: ch for i, ch in enumerate(ALPHABET)}

# =========================
# 2) Audio synthesis (NumPy)
# =========================
SR = 16000  # sample rate
F_MIN, F_MAX = 400.0, 900.0   # sidetone range for augmentation

def wpm_to_unit_seconds(wpm):
    # Standard PARIS definition: one word = 50 dit units
    # dit duration (sec) = 1.2 / wpm (common approximation)
    return 1.2 / wpm

import numpy as np, math

def render_tone(freq, duration_s, amp=0.8, phase=0.0,
                ramp_ms=2.0, fm_dev_hz=0.0, fm_rate_hz=0.0):
    """Sine with short ramps + optional slow FM warble."""
    N = int(SR * duration_s)
    t = np.arange(N, dtype=np.float32) / SR

    # envelope (shorter ramp to allow a bit more 'click')
    env = np.ones(N, np.float32)
    r = max(1, int(ramp_ms * 1e-3 * SR))
    if N >= 2*r:
        rwin = np.linspace(0, math.pi/2, r, dtype=np.float32)
        env[:r]  = np.sin(rwin)**2
        env[-r:] = np.sin(rwin[::-1])**2

    if fm_dev_hz > 0.0 and fm_rate_hz > 0.0:
        phi0 = np.random.uniform(0, 2*np.pi)
        inst_freq = freq + fm_dev_hz * np.sin(2*np.pi*fm_rate_hz*t + phi0)
        phase_acc = 2*np.pi*np.cumsum(inst_freq)/SR + phase
        sig = amp * env * np.sin(phase_acc)
    else:
        sig = amp * env * np.sin(2*np.pi*freq*t + phase)
    return sig.astype(np.float32)

def apply_qsb(sig, depth_db=3.0, rate_hz=0.6):
    """Slow amplitude fading (QSB)."""
    if depth_db <= 0 or rate_hz <= 0: return sig
    N = sig.shape[0]
    t = np.arange(N, dtype=np.float32)/SR
    phi = np.random.uniform(0, 2*np.pi)
    # convert depth in dB to linear swing around 1.0
    a = 10**(depth_db/20.0) - 1.0     # e.g., 3 dB ≈ +0.412
    env = 1.0 + 0.5*a*np.sin(2*np.pi*rate_hz*t + phi)
    return (sig * env).astype(np.float32)

def render_silence(duration_s):
    return np.zeros(int(SR*duration_s), dtype=np.float32)

def text_to_morse(s):
    out = []
    for ch in s.upper():
        if ch == ' ':
            out.append(' ')  # word gap marker
        elif ch in MORSE_TABLE:
            out.append(MORSE_TABLE[ch])
    return out

def synth_morse_audio_from_text(
    text, wpm=20, fwobble=0.0, jitter=0.1, snr_db=15.0, base_freq=600.0
):
    """
    text -> morse -> waveform with realistic timing.

    Timing in 'units':
      dit = 1
      dah = 3
      gap between elements (within char) = 1
      gap between letters = 3
      gap between words = 7
    We’ll draw unit_seconds from WPM, add jitter to each unit, and slight freq wobble.
    """
    unit = wpm_to_unit_seconds(wpm)
    # slight per-character frequency wobble
    freq = base_freq

    morse_seq = text_to_morse(text)
    audio = []

    for token in morse_seq:
        if token == ' ':  # word gap
            audio.append(render_silence(7*unit))
            continue

        # token is like ".-."
        for i, sym in enumerate(token):
            dur = (1 if sym=='.' else 3) * unit * np.random.uniform(1.0-jitter, 1.0+jitter)
            f   = base_freq * (1.0 + np.random.uniform(-fwobble, fwobble))
            # slow FM warble ~0.2–2 Hz, ±8–15 Hz
            fm_rate = np.random.uniform(0.2, 2.0)
            fm_dev  = np.random.uniform(8.0, 15.0)
            audio.append(render_tone(f, dur, ramp_ms=2.0, fm_dev_hz=fm_dev, fm_rate_hz=fm_rate))
            if i != len(token)-1:
                gap = unit * np.random.uniform(1.0-jitter, 1.0+jitter)
                audio.append(render_silence(gap))
        gap = 3*unit * np.random.uniform(1.0-jitter, 1.0+jitter)
        audio.append(render_silence(gap))

    sig = np.concatenate(audio) if audio else np.zeros(1, np.float32)

    # QSB fading 0.1–1.0 Hz, 1–6 dB
    sig = apply_qsb(sig,
        depth_db=np.random.uniform(1.0, 6.0),
        rate_hz=np.random.uniform(0.1, 1.0)
    )

    # Add band-limited-ish noise (white + mild HP/LP)
    if snr_db is not None:
        noise = np.random.normal(0, 1, size=sig.shape).astype(np.float32)
        # crude 1st-order HP+LP (two passes)
        def onepole_lp(x, alpha=0.05):
            y = np.zeros_like(x)
            v = 0.0
            for i, xi in enumerate(x):
                v = v + alpha*(xi - v)
                y[i] = v
            return y
        def onepole_hp(x, alpha=0.05):
            return x - onepole_lp(x, alpha)

        noise = onepole_lp(onepole_hp(noise, 0.02), 0.1)
        # scale noise for desired SNR
        s_pow = np.mean(sig**2) + 1e-9
        n_pow = np.mean(noise**2) + 1e-9
        target_n_pow = s_pow / (10**(snr_db/10.0))
        noise *= math.sqrt(target_n_pow / n_pow)
        sig = sig + noise

    # Normalize to [-1, 1]
    peak = np.max(np.abs(sig)) + 1e-6
    sig = (sig / peak).astype(np.float32)
    return sig

# =========================
# 3) Feature extraction (tf)
# =========================
N_FFT   = 1024
HOP     = 256
N_MELS  = 64

def wav_to_logmels(wav):
    """
    wav: 1D float32 Tensor
    returns: [T, N_MELS] float32 log-mel spectrogram
    """
    stft = tf.signal.stft(wav, frame_length=N_FFT, frame_step=HOP, fft_length=N_FFT)
    mag = tf.abs(stft)  # [frames, bins]
    mel_fb = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT//2 + 1,
        sample_rate=SR, lower_edge_hertz=F_MIN, upper_edge_hertz=F_MAX
    )
    mel = tf.matmul(tf.square(mag), mel_fb)  # power mel
    eps = tf.constant(1e-6, tf.float32)
    logmel = tf.math.log(mel + eps)
    return logmel

# =========================
# 4) Data generator
# =========================
def random_text(min_len=5, max_len=24):
    choices = list(MORSE_TABLE.keys()) + [' ']
    L = random.randint(min_len, max_len)
    # Avoid sequences of spaces and leading/trailing spaces
    s = []
    last_space = True
    for _ in range(L):
        ch = random.choice(choices)
        if ch == ' ' and last_space:
            ch = random.choice(list(MORSE_TABLE.keys()))
        last_space = (ch == ' ')
        s.append(ch)
    if s[0] == ' ': s[0] = random.choice(list(MORSE_TABLE.keys()))
    if s[-1] == ' ': s[-1] = random.choice(list(MORSE_TABLE.keys()))
    return ''.join(s)

def encode_text_to_ids(s):
    return np.array([TOK2ID[ch] for ch in s if ch in TOK2ID], dtype=np.int32)

def sample_example():
    # Draw augmentation params
    wpm   = random.uniform(12, 28)
    fwob  = random.uniform(0.0, 0.02)
    jitter= random.uniform(0.05, 0.15)
    snr   = random.uniform(0.0, 24.0)
    basef = random.uniform(F_MIN, F_MAX)

    text  = random_ham_text()
    wav   = synth_morse_audio_from_text(text, wpm=wpm, fwobble=fwob, jitter=jitter,
                                        snr_db=snr, base_freq=basef)
    feats = wav_to_logmels(tf.convert_to_tensor(wav))
    # Normalize per-utterance
    m = tf.reduce_mean(feats)
    s = tf.math.reduce_std(feats) + 1e-6
    feats = (feats - m) / s
    # return features [T, M] and label ids
    return feats.numpy().astype(np.float32), encode_text_to_ids(text)

def make_dataset(num_items=5000, max_timesteps=2000):
    def gen():
        for _ in range(num_items):
            feats, ids = sample_example()
            # Clip very long to keep padding modest
            if feats.shape[0] > max_timesteps:
                feats = feats[:max_timesteps]
            yield feats, ids

    output_sig = (
        tf.TensorSpec(shape=(None, N_MELS), dtype=tf.float32), # variable T
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)

    # Pad to batches
    def pad_batch(feats, ids):
        feat_len = tf.shape(feats)[0]
        label_len = tf.shape(ids)[0]
        return {
            "feats": feats,
            "feat_len": feat_len,
            "labels": ids,
            "label_len": label_len
        }

    ds = ds.map(pad_batch, num_parallel_calls=AUTOTUNE)

    def pack_batch(batch):
        # dynamic pad
        feats = batch["feats"]
        labels = batch["labels"]
        return (feats, labels,
                batch["feat_len"], batch["label_len"])

    return ds, pack_batch

BATCH_SIZE = 32

def collate(batches):
    feats, labels, feat_lens, label_lens = zip(*batches)
    feats = tf.ragged.stack(feats).to_tensor()  # [B, T, M]
    labels = tf.ragged.stack(labels).to_tensor()  # [B, L]
    feat_lens = tf.stack(feat_lens)
    label_lens = tf.stack(label_lens)
    return (feats, labels, feat_lens, label_lens)

MAX_T = 2000  # keep your dataset clipping consistent with this

def to_tf_dataset(ds, batch_size):
    ds = ds.padded_batch(
        batch_size,
        padded_shapes={
            "feats": [None, N_MELS],   # <- variable T per batch
            "labels": [None],
            "feat_len": [],
            "label_len": []
        },
        drop_remainder=True
    ).map(lambda d: (d["feats"], d["labels"], d["feat_len"], d["label_len"]),
          num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds
    
# =========================
# 5) Model (CRNN + CTC)
# =========================
def build_model(n_classes):
    inp = tf.keras.Input(shape=(None, N_MELS), name="feats")  # [T, M]
    x = inp

    # TimeConv via Conv1D on the feature axis (treat time as steps)
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    # BiLSTM to model temporal patterns
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    )(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    # Time-distributed logits
    logits = tf.keras.layers.Dense(n_classes + 1)(x)  # +1 for CTC blank
    return tf.keras.Model(inp, logits, name="morse_ctc")

@tf.function
def ctc_loss(logits, labels, logit_lens, label_lens):
    """
    logits: [B, T, C] (unnormalized)
    labels: [B, L] int ids (1..C-1), blank=0 reserved
    logit_lens: [B] actual lengths in time steps
    label_lens: [B] actual label lengths
    """
    logit_len32 = tf.cast(logit_lens, tf.int32)
    label_len32 = tf.cast(label_lens, tf.int32)
    # CTC expects time major: [T, B, C]
    logits_tm = tf.transpose(logits, [1, 0, 2])
    loss = tf.nn.ctc_loss(
        labels=labels, logits=logits_tm,
        label_length=label_len32,
        logit_length=logit_len32,
        logits_time_major=True,
        blank_index=BLANK_TOKEN
    )
    return tf.reduce_mean(loss)

def greedy_ctc_decode(logits, logit_lens):
    # logits: [B, T, C], return list of strings
    probs = tf.nn.log_softmax(logits, axis=-1)
    preds = tf.argmax(probs, axis=-1).numpy()  # [B, T]
    out = []
    for b in range(preds.shape[0]):
        T = int(logit_lens[b])
        seq = preds[b, :T]
        # collapse repeats + remove blanks (0)
        decoded_ids = []
        prev = -1
        for t in seq:
            if t != prev:
                if t != BLANK_TOKEN:
                    decoded_ids.append(int(t))
            prev = int(t)
        # map to chars
        s = ''.join(ID2TOK[i] for i in decoded_ids if i in ID2TOK)
        out.append(s)
    return out


# before: @tf.function def train_step(batch): ...
@tf.function(reduce_retracing=True)
def train_step(feats, labels, feat_lens, label_lens):
    with tf.GradientTape() as tape:
        logits = model(feats, training=True)
        logit_lens = tf.identity(feat_lens)   # <-- per-example T
        loss = ctc_loss(logits, labels, logit_lens, label_lens)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def dataset_pipeline(num_train=8000, num_val=800):
    train_ds, _ = make_dataset(num_train)
    val_ds, _   = make_dataset(num_val)

    train_ds = to_tf_dataset(train_ds, BATCH_SIZE)
    val_ds   = to_tf_dataset(val_ds, BATCH_SIZE)
    return train_ds, val_ds
    

# =========================
# 7) Inference helpers
# =========================
def preprocess_wav_for_model(wav_f32):
    # mono float32 [-1,1] → log-mel T x M → z-norm
    x = tf.convert_to_tensor(wav_f32, dtype=tf.float32)
    feats = wav_to_logmels(x)
    m = tf.reduce_mean(feats)
    s = tf.math.reduce_std(feats) + 1e-6
    feats = (feats - m) / s
    return feats[None, ...]  # [1, T, M]

def decode_wav(wav_bytes):
    wav, sr = tf.audio.decode_wav(wav_bytes)
    wav = tf.squeeze(wav, axis=-1)  # mono
    if sr != SR:
        wav = tfio.audio.resample(wav, rate_in=tf.cast(sr, tf.int64), rate_out=SR)  # if using tfio
        # If tfio not available, prefer to train/infer at SR and pre-resample offline.
    feats = preprocess_wav_for_model(wav.numpy())
    logits = model(feats, training=False).numpy()
    dec = greedy_ctc_decode(tf.convert_to_tensor(logits), [logits.shape[1]])
    return dec[0]

print("Training complete. Try inference by loading a wav and calling decode_wav().")



# =========================
# 6) Training loop
# =========================
def main():
    global model, optimizer   
    with STRATEGY.scope():
        # --- Option A: resume from saved .keras (fresh optimizer state) ---
        default_resume = "artifacts/morse_ctc_model.keras"
        resume_path = os.getenv("RESUME_MODEL")
        if not resume_path and os.path.exists(default_resume):
            resume_path = default_resume
    
        if resume_path and os.path.exists(resume_path):
            print(f">> Resuming from {resume_path}")
            model = tf.keras.models.load_model(resume_path, compile=False)
        else:
            model = build_model(n_classes=len(ALPHABET))
            if resume_path:
                print(f">> Requested resume path not found: {resume_path} — training from scratch.")
    
        optimizer = tf.keras.optimizers.Adam(1e-3)
    
        
    train_ds, val_ds = dataset_pipeline()
    
    EPOCHS = 1
    steps = 0
    for epoch in range(1, EPOCHS+1):
        # Train
        losses = []
        for batch in train_ds:
            feats, labels, feat_lens, label_lens = batch
            loss = train_step(feats, labels, feat_lens, label_lens)
            losses.append(float(loss.numpy()))
            steps += 1
            if steps % 50 == 0:
                print(f"step {steps}: loss {np.mean(losses[-50:]):.3f}")
        print(f"Epoch {epoch}: train loss {np.mean(losses):.3f}")
    
        # Quick sanity eval on a few val batches
        val_losses = []
        for i, batch in enumerate(val_ds.take(10)):
            feats, labels, feat_lens, label_lens = batch
            logits = model(feats, training=False)
            logit_lens = tf.cast(tf.shape(logits)[1], tf.int32) * tf.ones_like(feat_lens)
            val_losses.append(float(ctc_loss(logits, labels, logit_lens, label_lens).numpy()))
        print(f"Epoch {epoch}: val loss {np.mean(val_losses):.3f}")
    
    # Save
    os.makedirs("artifacts", exist_ok=True)
    model.save("artifacts/morse_ctc_model.keras")
    
    print("Training complete. Try inference by loading a wav and calling decode_wav().")

if __name__ == "__main__":
    main()
