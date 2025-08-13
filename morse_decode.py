# save as infer_morse.py
import tensorflow as tf, numpy as np, argparse, os

# --- alphabet must match training ---
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
MEL_FB = tf.constant(tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=N_MELS, num_spectrogram_bins=N_FFT//2+1,
    sample_rate=SR, lower_edge_hertz=20.0, upper_edge_hertz=SR/2.0
))

def wav_to_logmels(wav):
    stft = tf.signal.stft(wav, frame_length=N_FFT, frame_step=HOP, fft_length=N_FFT)
    mag = tf.abs(stft)
    mel = tf.matmul(tf.square(mag), MEL_FB)
    return tf.math.log(mel + 1e-6)

def preprocess(wav_f32):
    feats = wav_to_logmels(tf.convert_to_tensor(wav_f32, tf.float32))
    m, s = tf.reduce_mean(feats), tf.math.reduce_std(feats) + 1e-6
    return ((feats - m) / s)[None, ...]  # [1,T,M]

def greedy_decode(logits):
    probs = tf.nn.log_softmax(logits, -1)
    ids = tf.argmax(probs, -1).numpy()[0]  # [T]
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(ID2TOK.get(i, ''))
            print(out[-1])
        prev = i
    return ''.join(out)

def load_wav(path):
    audio_bytes = tf.io.read_file(path)
    audio, sr = tf.audio.decode_wav(audio_bytes)
    audio = tf.squeeze(audio, -1)  # mono
    if sr != SR:
        raise SystemExit("Please resample to 16 kHz: ffmpeg -i in.wav -ac 1 -ar 16000 out.wav")
    return audio.numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/morse_ctc_model.keras")
    ap.add_argument("wav")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    print("model loaded")
    wav = load_wav(args.wav)
    print("wav loaded")
    feats = preprocess(wav)
    print("wave preprocessed")
    logits = model(feats, training=False)
    print("logits processed")
    print(greedy_decode(logits))
    print("decode complete")

if __name__ == "__main__":
    main()
