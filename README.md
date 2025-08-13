# TouCans AI CW Decoder 

TouCans AI CW Decoder [TensorFlow]([url](https://www.tensorflow.org/)) model with initial scaffolding courtesy of ChatGPT 5. 

## morse_ctc_tpu.py 
Trains a model to decode Morse code. The script creates synthetic training data, sends it to the model, and monitors loss. At present, it only runs one epoch at a time and is setup to build an additional epoch's worth of training starting with the default .keras file output by a prior run of the script. 

## dump_training.py
Dumps out both .wav and .png files created by the data synthesis portion of the script. This was originally set up as a sanity check for training. (i.e. was the model actually getting any Morse code at all?)

## morse_decode.py
Decodes 16kHz sampled mono .wav files containing Morse code. In the initial revision, the model is only successful at decoding synthetic data so far.
