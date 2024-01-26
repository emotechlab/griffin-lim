# Stepping Through Griffin-Lim

This is an accompaniment to the Rustnation 2024 talk "Creating a Text-To-Speech
System in Rust" and is using a spectrogram of "Hello world" generated using a
version of that TTS system.

It also uses `examples/run_griffin_lim.rs` to generate the output with the `debug_dump`
feature enabled to grab npy files of intermediate phases.

So to start lets look at our spectrogram we're going to be vocoding.

![image](./resources/mel_spec.png)

To vocode this we need to reconstruct the phase spectrum so we can do an inverse
Short-Time Fourier Transform to reconstruct the audio.

The mel spectrogram uses mel frequency bands to compress a linear spectrogram to
a more compressed representation. It does that by multiplying the linear spectrogram
by a filter bank. So we need to invert that multiplication to go back to a linear
spectrogram. This is an optimisation problem as with matrix multiplication AB != BA.

So we use a limited-memory BGFS solver to solve for the inverse of the matrix 
multiplication. This is mainly to keep in line with the reference implementation we used.

So if we construct the mel filter bank we get:

![image](./resources/mel_basis.png)

Using these two matrices we invert and get the following linear spectrogram:

![image](./resources/linear_spec.png)

It's expected that this will look a lot emptier higher up, the human ear is more
sensitive to lower frequencies so the mel spectrogram conversion throws away a lot
of high frequency data and accentuates the low frequency data. Near the bottom of
the spectrogram we can see a hint of the mel spectrograms form.

## The Audio

And listen to the output:

![audio](https://github.com/emotechlab/griffin-lim/raw/docs/super-guide-time/doc/output.wav)
