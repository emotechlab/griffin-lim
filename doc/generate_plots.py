from librosa import display, feature
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt


S = np.load("hello_world.npy")
os.system(f"cargo run --features debug_dump --example run_griffin_lim -- --input hello_world.npy --power 1.7 --iters 10" )

mel_basis = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80)
print(mel_basis.shape)

linspec =np.load("linear_spectrogram.npy")

fig, ax = plt.subplots()
img = librosa.display.specshow(mel_basis, x_axis="linear")
ax.set(ylabel='Mel filter', title='Mel filter bank')
fig.colorbar(img, ax=ax)

plt.savefig('resources/mel_basis.png', bbox_inches='tight')


fig, ax = plt.subplots()
img = librosa.display.specshow(S, x_axis='time',
                         y_axis='mel', sr=22050,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')

plt.savefig('resources/mel_spec.png', bbox_inches='tight')

S = np.load("linear_spectrogram.npy")
fig, ax = plt.subplots()
img = librosa.display.specshow(S, x_axis='time',
                         y_axis='linear', sr=22050,
                         ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')

plt.savefig('resources/linear_spec.png', bbox_inches='tight')
