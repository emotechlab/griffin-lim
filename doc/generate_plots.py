from librosa import display, feature
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

# Rerun the rust code
re_run = False

def plot_complex_spec(spec, name):
    magnitude, phase = librosa.magphase(spec)
    fig, ax = plt.subplots(nrows=2)


    img = librosa.display.specshow(magnitude, x_axis='time',
                             y_axis='linear', sr=22050,
                             ax=ax[0])
    img = librosa.display.specshow(np.angle(phase), x_axis='time',
                             y_axis='linear', sr=22050,
                             ax=ax[1])

    plt.savefig(name)
    plt.close()

y, sr = librosa.load(librosa.ex('trumpet'))
S = librosa.stft(y)
plot_complex_spec(S, "resources/trumpet.png")

S = np.load("hello_world.npy")
if re_run:
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

for i in range(0, 11):
    plot_complex_spec(np.load(f"estimate_spec_{i}.npy"), f"resources/estimate_{i}.png")
