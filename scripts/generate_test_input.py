from librosa import filters
import librosa
import numpy as np


y, sr = librosa.load(librosa.ex('trumpet'))
print(sr)

m = filters.mel(sr=sr, n_fft=513, n_mels=80)
np.save("mel_basis.npy", m)

m = m.astype(np.float64)

pinv = np.linalg.pinv(m)
np.save("pinv_output.npy", pinv)

mel_spec =  librosa.feature.melspectrogram(y=y, sr=sr, n_fft=513, n_mels=80)

# You need to apply a patch to librosa to remove chunk optimisation here
val = librosa.util.nnls(m, mel_spec)

np.save("example_spectrogram.npy", mel_spec)
np.save("nnls_output.npy", val)

