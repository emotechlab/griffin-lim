use ndarray::prelude::*;

pub fn mel(
    sample_rate: f32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: Option<f32>,
) -> Array2<f32> {
    let fmax = match fmax {
        Some(m) => m,
        None => sample_rate / 2.0,
    };

    let mut weights: Array2<f32> = Array2::zeros((n_mels, 1 + n_fft / 2));

    let fft_freqs = fft_frequencies(n_fft as isize, sample_rate);
    let mel_freqs = mel_frequencies(n_mels + 2, Some(fmin), Some(fmax));

    // librosa/filters.py
    let mut fdiff: Array1<f32> = Array1::zeros(mel_freqs.len() - 1);
    for i in 1..mel_freqs.len() {
        fdiff[i - 1] = mel_freqs[i] - mel_freqs[i - 1];
    }

    let mut ramps: Array2<f32> = Array2::zeros((mel_freqs.len(), fft_freqs.len()));

    for r in 0..mel_freqs.len() {
        for c in 0..fft_freqs.len() {
            ramps[[r, c]] = mel_freqs[r] - fft_freqs[c];
        }
    }

    for i in 0..n_mels {
        let lower = -&ramps.row(i) / fdiff[i];
        let upper = &ramps.row(i + 2) / fdiff[i + 1];

        let mut row = weights.row_mut(i);
        for (store, (lower, upper)) in row.iter_mut().zip(lower.iter().zip(upper.iter())) {
            *store = 0.0_f32.max(lower.min(*upper));
        }
    }

    let norm: Array1<f32> =
        2.0 / (&mel_freqs.slice(s![2..(n_mels + 2)]) - &mel_freqs.slice(s![..n_mels]));

    for mut col in weights.columns_mut() {
        col *= &norm;
    }
    weights
}

/// Ported from [numpy](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html)
/// Returned DFT sampling frequencies. This takes in sample_rate instead of the sample spacing for
/// ease of compatibility with the internals.
fn fft_frequencies(window_len: isize, sample_rate: f32) -> Array1<f32> {
    // d is 1.0 / sample_rate
    let sample_spacing = 1.0 / sample_rate;
    let len = window_len as f32;

    let val = 1.0 / (len * sample_spacing);

    let n = (window_len - 1) / 2 + 1;

    let it = (0..n).chain(-window_len / 2..0).map(|x| x as f32 * val);
    Array1::from_iter(it)
}

/// Ported from librosa
/// [mel_frequencies](https://librosa.org/doc/main/generated/librosa.mel_frequencies.html#librosa-mel-frequencies)
fn mel_frequencies(n_mels: usize, fmin: Option<f32>, fmax: Option<f32>) -> Array1<f32> {
    let fmin = fmin.unwrap_or(0.0);
    let fmax = fmax.unwrap_or(11025.0);

    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);

    Array::linspace(min_mel, max_mel, n_mels).mapv(mel_to_hz)
}

/// Ported from librosa
/// [hz_to_mel](https://librosa.org/doc/main/generated/librosa.hz_to_mel.html#librosa.hz_to_mel)
fn hz_to_mel(frequency: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;

    if frequency > MIN_LOG_HZ {
        const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;
        let logstep = 6.4_f32.ln() / 27.0;
        MIN_LOG_MEL + (frequency / MIN_LOG_HZ).ln() / logstep
    } else {
        (frequency - F_MIN) / F_SP
    }
}

/// Ported from librosa [mel_to_hz](https://librosa.org/doc/main/generated/librosa.mel_to_hz.html)
fn mel_to_hz(mels: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;

    if mels >= MIN_LOG_MEL {
        let logstep = 6.4_f32.ln() / 27.0;
        MIN_LOG_HZ * (logstep * (mels - MIN_LOG_MEL)).exp()
    } else {
        F_MIN + F_SP * mels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use ndarray_npy::read_npy;

    #[test]
    fn compare_basis() {
        let mel_basis: Array2<f32> = read_npy("resources/mel_basis.npy").unwrap();

        let actual = mel(22050.0, 513, 80, 0.0, None);

        assert_eq!(mel_basis.dim(), actual.dim());

        for r in 0..mel_basis.nrows() {
            println!("Row: {}", r);
            for c in 0..mel_basis.ncols() {
                println!("Col: {}", c);
                assert_approx_eq!(f32, mel_basis[[r, c]], actual[[r, c]], epsilon = 0.0001);
            }
        }
    }

    #[test]
    fn generate_mel_basis() {
        let mels = mel_frequencies(40, None, None);
        let expected = &[
            0., 85.317, 170.635, 255.952, 341.269, 426.586, 511.904, 597.221, 682.538, 767.855,
            853.173, 938.49, 1024.856, 1119.114, 1222.042, 1334.436, 1457.167, 1591.187, 1737.532,
            1897.337, 2071.84, 2262.393, 2470.47, 2697.686, 2945.799, 3216.731, 3512.582, 3835.643,
            4188.417, 4573.636, 4994.285, 5453.621, 5955.205, 6502.92, 7101.009, 7754.107,
            8467.272, 9246.028, 10096.408, 11025.,
        ];

        for (i, (actual, expected)) in mels.iter().zip(expected).enumerate() {
            println!("Iteration: {}", i);
            assert_approx_eq!(f32, *actual, *expected, epsilon = 0.01);
        }
    }

    #[test]
    fn hz_to_mel_simple() {
        assert_approx_eq!(f32, hz_to_mel(60.0), 0.9);
        assert_approx_eq!(f32, hz_to_mel(110.0), 1.65);
        assert_approx_eq!(f32, hz_to_mel(220.0), 3.3);
        assert_approx_eq!(f32, hz_to_mel(440.0), 6.6);
    }

    #[test]
    fn mel_to_hz_simple() {
        assert_approx_eq!(f32, mel_to_hz(1.0), 66.667, epsilon = 0.001);
        assert_approx_eq!(f32, mel_to_hz(2.0), 133.333, epsilon = 0.001);
        assert_approx_eq!(f32, mel_to_hz(3.0), 200.0, epsilon = 0.001);
        assert_approx_eq!(f32, mel_to_hz(4.0), 266.667, epsilon = 0.001);
        assert_approx_eq!(f32, mel_to_hz(5.0), 333.333, epsilon = 0.001);
    }

    #[test]
    fn fftfreqs() {
        let res = fft_frequencies(10, 10.0);
        let expected = &[0.0, 1.0, 2.0, 3.0, 4.0, -5.0, -4.0, -3.0, -2.0, -1.0];

        for (i, (actual, expected)) in res.iter().zip(expected).enumerate() {
            println!("Iteration: {}", i);
            assert_approx_eq!(f32, *actual, *expected);
        }

        let expected = &[
            0., 0.625, 1.25, 1.875, 2.5, 3.125, 3.75, 4.375, -5., -4.375, -3.75, -3.125, -2.5,
            -1.875, -1.25, -0.625,
        ];

        let res = fft_frequencies(16, 10.0);

        for (i, (actual, expected)) in res.iter().zip(expected).enumerate() {
            println!("Iteration: {}", i);
            assert_approx_eq!(f32, *actual, *expected);
        }

        let expected = &[
            0.,
            470.58823529,
            941.17647059,
            1411.76470588,
            1882.35294118,
            2352.94117647,
            2823.52941176,
            3294.11764706,
            3764.70588235,
            -3764.70588235,
            -3294.11764706,
            -2823.52941176,
            -2352.94117647,
            -1882.35294118,
            -1411.76470588,
            -941.17647059,
            -470.58823529,
        ];

        let res = fft_frequencies(17, 8000.0);
        for (i, (actual, expected)) in res.iter().zip(expected).enumerate() {
            println!("Iteration: {}", i);
            assert_approx_eq!(f32, *actual, *expected);
        }
    }
}
