use lbfgsb::lbfgsb;
use ndarray::{par_azip, prelude::*, ScalarOperand};
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::svd::SVD;
use ndarray_linalg::{Lapack, Scalar};
use ndarray_npy::read_npy;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ndarray_stats::errors::MinMaxError;
use ndarray_stats::QuantileExt;
use num_traits::{Float, FloatConst, FromPrimitive};
use rand::SeedableRng;
use rand_isaac::isaac64::Isaac64Rng;
use realfft::num_complex::Complex;
use realfft::num_traits;
use realfft::num_traits::AsPrimitive;
use realfft::RealFftPlanner;
use std::env::{self, VarError};
use std::error::Error;
use std::fmt::Display;
use std::str::FromStr;
use tracing::warn;

mod mel;

pub struct GriffinLim {
    mel_basis: Array2<f32>,
    noverlap: usize,
    power: f32,
    pub iter: usize,
    pub momentum: f32,
}

/// Utility to get environment variable with name `name` and parses it to type `T`
/// returning `default` if the variable is not set and an error if parsing fails
fn env_default<T: FromStr>(name: &'static str, default: T) -> Result<T, String>
where
    <T as FromStr>::Err: std::error::Error,
{
    match env::var(name) {
        Ok(val) => match val.parse() {
            Ok(result) => Ok(result),
            Err(e) => Err(format!("Failed to parse env var {}: {}", name, e)),
        },
        Err(e) => match e {
            VarError::NotPresent => Ok(default),
            VarError::NotUnicode(_) => {
                Err(format!("Failed to parse env var {}: invalid unicode", name))
            }
        },
    }
}

impl GriffinLim {
    pub fn new_from_env() -> Result<Self, Box<dyn Error>> {
        let mel_basis_npy = env_default(
            "GRIFFINLIM_MEL_BASIS",
            "resources/mel_basis.npy".to_string(),
        )?;
        let mel_basis = read_npy::<_, Array2<f32>>(mel_basis_npy)?;
        let noverlap = env_default("GRIFFINLIM_NOVERLAP", 768)?;
        let power = env_default("GRIFFINLIM_POWER", 2.0 / 3.0)?;
        let iter = env_default("GRIFFINLIM_ITER", 10)?;
        let momentum = env_default("GRIFFINLIM_MOMENTUM", 0.99)?;
        Ok(Self::new(mel_basis, noverlap, power, iter, momentum)?)
    }

    pub fn new_from_env_static(mel_basis: Array2<f32>) -> Result<Self, Box<dyn Error>> {
        let noverlap = env_default("GRIFFINLIM_NOVERLAP", 768)?;
        let power = env_default("GRIFFINLIM_POWER", 2.0 / 3.0)?;
        let iter = env_default("GRIFFINLIM_ITER", 10)?;
        let momentum = env_default("GRIFFINLIM_MOMENTUM", 0.99)?;
        Ok(Self::new(mel_basis, noverlap, power, iter, momentum)?)
    }

    pub fn new(
        mel_basis: Array2<f32>,
        noverlap: usize,
        power: f32,
        iter: usize,
        momentum: f32,
    ) -> Result<Self, Box<dyn Error>> {
        let nfft = 2 * (mel_basis.dim().1 - 1);
        if noverlap >= nfft {
            return Err(format!(
                "nnft must be < noverlap - nfft: {}, noverlap: {}",
                nfft, noverlap
            )
            .into());
        }

        if momentum > 1.0 || momentum < 0.0 {
            return Err(format!("Momentum is {}, should be in range [0,1]", momentum).into());
        }

        if power <= 0.0 {
            return Err(format!("Power is {}, should be > 0", power).into());
        }

        Ok(Self {
            mel_basis,
            noverlap,
            power,
            iter,
            momentum,
        })
    }

    pub fn infer(&self, mel_spec: &Array2<f32>) -> Result<Array1<f32>, Box<dyn Error>> {
        // mel_basis has dims (nmel, nfft)
        // lin_spec has dims (nfft, time)
        // mel_spec has dims (nmel, time)
        // need to transpose for griffin lim - maybe do this internally?
        let mut lin_spec = nnls(&self.mel_basis, &mel_spec.mapv(|x| 10.0_f32.powf(x)), 1e-15)?;

        // correct for "power" parameter of mel-spectrogram
        lin_spec.mapv_inplace(|x| x.powf(1.0 / self.power));

        let params = Parameters {
            momentum: self.momentum,
            iter: self.iter,
            ..Default::default()
        };

        // reconstruct signal
        let mut reconstructed = griffin_lim_with_params(
            &lin_spec.t().as_standard_layout().to_owned(),
            2 * (self.mel_basis.dim().1 - 1),
            self.noverlap,
            params,
        )?;

        // Normalise audio by RMS
        let signal_rms = f32::sqrt(
            reconstructed.iter().fold(0.0, |_s, s| _s + s.powf(2.0)) / reconstructed.len() as f32,
        );

        let target_gain: f32 = 0.1;

        // Gain factor to reach the target RMS based on the
        // RMS of the signal, more samples = more flat amplitude.
        let scale = target_gain / signal_rms;

        reconstructed.mapv_inplace(|x| x * scale);

        Ok(reconstructed)
    }
}

/// Parameters to provide to the griffin-lim vocoder
pub struct Parameters<T> {
    momentum: T,
    seed: u64,
    iter: usize,
    init_random: bool,
}

impl<T> Default for Parameters<T>
where
    T: FromPrimitive + Float,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Parameters<T>
where
    T: FromPrimitive + Float,
{
    /// Creates default parameters for the algorithm.
    pub fn new() -> Self {
        Self {
            momentum: T::from_f32(0.99).unwrap(),
            seed: 42,
            iter: 32,
            init_random: true,
        }
    }

    /// Sets the momentum for fast Griffin-Lim. If set to zero this is the original implementation
    /// as defined by _D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time
    /// Fourier transform,"_. The fast algorithm is defined in _Perraudin, N., Balazs, P., &
    /// Søndergaard, P. L. "A fast Griffin-Lim algorithm,"_.
    ///
    /// Default value is 0.99
    pub fn momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }

    /// A random seed to use for initializing the phase.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Number of iterations to run - default value is 32.
    pub fn iter(mut self, iter: usize) -> Self {
        self.iter = iter;
        self
    }

    /// Specifies whether the phase values are initialised randomly, is true by default.
    pub fn init_random(mut self, init_random: bool) -> Self {
        self.init_random = init_random;
        self
    }
}

/// Get 'hann' window of length `n`
/// Implements the path of functions:
/// get_window() -> general_hamming() -> general_cosine()
/// with sym=False
/// https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/signal/windows/_windows.py
fn get_hann_window<T: Float + FloatConst>(n: usize) -> Array1<T> {
    let alpha = 0.5;
    let a = vec![T::from(alpha).unwrap(), T::from(1. - alpha).unwrap()];

    if n <= 1 {
        return Array1::zeros(n);
    }; // _len_guards(M)

    let m = n + 1; // _extend(M, False)
    let fac = Array1::linspace(-T::PI(), T::PI(), m);
    let mut w = Array1::zeros(m);
    for k in 0..a.len() {
        let fac_cos = fac.mapv(|x| (x * T::from(k).unwrap()).cos() * a[k]);
        w.assign(&(&w + fac_cos));
    }
    w.slice(s![..w.shape()[0] - 1]).to_owned() // _truncate(w, True)
}

/// Applies a Fourier Transform to `x`
fn fft_helper<T: realfft::FftNum>(
    x: &Array1<T>,
    window: &Array1<T>,
    planner: &mut RealFftPlanner<T>,
    nfft: usize,
    noverlap: usize,
) -> Array2<Complex<T>> {
    let step = nfft - noverlap;
    let shape = ((x.shape()[0] - noverlap) / step, nfft);
    let mut strided = Array2::<T>::zeros(shape);
    for (ii, mut windowed_segment) in strided.rows_mut().into_iter().enumerate() {
        let segment = x.slice(s![ii * step..ii * step + nfft]);
        windowed_segment.assign(&(&segment * window));
    }

    // 2D rfft
    let r2c = planner.plan_fft_forward(nfft);
    let mut result = Array2::<Complex<T>>::zeros((shape.0, nfft / 2 + 1));
    par_azip!((mut input in strided.rows_mut(), mut output in result.rows_mut())
        {
            // slices are safe as rows are the last dimension
            let input_slice = input.as_slice_mut().unwrap();
            let output_slice = output.as_slice_mut().unwrap();
            r2c.process(input_slice, output_slice).unwrap();
        }
    );
    result
}

/// Compute the Short Time Fourier Transform (STFT)
/// <https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/signal/_spectral_py.py#L1025>
fn stft<T: realfft::FftNum + Float>(
    signal: &Array1<T>,
    window: &Array1<T>,
    planner: &mut RealFftPlanner<T>,
    nfft: usize,
    noverlap: usize,
    scale_result: bool,
    padded: bool,
) -> Array2<Complex<T>> {
    let x = if padded {
        // Pad to integer number of windowed segments
        let nstep = nfft - noverlap;
        let y_len = (signal.shape()[0] + nfft) as i32;
        let nadd = python_mod(
            python_mod(-(y_len - nfft as i32), nstep as i32),
            nfft as i32,
        ) as usize; // (-(x.shape[-1]-nperseg) % nstep) % nperseg
        let mut y = Array1::<T>::zeros(y_len as usize + nadd);
        let pad_len = nfft / 2;
        y.slice_mut(s![pad_len..y.shape()[0] - pad_len - nadd])
            .assign(&signal);
        y
    } else {
        signal.clone()
    };

    let mut result = fft_helper(&x, window, planner, nfft, noverlap);

    if scale_result {
        let mut scale = T::one() / window.sum().powf(T::from_f32(2.0).unwrap()); // scaling='spectrum'
        scale = scale.sqrt(); // mode == 'stft'
        result.mapv_inplace(|x| {
            x * Complex {
                re: scale,
                im: T::zero(),
            }
        });
    }

    result
}

/// Python's modulo operator
/// <https://stackoverflow.com/questions/3883004/the-modulo-operation-on-negative-numbers-in-python>
fn python_mod(n: i32, base: i32) -> i32 {
    n - base * (n as f32 / base as f32).floor() as i32
}

pub fn griffin_lim<T>(
    spectrogram: &Array2<T>,
    nfft: usize,
    noverlap: usize,
) -> Result<Array1<T>, Box<dyn std::error::Error>>
where
    T: realfft::FftNum + Float + FloatConst + Display + SampleUniform,
    Complex<T>: ScalarOperand,
{
    griffin_lim_with_params(spectrogram, nfft, noverlap, Parameters::new())
}

/// compute an estimate of real-valued time-domain signal
/// from a magnitude spectrogram
pub fn griffin_lim_with_params<T>(
    spectrogram: &Array2<T>,
    nfft: usize,
    noverlap: usize,
    params: Parameters<T>,
) -> Result<Array1<T>, Box<dyn std::error::Error>>
where
    T: realfft::FftNum + Float + FloatConst + Display + SampleUniform,
    Complex<T>: ScalarOperand,
{
    // set up griffin lim parameters
    let mut rng = Isaac64Rng::seed_from_u64(params.seed);
    if params.momentum > T::one() || params.momentum < T::zero() {
        return Err(format!("Momentum is {}, should be in range [0,1]", params.momentum).into());
    }

    let window = get_hann_window(nfft);
    let spectrogram = spectrogram.mapv(|r| Complex::from(r));

    // Initialise estimate
    let mut estimate = if params.init_random {
        let mut angles = Array2::<T>::random_using(
            spectrogram.raw_dim(),
            Uniform::from(-T::PI()..T::PI()),
            &mut rng,
        );
        // realfft doesn't handle invalid input
        angles.slice_mut(s![.., 0]).fill(T::zero());
        angles.slice_mut(s![.., -1]).fill(T::zero());
        &spectrogram * &angles.mapv(|t| Complex::from_polar(T::one(), t))
    } else {
        spectrogram.clone()
    };
    // TODO: Pre-allocate inverse and rebuilt and use `.assign` instead of `=`
    // this requires some fighting with the borow checker
    let mut inverse: Array1<T>;
    let mut rebuilt: Array2<Complex<T>>;
    let mut tprev: Option<Array2<Complex<T>>> = None;
    let planner = &mut RealFftPlanner::<T>::new();

    for _ in 0..params.iter {
        // Reconstruct signal approximately from the estimate
        inverse = istft(&estimate, &window, planner, nfft, noverlap);
        // Rebuild the spectrogram
        rebuilt = stft(&inverse, &window, planner, nfft, noverlap, true, true);
        estimate.assign(&rebuilt);
        // Momentum parameter ensures faster convergence
        if let Some(tprev) = &mut tprev {
            let delta = &*tprev * Complex::from(params.momentum / (T::one() + params.momentum));
            estimate = estimate - delta;
            std::mem::swap(tprev, &mut rebuilt);
        } else {
            tprev = Some(rebuilt);
        }
        // Get angles from estimate and apply to magnitueds
        let eps = T::min_positive_value();
        // get angles from new estimate
        estimate.mapv_inplace(|x| x / (x.norm() + eps));
        // enforce magnitudes
        estimate.assign(&(&estimate * &spectrogram));
    }
    let mut signal = istft(&estimate, &window, planner, nfft, noverlap);
    let norm = T::from(nfft).unwrap();
    signal.mapv_inplace(|x| x / norm);
    Ok(signal)
}

/// https://github.com/scipy/scipy/blob/v1.10.1/scipy/signal/_spectral_py.py#L1220-L1506
pub fn istft<T: realfft::FftNum + num_traits::Float>(
    spectrogram: &Array2<Complex<T>>,
    window: &Array1<T>,
    planner: &mut RealFftPlanner<T>,
    nfft: usize,
    noverlap: usize,
) -> Array1<T> {
    let dims = spectrogram.raw_dim();
    // spectrogram is the STFT, which is FTs of small chunks of the signal
    // it has dims (chunk index/time (nseg), frequency bin (steps))
    let nseg = dims[0];
    let nstep = nfft - noverlap;

    // Set up window
    let win = window.view();
    let winsum = win.sum();
    let win2 = win.mapv(|x| x * x);
    let win2 = win2.view();

    // Do ifft
    let inverse_plan = planner.plan_fft_inverse(nfft);
    let mut spectrogram = spectrogram.as_standard_layout();
    let mut ifft_subs = Array2::zeros([nseg, nfft]);
    let nfft_float = T::from_usize(nfft).unwrap();
    par_azip!((mut fft in spectrogram.rows_mut(), mut ifft in ifft_subs.rows_mut())
        {
            inverse_plan
                .process(fft.as_slice_mut().unwrap(), ifft.as_slice_mut().unwrap())
                .unwrap();
            ifft.mapv_inplace(|x| x * winsum / nfft_float);
            ifft.assign(&(&ifft * &win));
        }
    );

    // initialize output
    let output_len = nfft + (nseg - 1) * nstep;
    let mut output = Array1::zeros(output_len);
    let mut norm = Array1::<T>::zeros(output_len);

    // fill in output with overlap-add
    for ii in 0..nseg {
        let idx = ii * nstep;
        // Compound assignment for arrays requires nightly, so have to assign
        let mut output_slice = output.slice_mut(s![idx..idx + nfft]);
        output_slice.assign(&(&output_slice + &ifft_subs.row(ii)));
        let mut norm_slice = norm.slice_mut(s![idx..idx + nfft]);
        norm_slice.assign(&(&norm_slice + &win2));
    }
    let tr = nfft as i32 / 2;
    let norm = norm.slice(s![tr..-tr]).to_owned();
    let min_norm = T::from_f32(1e-10).unwrap();
    if norm.iter().any(|x| *x < min_norm) {
        warn!("NOLA condition failed, STFT may not be invertible!");
    }
    let output = output.slice(s![tr..-tr]).to_owned();
    output / norm
}

#[derive(Debug, thiserror::Error)]
pub enum PinvError {
    #[error("SVD calculation failed in pinv: {0}")]
    SvdError(LinalgError),
    #[error("Nans in SVD: {0}")]
    MinMaxError(MinMaxError),
}

/// Compute Moore-Penrose psuedo-inverse matrix using SVD
/// See https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1463-L1656
fn pinv<T: Scalar<Real = T> + Lapack + num_traits::Float>(
    x: &Array2<T>,
    rcond: T,
) -> Result<Array2<T>, PinvError> {
    let dim = x.dim();
    let (u, mut s, vt) = x.svd(true, true).map_err(PinvError::SvdError)?;
    let ut = u.unwrap().reversed_axes();
    let v = vt.unwrap().reversed_axes();
    // Discard extra rows of SVD
    let ut = ut.slice(s![..dim.1.min(ut.dim().0), ..]);
    let v = v.slice(s![.., ..dim.0.min(v.dim().1)]);
    // Ignore small singular values
    // NOTE: Maybe don't need this error if NaNs in results causes .svd() to return an error
    let cutoff = *Array1::max(&s).map_err(PinvError::MinMaxError)? * rcond;
    s.mapv_inplace(|x| if x > cutoff { T::one() / x } else { T::zero() });
    // Could use broadcasting instead of from_diag?
    Ok(v.dot(&Array2::from_diag(&s).dot(&ut)))
}

#[derive(Debug, thiserror::Error)]
pub enum NnlsError {
    #[error("Failure to find pseudo-inverse matrix: {0}")]
    PinvError(PinvError),
    #[error("Optimisation failed: {0}")]
    OptimiserError(anyhow::Error),
}

/// Compute x to minimise ||a.dot(x) - b|| where a, x, b are matrices using l-bfgs-b optimizer
/// See `_nnls_obj` here: https://librosa.org/doc/main/_modules/librosa/util/_nnls.html
/// NOTE: The librosa top-level nnls function https://librosa.org/doc/main/_modules/librosa/util/_nnls.html#nnls
/// splits the optimisation into chunks to avoid large memory allocations for single calls
/// For simplicity this is not replicated, but it may be worth investigating if this
/// function becomes non-performant.
pub fn nnls<T: Scalar<Real = T> + Lapack + num_traits::Float + num_traits::AsPrimitive<f64>>(
    a: &Array2<T>,
    b: &Array2<T>,
    rcond: T,
) -> Result<Array2<T>, NnlsError>
where
    f64: AsPrimitive<T>,
{
    let x_init = pinv(a, rcond)
        .map_err(NnlsError::PinvError)?
        .dot(b)
        .mapv(|x| x.as_().clamp(0.0, f64::INFINITY));
    // T::as_() defined in trait AsPrimitive<f64>
    let a = &a.mapv(|x| x.as_());
    let b = &b.mapv(|x| x.as_());
    let x_shape = x_init.raw_dim();
    let x_init = x_init.into_raw_vec();
    let bounds = vec![(0.0, f64::INFINITY); x_init.len()];
    // Evaluates cost (return value) and gradient (by updating g)
    // with current estimate x
    let eval_fn = |x: &[f64], g: &mut [f64]| -> anyhow::Result<f64> {
        // copy x out of slice
        let x = ArrayView::from_shape(x_shape, x).unwrap();
        let diff = a.dot(&x) - b;
        let norm = b.len() as f64;
        let mut grad = ArrayViewMut::from_shape(x_shape, g).unwrap();
        grad.assign(&(a.t().dot(&diff) / norm));
        let val = 0.5 * (&diff * &diff).sum() / norm;
        Ok(val)
    };
    let result = lbfgsb(x_init, &bounds, eval_fn).map_err(NnlsError::OptimiserError)?;
    Ok(Array2::from_shape_vec(x_shape, result.x().to_vec())
        .unwrap()
        .mapv(|x| <f64 as AsPrimitive<T>>::as_(x)))
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use ndarray_npy::read_npy;

    use super::*;

    #[test]
    fn test_irfft() {
        // Set up
        // input length 500
        let mut input: Array1<Complex<f64>> = read_npy("resources/irfft_input.npy").unwrap();
        // irfft doesn't handle invalid input
        assert_eq!(0.0, input[0].im());
        assert_eq!(0.0, input[input.len() - 1].im());
        // python: numpy.fft.irrft(input, 998);
        // where input.dtype = numpy.complex64
        let expected: Array1<f64> = read_npy("resources/irfft_output.npy").unwrap();
        let nfft = expected.len();

        // Test
        let mut actual = Array1::zeros(nfft);
        let mut planner = RealFftPlanner::new();
        let plan = planner.plan_fft_inverse(nfft);
        plan.process(
            input.as_slice_mut().unwrap(),
            actual.as_slice_mut().unwrap(),
        )
        .unwrap();

        // Assert
        for (e, a) in expected.iter().zip(actual.iter()) {
            // np.fft.irfft does normalisation internally, c2r.process does not
            let e = *e as f32 * nfft as f32;
            assert_approx_eq!(f32, e, *a as f32);
        }
    }

    #[test]
    fn test_istft() {
        // Set up
        // to generate input:
        // ```
        // _f, _t, input = scipy.signal.stft(np.random.random(500), nfft=128, nperseg=128)
        // ```
        let input: Array2<Complex<f64>> = read_npy("resources/istft_input.npy").unwrap();
        let input = input.t().into_owned();
        let expected: Array1<f64> = read_npy("resources/istft_output.npy").unwrap();

        // Test
        let planner = &mut RealFftPlanner::new();
        let window = get_hann_window(128);
        let actual = istft::<f64>(&input, &window, planner, 128, 64);

        // Assert
        assert_eq!(expected.len(), actual.len());
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert_approx_eq!(f32, *e as f32, *a as f32);
        }
    }

    #[test]
    fn test_invertible() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let signal = Array1::<f64>::random_using(500, Uniform::from(0.0..1.0), &mut rng);
        let window = &get_hann_window(128);
        let planner = &mut RealFftPlanner::new();
        let transformed = stft(&signal, window, planner, 128, 64, true, true);
        let inverted = istft(&transformed, window, planner, 128, 64);
        assert!(inverted.len() >= signal.len());
        for (s, i) in signal.iter().zip(inverted.iter()) {
            assert_approx_eq!(f32, *s as f32, *i as f32);
        }
        let re_transformed = stft(&inverted, window, planner, 128, 64, true, true);
        for (t, r) in transformed.iter().zip(re_transformed.iter()) {
            assert_approx_eq!(f32, t.re as f32, r.re as f32);
            assert_approx_eq!(f32, t.im as f32, r.im as f32);
        }
    }

    #[test]
    fn test_pinv() {
        // Set up
        let input: Array2<f32> = read_npy("resources/mel_basis.npy").unwrap();
        let input = input.mapv(|x| x as f64);
        // python: numpy.linalg.pinv(mel_basis)
        let expected: Array2<f64> = read_npy("resources/pinv_output.npy").unwrap();

        // Test
        let actual = pinv::<f64>(&input, 1e-15).unwrap();

        // Assert
        assert_eq!(expected.dim().0, actual.dim().0);
        assert_eq!(expected.dim().1, actual.dim().1);
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert_approx_eq!(f32, *e as f32, *a as f32);
        }
    }

    #[test]
    #[ignore]
    fn test_nnls() {
        // Set up
        let mel_basis: Array2<f32> = read_npy("resources/mel_basis.npy").unwrap();
        let mel_basis = mel_basis.mapv(|x| x as f64);
        let mel_spec: Array2<f32> = read_npy("resources/example_spectrogram.npy").unwrap();
        let mel_spec = mel_spec.mapv(|x| x as f64);
        // python: librosa.util.nnls(mel_basis, mel_spec)
        // NOTE: To generate this test case, the nnls code
        // https://librosa.org/doc/main/_modules/librosa/util/_nnls.html#nnls
        // was edited to disable chunking of the optimisation by removing the line
        // `if B.shape[-1] <= n_columns`
        let expected: Array2<f64> = read_npy("resources/nnls_output.npy").unwrap();

        // Test
        let actual = nnls::<f64>(&mel_basis, &mel_spec, 1e-15).unwrap();

        // Assert
        assert_eq!(expected.dim().0, actual.dim().0);
        assert_eq!(expected.dim().1, actual.dim().1);
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert_approx_eq!(f32, *e as f32, *a as f32);
        }
    }

    #[test]
    fn test_griffinlim() {
        // Set up
        let spectrogram = read_npy::<_, Array2<f64>>("resources/griffinlim_input.npy")
            .unwrap()
            .reversed_axes();
        // python:
        // nfft = 2 * (spectrogram.shape[0] - 1)
        // librosa.griffinlim(spectrogram, init=None, hop_length=nfft//2, win_length=nfft, n_fft=nfft, momentum=0.4, n_iter=3)
        let expected: Array1<f64> = read_npy("resources/griffinlim_output.npy").unwrap();

        // Test
        let nfft = 2 * (spectrogram.dim().1 - 1);
        let noverlap = nfft / 2;
        let params = Parameters::new().momentum(0.4).iter(3).init_random(false);
        let actual = griffin_lim_with_params::<f64>(&spectrogram, nfft, noverlap, params).unwrap();

        // Assert
        assert_eq!(expected.len(), actual.len());
        for (e, a) in expected.iter().zip(actual.iter()) {
            // ISTFT / STFT normalisation is different
            // Factor of 2.0 here as librosa normalisation is different to scipy
            assert_approx_eq!(f32, *e as f32, *a as f32 * 2.0);
        }
    }
}
