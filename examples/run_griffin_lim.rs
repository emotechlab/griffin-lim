use griffin_lim::GriffinLim;
use hound::{SampleFormat, WavSpec, WavWriter};
use ndarray::prelude::*;
use ndarray_npy::read_npy;
use std::error::Error;
use std::result::Result;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let spectrogram: Array2<f32> = read_npy("resources/example_spectrogram.npy")?;

    let mel_basis = griffin_lim::mel::create_mel_filter_bank(22050.0, 1024, 80, 0.0, Some(8000.0));

    for iter in [0, 1, 2, 5, 10] {
        let timer = Instant::now();
        let mut vocoder = GriffinLim::new(mel_basis.clone(), 1024 - 256, 1.5, 1, 0.99)?;
        vocoder.iter = iter;
        let audio = vocoder.infer(&spectrogram)?;
        let duration = Instant::now().duration_since(timer);
        let rtf = duration.as_secs_f32() / (audio.len() as f32 / 22050_f32);
        println!("Iterations: {}, rtf: {}", iter, rtf);
        let spec = WavSpec {
            channels: 1,
            sample_rate: 22050,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };

        let mut writer = WavWriter::create(format!("audio_output_griffinlim_{}.wav", iter), spec)?;

        for sample in audio {
            writer.write_sample(sample)?;
        }

        writer.finalize()?;
        println!("Saved audio_output_griffinlim_{}.wav", iter);
    }
    Ok(())
}
