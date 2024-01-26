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

    let iter = 30;

    let timer = Instant::now();
    let mut vocoder = GriffinLim::new(mel_basis.clone(), 1024 - 256, 1.7, iter, 0.99)?;
    let audio = vocoder.infer(&spectrogram)?;
    let duration = Instant::now().duration_since(timer);
    let rtf = duration.as_secs_f32() / (audio.len() as f32 / 22050_f32);
    println!("Iterations: {}, rtf: {}", iter, rtf);
        
    let spec = WavSpec {
        channels: 1,
        sample_rate: 22050,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut wav_writer = WavWriter::create(format!("audio_output_griffinlim_{}.wav", iter), spec)?;

    let mut i16_writer = wav_writer.get_i16_writer(audio.len() as u32);
    for sample in &audio {
        i16_writer.write_sample((*sample * i16::MAX as f32) as i16);
    }
    i16_writer.flush()?;

    println!("Saved audio_output_griffinlim_{}.wav", iter);
    Ok(())
}
