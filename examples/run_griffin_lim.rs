use clap::Parser;
use griffin_lim::GriffinLim;
use hound::{SampleFormat, WavSpec, WavWriter};
use ndarray::prelude::*;
use ndarray_npy::read_npy;
use std::error::Error;
use std::path::PathBuf;
use std::result::Result;
use std::time::Instant;

#[derive(Parser, Debug)]
pub struct Args {
    #[clap(long, short, default_value = "example_spectrogram.npy")]
    input: String,
    #[clap(short, long, default_value = "output.wav")]
    output: PathBuf,
    #[clap(long, default_value = "22050")]
    sample_rate: u32,
    #[clap(long, default_value = "1024")]
    ffts: usize,
    #[clap(long, default_value = "80")]
    mels: usize,
    #[clap(long, default_value = "256")]
    hop_length: usize,
    #[clap(long, default_value = "8000.0")]
    max_frequency: f32,
    #[clap(long, default_value = "1.7")]
    power: f32,
    #[clap(long, default_value = "10")]
    iters: usize
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let spectrogram: Array2<f32> = read_npy(args.input)?;

    let mel_basis = griffin_lim::mel::create_mel_filter_bank(args.sample_rate as f32, args.ffts, args.mels, 0.0, Some(args.max_frequency));

    let timer = Instant::now();
    let vocoder = GriffinLim::new(mel_basis.clone(), args.ffts - args.hop_length, args.power, args.iters, 0.99)?;
    let audio = vocoder.infer(&spectrogram)?;
    let duration = Instant::now().duration_since(timer);
    let rtf = duration.as_secs_f32() / (audio.len() as f32 / args.sample_rate as f32);
    println!("Iterations: {}, rtf: {}", args.iters, rtf);
        
    let spec = WavSpec {
        channels: 1,
        sample_rate: args.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut wav_writer = WavWriter::create(&args.output, spec)?;

    let mut i16_writer = wav_writer.get_i16_writer(audio.len() as u32);
    for sample in &audio {
        i16_writer.write_sample((*sample * i16::MAX as f32) as i16);
    }
    i16_writer.flush()?;

    println!("Saved {}", args.output.display());
    Ok(())
}
