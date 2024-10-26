use burn::backend;
use burn::config::Config;
use burn::nn::transformer::TransformerEncoderRecord;
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;

pub mod model;
mod data;
mod tokenizer;
mod train;
mod gpt_config;

type MyBackend = backend::Wgpu;

fn main() {
    let device = <MyBackend as Backend>::Device::default();
    let pt_recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
    let loaded: TransformerEncoderRecord<MyBackend> = pt_recorder.load("./kogpt2-base-v2/pytorch_model.bin".into(), &device).unwrap();

    dbg!(loaded.layers.len());
}
