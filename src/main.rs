use crate::data::TestCase;
use burn::backend;
use std::fs::File;
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;
use crate::model::KoGPT2ModelRecord;

pub mod model;
mod data;
mod tokenizer;
mod train;

type MyBackend = backend::Wgpu;

fn main() {
    let device = <MyBackend as Backend>::Device::default();
    let pt_recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
    let loaded: KoGPT2ModelRecord<MyBackend> = pt_recorder.load("/home/tmvkrpxl0/kogpt2-base-v2/pytorch_model.bin".into(), &device).unwrap();

    dbg!(loaded.pad_token);
}
