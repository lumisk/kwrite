use crate::data::TestCase;
use burn::backend;
use std::fs::File;

pub mod model;
mod data;
mod tokenizer;
mod train;

type MyBackend = backend::Wgpu;

fn main() {
    /*let device = <MyBackend as Backend>::Device::default();
    let pt_recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
    let loaded: KoGPT2ModelRecord<MyBackend> = pt_recorder.load("/home/tmvkrpxl0/kogpt2-base-v2".into(), &device).unwrap();

    dbg!(loaded.pad_token);*/

    let test_values = File::open("./data/template.json").unwrap();
    let test_values: Vec<TestCase> = serde_json::from_reader(test_values).unwrap();

    test_values.into_iter().for_each(|value| {
        println!("{:?}", value);
    });
}
