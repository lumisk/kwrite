use std::sync::Arc;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::SamplerDataset;
use burn::lr_scheduler::noam::NoamLrSchedulerConfig;
use burn::module::Module;
use burn::nn::transformer::TransformerEncoderConfig;
use burn::optim::AdamConfig;
use burn::record::{CompactRecorder, DefaultRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::train::LearnerBuilder;
use burn::train::metric::{AccuracyMetric, CudaMetric, LearningRateMetric, LossMetric};
use crate::data::{TextGenerationBatcher, TextGenerationItem};
use crate::model::{KoGPT2Model, KoGPT2ModelConfig};
use crate::tokenizer::GPT2Tokenizer;

#[derive(Config)]
pub struct FineTuningConfig {
    transformer: TransformerEncoderConfig,
    optimizer: AdamConfig,
    #[config(default = 512)]
    max_seq_length: usize,
    #[config(default = 6)]
    batch_size: usize,
    #[config(default = 50)]
    num_epochs: usize,
}

pub fn train<B: AutodiffBackend, D: Dataset<TextGenerationItem> + 'static>(
    device: B::Device,
    dataset_train: D,
    dataset_test: D,
    config: FineTuningConfig,
    artifact_dir: &str
) {
    let tokenizer: Arc<_> = GPT2Tokenizer::default().into();
    let batcher_train = TextGenerationBatcher {
        tokenizer: tokenizer.clone(),
        max_seq_length: config.max_seq_length,
    };
    let batcher_test = TextGenerationBatcher {
        tokenizer: tokenizer.clone(),
        max_seq_length: config.max_seq_length,
    };

    let model: KoGPT2Model<B> = KoGPT2ModelConfig::new(
        config.transformer.clone(),
        tokenizer.vocab_size(),
        tokenizer.pad_token() as usize,
        config.max_seq_length
    ).init(&device);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_train, 10_000));

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_test, 1000));

    let accum = 6;
    let optimizer = config.optimizer.init();
    let lr_scheduler = NoamLrSchedulerConfig::new(0.01 / accum as f64)
        .with_warmup_steps(6000)
        .with_model_size(config.transformer.d_model)
        .init();

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token() as usize))
        .metric_valid_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token() as usize))
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .grads_accumulation(accum)
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, optimizer, lr_scheduler);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config.save(format!("{}/config.json", artifact_dir)).unwrap();

    DefaultRecorder::new()
        .record(model_trained.into_record(), format!("{}/model", artifact_dir).into())
        .unwrap();
}
