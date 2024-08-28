use crate::data::{TextGenerationBatch, TextGenerationTrainingBatch};
use burn::config::Config;
use burn::module::Module;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};

#[derive(Config)]
pub struct KoGPT2ModelConfig {
    transformer: TransformerEncoderConfig,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}


#[derive(Module, Debug)]
pub struct KoGPT2Model<B: Backend> {
    transformer: TransformerEncoder<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}

impl KoGPT2ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> KoGPT2Model<B> {
        let output = LinearConfig::new(self.transformer.d_model, self.vocab_size).init(device);
        let transformer = self.transformer.init(device);
        let embedding_token = EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);
        let embedding_pos = EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model).init(device);

        KoGPT2Model {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            vocab_size: self.vocab_size,
            pad_token: self.pad_token,
            max_seq_length: self.max_seq_length,
        }
    }
}

impl<B: Backend> KoGPT2Model<B> {
    pub fn forward_training(&self, item: TextGenerationTrainingBatch<B>) -> ClassificationOutput<B>{
        let [batch_size, seq_length] = item.tokens_inputs.dims();
        let device = &self.devices()[0];

        let inputs = item.tokens_inputs.to_device(device);
        let targets = item.targets.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(inputs);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let encoded = self.transformer.forward(TransformerEncoderInput::new(embedding)
            .mask_pad(mask_pad)
            .mask_attn(mask_attn));

        let output = self.output.forward(encoded);
        let output_flatten = output.reshape([batch_size * seq_length, self.vocab_size]);
        let targets_flatten = targets.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_token]))
            .init(&output_flatten.device());
        let loss = loss.forward(output_flatten.clone(), targets_flatten.clone());

        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: targets_flatten,
        }
    }

    pub fn forward(&self, item: TextGenerationBatch<B>) -> Tensor<B, 2> {
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.devices()[0];

        let inputs = item.tokens.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(inputs);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let encoded = self.transformer.forward(TransformerEncoderInput::new(embedding)
            .mask_pad(mask_pad)
            .mask_attn(mask_attn));

        let output = self.output.forward(encoded);
        let output_flatten = output.reshape([batch_size * seq_length, self.vocab_size]);

        output_flatten
    }
}

impl<B: AutodiffBackend> TrainStep<TextGenerationTrainingBatch<B>, ClassificationOutput<B>> for KoGPT2Model<B> {
    fn step(&self, item: TextGenerationTrainingBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_training(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<TextGenerationTrainingBatch<B>, ClassificationOutput<B>> for KoGPT2Model<B> {
    fn step(&self, item: TextGenerationTrainingBatch<B>) -> ClassificationOutput<B> {
        self.forward_training(item)
    }
}
