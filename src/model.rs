use crate::data::{TextGenerationBatch, TextGenerationTrainingBatch};
use crate::gpt_config::GptConfig;
use burn::module::Module;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};

#[derive(Module, Debug)]
pub struct KoGPT2Model<B: Backend> {
    embed_dim: usize,
    wte: Embedding<B>,
    wpe: Embedding<B>,
    drop: Dropout,
    h: TransformerEncoder<B>,
    ln_f: LayerNorm<B>,
    vocab_size: usize,
    pad_token_id: usize
}

#[derive(Module, Debug)]
pub struct KoGPT2LMHeadModel<B: Backend> {
    //transformer: KoGPT2Model<B>,
    transformer: TransformerEncoder<B>,
    lm_head: Linear<B>,f
}

impl GptConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> KoGPT2LMHeadModel<B> {
        let embed_dim = self.n_embd;
        let wte = EmbeddingConfig::new(self.vocab_size, embed_dim).init(device);
        let wpe = EmbeddingConfig::new(self.task_specific_params.text_generation.max_length, embed_dim).init(device);
        let drop = DropoutConfig::new(self.embd_pdrop).init();
        let d_ff = if self.n_inner != 0 {
            self.n_inner
        } else {
            embed_dim * 4
        };
        let h = TransformerEncoderConfig::new(embed_dim, d_ff, self.n_head, self.n_layer).init(device);
        let ln_f = LayerNormConfig::new(embed_dim).with_epsilon(self.layer_norm_epsilon).init(device);

        let transformer = KoGPT2Model {
            embed_dim,
            wte,
            wpe,
            drop,
            h,
            ln_f,
            vocab_size: self.vocab_size,
            pad_token_id: self.pad_token_id,
        };
        let lm_head = LinearConfig::new(embed_dim, self.vocab_size).init(device);

        KoGPT2LMHeadModel {
            transformer,
            lm_head,
        }
    }
}

impl<B: Backend> KoGPT2LMHeadModel<B> {
    pub fn forward_training(&self, item: TextGenerationTrainingBatch<B>) -> ClassificationOutput<B>{
        let [batch_size, seq_length] = item.tokens_inputs.dims();
        let device = &self.devices()[0];

        let inputs = item.tokens_inputs.to_device(device);
        let targets = item.targets.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);

        let embedding_positions = self.transformer.wpe.forward(index_positions);
        let embedding_tokens = self.transformer.wte.forward(inputs);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let encoded = self.transformer.h.forward(TransformerEncoderInput::new(embedding)
            .mask_pad(mask_pad)
            .mask_attn(mask_attn));

        let output = self.lm_head.forward(encoded);
        let output_flatten = output.reshape([batch_size * seq_length, self.transformer.vocab_size]);
        let targets_flatten = targets.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.transformer.pad_token_id]))
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

        let embedding_positions = self.transformer.wpe.forward(index_positions);
        let embedding_tokens = self.transformer.wte.forward(inputs);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let encoded = self.transformer.h.forward(TransformerEncoderInput::new(embedding)
            .mask_pad(mask_pad)
            .mask_attn(mask_attn));

        let output = self.lm_head.forward(encoded);
        let output_flatten = output.reshape([batch_size * seq_length, self.transformer.vocab_size]);

        output_flatten
    }
}

impl<B: AutodiffBackend> TrainStep<TextGenerationTrainingBatch<B>, ClassificationOutput<B>> for KoGPT2LMHeadModel<B> {
    fn step(&self, item: TextGenerationTrainingBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_training(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<TextGenerationTrainingBatch<B>, ClassificationOutput<B>> for KoGPT2LMHeadModel<B> {
    fn step(&self, item: TextGenerationTrainingBatch<B>) -> ClassificationOutput<B> {
        self.forward_training(item)
    }
}
