use crate::tokenizer::GPT2Tokenizer;
use burn::data::dataloader::batcher::Batcher;
use burn::nn::attention::generate_padding_mask;
use burn::prelude::Backend;
use burn::tensor::{Bool, Int, Tensor};
use serde::{Deserialize, Serialize};
use std::ops::RangeInclusive;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct TextGenerationBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

#[derive(Debug, Clone)]
pub struct TextGenerationTrainingBatch<B: Backend> {
    pub tokens_inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

#[derive(Clone)]
pub struct TextGenerationBatcher {
    pub tokenizer: Arc<GPT2Tokenizer>,
    pub max_seq_length: usize
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TestCase {
    pub text: String,
    pub range: RangeInclusive<usize>,
    pub replacement: String,
    pub reason: String
}

#[derive(Clone, Debug)]
pub struct TextGenerationItem {
    pub text: String
}

impl<B: Backend> Batcher<TextGenerationItem, TextGenerationBatch<B>> for TextGenerationBatcher {
    fn batch(&self, items: Vec<TextGenerationItem>) -> TextGenerationBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        for item in items {
            tokens_list.push(self.tokenizer.encode(&item.text, true, true));
        }

        let mask = generate_padding_mask(
            self.tokenizer.pad_token() as usize,
            tokens_list,
            Some(self.max_seq_length),
            &B::Device::default()
        );
        
        TextGenerationBatch {
            tokens: mask.tensor,
            mask_pad: mask.mask,
        }
    }
}

impl<B: Backend> Batcher<TextGenerationItem, TextGenerationTrainingBatch<B>>
for TextGenerationBatcher
{
    fn batch(&self, items: Vec<TextGenerationItem>) -> TextGenerationTrainingBatch<B> {
        let item: TextGenerationBatch<B> = self.batch(items);
        let [batch_size, seq_length] = item.tokens.dims();

        let inputs = item
            .tokens
            .clone()
            .slice([0..batch_size, 0..seq_length - 1]);
        let targets = item.tokens.slice([0..batch_size, 1..seq_length]);
        let mask_pad = item.mask_pad.slice([0..batch_size, 0..seq_length - 1]);

        TextGenerationTrainingBatch {
            tokens_inputs: inputs,
            targets,
            mask_pad,
        }
    }
}
