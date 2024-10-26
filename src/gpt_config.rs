use std::collections::HashMap;
use std::num::NonZeroUsize;
use burn::config::Config;
use serde::{Serialize,
            Deserialize};

#[derive(Config)]
pub struct GptConfig {
    pub architectures: Vec<String>,

    #[config(default = 768)]
    pub n_embd: usize,
    #[config(default = 0)]
    pub n_inner: usize,

    #[config(default = true)]
    pub use_cache: bool,

    #[config(default = 0.1)]
    pub attn_pdrop: f64,

    pub summary_type: String,

    #[config(default = 50256)]
    pub eos_token_id: usize,

    #[config(default = 0.1)]
    pub resid_pdrop: f64,

    pub transformers_version: String,

    #[config(default = 5021257)]
    pub vocab_size: usize,

    #[config(default = 1024)]
    pub n_positions: usize,

    pub id2_label: HashMap<usize,
        String>,

    #[config(default = true)]
    pub summary_proj_to_labels: bool,

    pub num_labels: usize,

    pub gradient_checkpointing: bool,

    pub task_specific_params: TaskSpecificParams,

    #[config(default = true)]
    pub summary_use_proj: bool,

    pub author: String,

    pub pad_token_id: usize,

    pub model_type: String,

    #[config(default = 12)]
    pub n_layer: usize,

    #[config(default = 0.02)]
    pub initializer_range: f64,

    pub n_ctx: usize,

    pub label2_id: HashMap<String,
        usize>,

    #[config(default = 50257)]
    pub n_head: usize,

    pub license: String,

    #[config(default = 50256)]
    pub bos_token_id: usize,

    pub activation_function: String,

    #[config(default = 1e-5)]
    pub layer_norm_epsilon: f64,

    #[config(default = 0.1)]
    pub summary_first_dropout: f64,

    pub created_date: String,

    #[config(default = 0.1)]
    pub embd_pdrop: f64,
}

#[derive(Config)]
pub struct TaskSpecificParams {
    pub text_generation: TextGeneration,
}

#[derive(Config)]
pub struct TextGeneration {
    pub do_sample: bool,
    pub max_length: usize,
}
