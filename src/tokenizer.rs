use tokenizers::{Encoding, Tokenizer};

const START_TOKEN: &str = "[START]";
const END_TOKEN: &str = "[END]";
const PAD_TOKEN: &str = "[PAD]";

#[derive(Clone)]
pub struct GPT2Tokenizer {
    tokenizer: Tokenizer
}

impl Default for GPT2Tokenizer {
    fn default() -> Self {
        let mut tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        tokenizer.add_special_tokens(&[
            tokenizers::AddedToken::from(START_TOKEN, true),
            tokenizers::AddedToken::from(END_TOKEN, true),
            tokenizers::AddedToken::from(PAD_TOKEN, true),
        ]);

        Self { tokenizer }
    }
}

impl GPT2Tokenizer {
    pub fn encode(&self, value: &str, is_special: bool, compute_offset: bool) -> Vec<usize> {
        let text = Self::process_special(value, is_special);

        let tokens = if compute_offset {
            self.tokenizer.encode(text, true)
        } else {
            self.tokenizer.encode_fast(text, true)
        };
        let tokens = tokens.unwrap();
        tokens.get_ids().into_iter().map(|v| *v as usize).collect()
    }

    pub fn decode(&self, tokens: Encoding) -> String {
        self.tokenizer.decode(tokens.get_ids(), false).unwrap()
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn pad_token(&self) -> u32 {
        self.tokenizer.token_to_id(PAD_TOKEN).unwrap()
    }

    pub fn start_token(&self) -> u32 {
        self.tokenizer.token_to_id(START_TOKEN).unwrap()
    }
    pub fn end_token(&self) -> u32 {
        self.tokenizer.token_to_id(END_TOKEN).unwrap()
    }

    fn process_special(value: &str, is_special: bool) -> String {
        if is_special {
            let mut str = String::with_capacity(START_TOKEN.len() + value.len() + END_TOKEN.len());
            str.push_str(START_TOKEN);
            str.push_str(value);
            str.push_str(END_TOKEN);
            str
        } else {
            value.to_string()
        }
    }
}

impl AsRef<Tokenizer> for GPT2Tokenizer {
    fn as_ref(&self) -> &Tokenizer {
        &self.tokenizer
    }
}
