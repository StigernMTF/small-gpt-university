import tokenizer_utils
tok = tokenizer_utils.getTokenizer()
print("pad_idx", tok.pad_token_id)
print("bos_idx", tok.bos_token_id)
print("eos_idx", tok.eos_token_id)
print("vocab_size", len(tok))
