from typing import Type

from transformers import PreTrainedTokenizerBase


def count_tokens(text, tokenizer: Type[PreTrainedTokenizerBase]) -> int:
    length = tokenizer(text, return_length=True, return_attention_mask=False, return_token_type_ids=False)["length"][0]
    return length
