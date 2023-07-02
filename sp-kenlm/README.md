# How to train subword-language model by sentencepiece, kenlm
## Prelimnary
```bash
./install_kenlm-bin.sh
```
## Train subword tokenizer by sentencepiece
```bash
# must be line by line document jsonl
python train_sp.py \
--input_jsonl_filepath ${JSONL_FILEPATH:-"kowiki.json"} \
--model_prefix ${MODEL_PREFIX:-"ko.sp"} \
--vocab_size 40000 \
--max_sentence_length 10000 \
--split_by_whitespace
```

## Train n-gram subword language model by kenlm
### Pretokenize
```bash
python pretokenize.py \
--input_jsonl_filepath ${JSONL_FILEPATH:-"kowiki.json"} \
--input_model_filepath ${MODEL_FILEPATH:-"ko.sp.model"} \
--output_text_filepath ${TEXT_FILEPATH:-"kowiki.txt"}
```

### Train kenlm
```bash
tmp/kenlm/build/bin/lmplz -o 5 <${TEXT_FILEPATH:-"kowiki.txt"} >${MODEL_PREFIX:-"ko"}.arpa && \
tmp/kenlm/build/bin/build_binary ${MODEL_PREFIX:-"ko"}.arpa ${MODEL_PREFIX:-"ko"}.arpa.bin
```