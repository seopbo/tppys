# https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
import json
from argparse import ArgumentParser

import sentencepiece as spm
from kss import split_sentences


def get_args():
    parser = ArgumentParser()
    group = parser.add_argument_group("input")
    group.add_argument("--input_jsonl_filepath", type=str, required=True, help="Corpus for training subword tokenizer")
    group.add_argument(
        "--max_sentence_length",
        type=int,
        default=4192,  # library default value 4192
        help="Ignore line whose length is longer than {max_sentence_length}}",
    )
    group = parser.add_argument_group("Randomizing training data")
    group.add_argument("--input_sentence_size", type=int, default=-1)
    group.add_argument("--shuffle_input_sentence", action="store_true")
    group = parser.add_argument_group("Model specification")
    group.add_argument("--vocab_size", type=int, default=40000, help="Vocab size")
    group.add_argument(
        "--model_prefix",
        type=str,
        required=True,
        help="Prefix of outputs, e.g. {model_prefix}.model, {model_prefix}.vocab",
    )
    group.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe", "word", "char"])
    group = parser.add_argument_group("Text normalization")
    parser.add_argument(
        "--normalization_rule_name",
        type=str,
        default="nmt_nfkc",
        choices=["nmt_nfkc", "nfkc", "nmt_nfkc_cf", "nfkc_cf", "identity"],
    )
    group = parser.add_argument_group(title="Extracting crossing-words pieces")
    group.add_argument("--split_by_whitespace", action="store_true")
    args = parser.parse_args()
    return args


def get_sentence_generator_from_document_jsonl(filepath, key="text"):
    io = open(filepath, mode="r", encoding="utf-8")

    for line in io:
        yield from split_sentences(json.loads(line)[key])


def main():
    args = get_args()

    gen = get_sentence_generator_from_document_jsonl(args.input_jsonl_filepath)
    if args.input_sentence_size == -1:
        spm.SentencePieceTrainer.train(
            sentence_iterator=gen,
            max_sentence_length=args.max_sentence_length,
            vocab_size=args.vocab_size,
            model_prefix=args.model_prefix,
            model_type=args.model_type,
            normalization_rule_name=args.normalization_rule_name,
            split_by_whitespace=args.split_by_whitespace,
        )
    else:
        spm.SentencePieceTrainer.train(
            sentence_iterator=gen,
            vocab_size=args.vocab_size,
            model_prefix=args.model_prefix,
            model_type=args.model_type,
            normalization_rule_name=args.normalization_rule_name,
            split_by_whitespace=args.split_by_whitespace,
            input_sentence_size=args.input_sentence_size,
            shuffle_input_sentence=args.shuffle_input_sentence,
        )


if __name__ == "__main__":
    main()
