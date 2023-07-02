import json
from argparse import ArgumentParser

import sentencepiece as spm
from kss import split_sentences
from tqdm import tqdm


def get_args():
    parser = ArgumentParser()
    group = parser.add_argument_group("input")
    group.add_argument("--input_jsonl_filepath", type=str, required=True, help="Raw corpus (jsonl) for training kenlm")
    group.add_argument(
        "--input_model_filepath", type=str, required=True, help="Tokenizers filepath trained by sentencepiece"
    )
    group.add_argument(
        "--output_text_filepath", type=str, required=True, help="Pretokenized corpus (txt) for training kenlm"
    )
    args = parser.parse_args()
    return args


def get_sentence_generator_from_document_jsonl(filepath, key="text"):
    io = open(filepath, mode="r", encoding="utf-8")

    for line in io:
        yield from split_sentences(json.loads(line)[key])


def main():
    args = get_args()
    gen = get_sentence_generator_from_document_jsonl(args.input_jsonl_filepath)

    sp = spm.SentencePieceProcessor()
    sp.load(args.input_model_filepath)

    with open(args.output_text_filepath, mode="w", encoding="utf-8") as io:
        for sentence in tqdm(gen):
            io.write(" ".join(sp.encode_as_pieces(sentence)) + "\n")


if __name__ == "__main__":
    main()
