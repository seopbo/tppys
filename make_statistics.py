"""
This codes is just for proof of concept
written by @seopbo
"""
import json
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import IntegerType
from transformers import AutoTokenizer

from tppys.preprocessing import count_tokens


def get_args():
    parser = ArgumentParser()
    io_group = parser.add_argument_group("io")
    io_group.add_argument("--input_dirpath")
    io_group.add_argument("--input_file_extension")
    io_group.add_argument("--pretrained_model_name_or_path")
    io_group.add_argument("--output_dirpath")
    spark_group = parser.add_argument_group("spark")
    spark_group.add_argument("--executor_memory")
    args = parser.parse_args()
    return args


def get_spark_session(args):
    spark = (
        SparkSession.builder.appName(f"make-statistics: {args.input_dirpath}")
        .config("spark.executor.memory", args.executor_memory)
        .getOrCreate()
    )
    return spark


def main():
    args = get_args()
    spark = get_spark_session(args)
    list_of_filepaths = [str(filepath) for filepath in Path(args.input_dirpath).glob(f"*.{args.input_file_extension}")]
    df = spark.read.json(list_of_filepaths)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    count_tokens_ = partial(count_tokens, tokenizer=tokenizer)

    @pandas_udf(IntegerType())
    def count_tokens_udf(text: pd.Series) -> pd.Series:
        num_tokens = text.apply(lambda string: count_tokens_(string))
        return num_tokens

    df = df.withColumn("num_tokens", count_tokens_udf(col("text")))  # transformation

    pdf = df.agg(
        F.count(col("num_tokens")).alias("num_documents"),
        F.sum(col("num_tokens")).alias("total_tokens"),
        F.mean(col("num_tokens")).alias("tokens_per_document"),
    ).toPandas()

    with open(f"{args.output_dirpath}/statistics.json", mode="w", encoding="utf-8") as io:
        json.dump(pdf.loc[0].to_dict(), io, ensure_ascii=False, indent=4)
