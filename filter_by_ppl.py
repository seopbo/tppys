"""
This codes is just for proof of concept
written by @seopbo
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import FloatType

from tppys.model import KenlmModel


def get_args():
    parser = ArgumentParser()
    io_group = parser.add_argument_group("io")
    io_group.add_argument("--input_dirpath")
    io_group.add_argument("--input_file_extension", default="gz")
    io_group.add_argument("--output_dirpath")
    asset_group = parser.add_argument_group("asset")
    asset_group.add_argument(
        "--path_kenlm_model", type=str, default="./ko.arpa.bin", help="Executing by spark-submit, keep default value."
    )
    asset_group.add_argument(
        "--path_sentencepiece_model",
        type=str,
        default="./ko.sp.model",
        help="Executing by spark-submit, keep default value.",
    )
    asset_group.add_argument("--ppl_upperbound", type=float, default=300)
    spark_group = parser.add_argument_group("spark")
    spark_group.add_argument("--executor_memory", type=str, default="2g")
    args = parser.parse_args()
    return args


def get_spark_session(args):
    spark = (
        SparkSession.builder.appName(f"filter-corpus: {args.input_dirpath}")
        .config("spark.executor.memory", args.executor_memory)
        .getOrCreate()
    )
    return spark


def main():
    args = get_args()
    spark = get_spark_session(args)
    list_of_filepaths = [str(filepath) for filepath in Path(args.input_dirpath).glob(f"*.{args.input_file_extension}")]
    df = spark.read.json(list_of_filepaths)

    model = KenlmModel.from_pretrained(args.path_kenlm_model, args.path_sentencepiece_model)

    @pandas_udf(FloatType())
    def calculate_ppl_udf(text: pd.Series) -> pd.Series:
        ppl = text.apply(lambda string: model.get_perplexity(string))
        return ppl

    processed_df = df.withColumn("perplexity", calculate_ppl_udf(col("text"))).persist(StorageLevel.MEMORY_AND_DISK)
    num_rows = processed_df.count()
    filtered_df = processed_df.filter(col("perplexity") <= args.ppl_upperbound).select(col("text"))
    num_filtered_rows = filtered_df.count()
    filtered_df.write.option("compression", "gzip").mode("errorifexists").format("json").save(args.output_dirpath)

    statistics = {
        "input": {
            "list_of_filepaths": list_of_filepaths,
        },
        "processed": {
            "ppl_upperbound": args.ppl_upperbound,
            "num_rows": num_rows,
            "num_filtered_rows": num_filtered_rows,
        },
    }

    with open(f"{args.output_dirpath}/statistics.json", mode="w", encoding="utf-8") as io:
        json.dump(statistics, io, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
