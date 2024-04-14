import argparse
import os
from typing import Dict

import datasets
from vec2text.data_helpers import retain_dataset_columns
from vec2text.experiments import get_dataset_cache_path
from vec2text.utils import get_num_proc
from vec2text.utils.tokenizer import load_encoder, AbstractTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize precomputed embeddings")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Tevatron/msmarco-passage-corpus",
        help="Name of the Hugging Face dataset (default: Tevatron/msmarco-passage-corpus)",
    )
    parser.add_argument(
        "--embedder_name",
        type=str,
        default="text-embedding-ada-002",
        help="Name of the embedder model (default: text-embedding-ada-002)",
    )
    parser.add_argument(
        "--inverter_model_id",
        type=str,
        default="google/flan-t5-base",
        help="Name of the inverter model (default: google/flan-t5-base)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens per passage (default: 128)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run on subset of data to test system",
    )
    return parser.parse_args()


def load_dataset_from_file(file_path: str) -> datasets.Dataset:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    dataset = datasets.load_from_disk(file_path)

    if "length" not in dataset.column_names:
        raise ValueError("Need pre-truncated dataset with 'length' column")

    return dataset


def tokenize_row(encoder: AbstractTokenizer, max_length: int, example: Dict):
    input_ids = encoder.encode_batch(example["text"])
    example["input_ids"] = input_ids
    return example


def main():
    args = parse_args()
    print(args)

    def escape_id(id: str):
        return id.replace("/", "_")

    encoder = load_encoder(args.inverter_model_id)
    print(f"[*] attempting to load {args.dataset}")
    file_path = f"{get_dataset_cache_path()}/{escape_id(args.dataset)}__{escape_id(args.embedder_name)}__{args.max_tokens}"

    try:
        dataset = load_dataset_from_file(file_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading dataset: {str(e)}")
        exit(1)

    print(f"[*] tokenizing {args.dataset}")
    dataset = dataset.map(
        lambda e: tokenize_row(encoder=encoder, max_length=args.max_tokens, example=e),
        batched=True,
        batch_size=2048,
        num_proc=get_num_proc(),
    )
    print(args.limit)
    if args.limit is not None:
        dataset = dataset.select(range(args.limit))

    dataset = retain_dataset_columns(dataset, ["input_ids"])

    if args.limit is None:
        dataset.save_to_disk(
            f"{get_dataset_cache_path()}/inverter_tokens__{escape_id(args.dataset)}__{escape_id(args.inverter_model_id)}__{args.max_tokens}",
            max_shard_size="5GB",
        )


if __name__ == "__main__":
    main()
