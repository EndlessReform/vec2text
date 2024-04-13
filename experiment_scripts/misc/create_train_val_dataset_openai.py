from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
import os
from typing import Dict, List, Optional
import threading

import datasets
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
from tenacity import retry, stop_after_attempt, wait_fixed
from vec2text.utils import get_num_proc
from vec2text.data_helpers import retain_dataset_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding Generation Script")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Tevatron/msmarco-passage-corpus",
        help="Name of the Hugging Face dataset (default: Tevatron/msmarco-passage-corpus)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="Dataset splits to process (default: train)",
    )
    parser.add_argument(
        "--embedder_name",
        type=str,
        default="text-embedding-ada-002",
        help="Name of the embedder model (default: text-embedding-ada-002)",
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default=None,
        help="Embedder API base URL (optional, defaults to OAI)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens per embedding (default: 128)",
    )
    parser.add_argument(
        "--save_to_hub",
        action="store_true",
        help="Save the embeddings to the Hugging Face Hub",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run on subset of data to test system",
    )
    return parser.parse_args()


OAI_EMBEDDING_MODELS = [
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
]


class AbstractTokenizer(ABC):
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        pass

    @abstractmethod
    def decode_batch(self, token_ids: List[List[int]]) -> List[str]:
        pass


class TiktokenTokenizer(AbstractTokenizer):
    def __init__(self, model_name: str):
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return self.tokenizer.encode_batch(texts)

    def decode_batch(self, token_ids: List[List[int]]) -> List[str]:
        return self.tokenizer.decode_batch(token_ids)


class HFTokenizer(AbstractTokenizer):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return self.tokenizer(texts)["input_ids"]

    def decode_batch(self, token_ids: List[List[int]]) -> List[str]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def load_encoder(embedder_name: str) -> AbstractTokenizer:
    if embedder_name in OAI_EMBEDDING_MODELS:
        return TiktokenTokenizer(embedder_name)
    else:
        # Try HF hub
        return HFTokenizer(embedder_name)


thread_local = threading.local()


@dataclass
class OpenAIParams:
    api_key: Optional[str]
    base_url: Optional[str]


def get_client(params: OpenAIParams):
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(
            api_key=params.api_key,
            base_url=params.base_url,
        )
    return thread_local.client


def load_train_split(dataset_id: str) -> datasets.Dataset:
    # has columns ["title", "text"]. only one split ("train")
    dataset_dict = datasets.load_dataset(dataset_id)
    return dataset_dict["train"]


def tokenize_row(encoder: AbstractTokenizer, max_length: int, example: Dict):
    text_tokens = encoder.encode_batch(example["text"])
    text_tokens = [passage[:max_length] for passage in text_tokens]
    text_list = encoder.decode_batch(text_tokens)
    example["text"] = text_list
    return example


@retry(wait=wait_fixed(1), stop=stop_after_attempt(2))
def embed_row(params: OpenAIParams, model: str, example: Dict) -> Dict:
    client = get_client(params)
    # Assume tokenization and embedding batches are the same
    response = client.embeddings.create(
        input=example["text"], model=model, encoding_format="float"
    )
    embeddings = [e.embedding for e in response.data]
    example["embedding"] = embeddings
    return example


def main():
    args = parse_args()
    print(args)

    full_name = "__".join(
        (
            args.dataset,
            "openai_ada2",
        )
    )

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if (
        args.embedder_name in OAI_EMBEDDING_MODELS
        and api_key is None
        and args.api_base_url is None
    ):
        print("API key required to use OpenAI embeddings!")
        exit(1)
    elif args.embedder_name not in OAI_EMBEDDING_MODELS and args.api_base_url is None:
        print("Model not from OpenAI; third-party endpoint required")
        exit(1)

    encoder = load_encoder(args.embedder_name)
    print(f"[*] attempting to load {args.dataset}")
    dataset = load_train_split(args.dataset)
    if args.limit is not None:
        # TODO: parameterize subset size
        dataset = dataset.select(range(args.limit))

    print(f"[*] tokenizing {args.dataset}")
    dataset = dataset.map(
        lambda e: tokenize_row(encoder=encoder, max_length=args.max_tokens, example=e),
        batched=True,
        batch_size=2048,
        num_proc=get_num_proc(),
    )

    print(f"[*] embedding {args.dataset}")
    dataset = dataset.map(
        lambda e: embed_row(
            OpenAIParams(api_key=api_key, base_url=args.api_base_url),
            args.embedder_name,
            example=e,
        ),
        batched=True,
        batch_size=256,
        num_proc=get_num_proc(),
    )
    dataset = retain_dataset_columns(dataset, ["text", "embedding"])


if __name__ == "__main__":
    main()
