from abc import ABC, abstractmethod
from transformers import AutoTokenizer
import tiktoken
from typing import List

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
