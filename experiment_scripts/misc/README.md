## Convenience scripts for precomputing embeddings

## Setup

Install repo dependencies with pip.

If you're using OpenAI proper, set your API key as `OPENAI_API_KEY` in environment variables or `.env`.

If you're using a model from the HF Hub:
1. Go to `infra/tei` 
2. In `docker-compose.yml`, change the `--model-id` being served to your desired model
3. `docker compose up`

## `create_train_dataset_openai.py`

Embeds a HF Dataset with your chosen embedding model (and optional local endpoint), truncating to the number of tokens to reconstruct.
**NOTE:** You will need to tokenize this again for the specific inversion model you're using (see below).

Saves to `datasets/$DATASET_ID__$MODEL_ID__$N_TOKENS` as uncompressed Arrow.

To run the embedding generation script, use the following command:

```
python embedding_generation_script.py [--dataset DATASET] [--splits {train,test,all}] [--embedder_name EMBEDDER_NAME] [--api_base_url API_BASE_URL] [--max_tokens MAX_TOKENS] [--save_to_hub] [--limit LIMIT]
```

**Arguments:**
- `--dataset DATASET`: Name of the Hugging Face dataset (default: "Tevatron/msmarco-passage-corpus").
- `--splits {train,test,all}`: Dataset splits to process (default: "train").
- `--embedder_name EMBEDDER_NAME`: Name of the embedder model (default: "text-embedding-ada-002").
- `--api_base_url API_BASE_URL`: Embedder API base URL (optional, defaults to OpenAI's API).
- `--max_tokens MAX_TOKENS`: Maximum number of tokens per embedding (default: 128).
- `--save_to_hub`: Save the embeddings to the Hugging Face Hub (optional).
- `--limit LIMIT`: Run on a subset of data to test the system (optional).

**Example:**
```
python embedding_generation_script.py --dataset "Tevatron/msmarco-passage-corpus" --splits "train" --embedder_name "text-embedding-ada-002" --max_tokens 128 --save_to_hub --limit 1000
```

**Warning:** HF Datasets caches work to the OS `/tmp` by default before saving to disk. `/tmp` can be MUCH smaller than a full embedded dataset (~50-100GB for MS MARCO). If you don't have enough space in RAM or swap, set the `TMPDIR` environment variable to somewhere on disk when running the command! Don't find this out the hard way.