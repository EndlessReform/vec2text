# text-embeddings-inference local server

Convenience `docker-compose.yml` for precomputing local embeddings with [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference).

Requires Docker and Nvidia Container Toolkit.

Change `--model-id` line to your desired OSS embedder from the Hub. Set to [nomic-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) by default.