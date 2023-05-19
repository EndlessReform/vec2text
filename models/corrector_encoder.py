import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers


class CorrectorEncoderModel(torch.nn.Module):
    """Embeds text and concats with a provided embedding.
    
    TODO improve comment here.
    """

    encoder_decoder: transformers.PreTrainedModel

    def __init__(
        self, 
        encoder_decoder: transformers.PreTrainedModel,
        embedder_dim: int = 768,
        num_repeat_tokens: int = 16,
        bottleneck_dim: int = 768,
    ):
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.embedder_dim = embedder_dim
        self.num_repeat_tokens = num_repeat_tokens
        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.GELU(),  # TODO consider dropout or normalization here.
            nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens),
        )
    
    def get_encoder_embedding(
        self, 
        embedding: torch.Tensor,
        hypothesis_embedding: torch.Tensor,
        hypothesis_input_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hypothesis_input_ids.shape
        assert embedding.shape == (batch_size, 768)
        assert hypothesis_embedding.shape == (batch_size, 768)
        embedding = self.embedding_transform(embedding)
        embedding = embedding.reshape((batch_size, self.num_repeat_tokens, -1))
        hypothesis_embedding = self.embedding_transform(hypothesis_embedding)
        hypothesis_embedding = hypothesis_embedding.reshape((batch_size, self.num_repeat_tokens, -1))
        inputs_embeds = self.encoder_decoder.encoder.embed_tokens(hypothesis_input_ids)
        #
        ones = torch.ones((batch_size, 1), dtype=torch.long, device=hypothesis_input_ids.device)
        # TODO: pad_token_id or eos_token_id? Or does it not matter?
        sep_token = ones * self.encoder_decoder.config.eos_token_id
        sep_token = self.encoder_decoder.encoder.embed_tokens(sep_token)
        # TODO: support & ablate concatenation methods.
        inputs_embeds = torch.cat((sep_token, embedding, sep_token, hypothesis_embedding, inputs_embeds), dim=1)
        attention_mask = torch.cat((ones, ones.repeat(1, self.num_repeat_tokens), ones, ones.repeat(1, self.num_repeat_tokens), hypothesis_attention_mask), dim=1)
        return (inputs_embeds, attention_mask)

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit

        if "max_length" not in generation_kwargs:
            generation_kwargs["max_length"] = inputs["input_ids"].shape[1] + 1
        
        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=inputs["frozen_embeddings"],
            hypothesis_input_ids=inputs["hypothesis_input_ids"],
            hypothesis_attention_mask=inputs["hypothesis_attention_mask"],
            hypothesis_embedding=inputs["hypothesis_embedding"],
        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_input_ids=inputs["decoder_input_ids"],
                # decoder_attention_mask=inputs["decoder_attention_mask"],
                **generation_kwargs,
            )
        else:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                **generation_kwargs,
            )

    def forward(
        self,
        embedding: torch.Tensor,
        hypothesis_embedding,
        hypothesis_input_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=embedding,
            hypothesis_embedding=hypothesis_embedding,
            hypothesis_input_ids=hypothesis_input_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )