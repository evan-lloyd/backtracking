from typing import List

import numpy
import torch
from sae_lens import SAE
from torch import Tensor, no_grad
from torch.nn import CosineSimilarity
from transformers import PreTrainedTokenizer, TextStreamer


def token_probs(
    text: List[str],
    activations,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
):
    """Convert final layer activations into a probability distribution over vocabulary."""
    token_ids = [tokenizer(t, add_special_tokens=False).input_ids for t in text]
    with torch.inference_mode():
        logits = model.lm_head(
            model.model.norm(torch.tensor(activations).to("cuda"))
        ).to("cpu")
        probs = logits.softmax(-1)
        return {
            text[j]: [probs[0, ids[i]].item() for i in range(len(ids))]
            for j, ids in enumerate(token_ids)
        }


def compare_activation(vector_name: str, vector: Tensor, sae: SAE):
    print(f'Cosine similarity between {sae.cfg.neuronpedia_id} and "{vector_name}"')
    with no_grad():
        sim = CosineSimilarity()(sae.W_dec, vector.unsqueeze(0)).to("cpu").numpy()
    top_part = numpy.argpartition(sim, sim.shape[0] - 10, 0)[-10:][::-1]
    print("Most correlated SAE features")
    for t in zip(top_part, numpy.take_along_axis(sim, top_part, 0)):
        print(f"  SAE feature {t[0].item()}, cosine sim: {t[1].item()}")

    bottom_part = numpy.argpartition(sim, 10, 0)[:10]
    print("Most anti-correlated SAE features")
    for t in zip(bottom_part, numpy.take_along_axis(sim, bottom_part, 0)):
        print(f"  SAE feature {t[0].item()}, cosine sim: {t[1].item()}")


def generate(inputs, model, tokenizer, stream=True, stream_callback=None):
    if isinstance(inputs, str):
        inputs = tokenizer(inputs, return_tensors="pt").to("cuda")

    class TextStreamerWithCallback(TextStreamer):
        def put(self, value):
            super().put(value)

            if stream_callback:
                stream_callback(self.tokenizer.decode(value[0], **self.decode_kwargs))

    streamer = TextStreamerWithCallback(tokenizer, skip_prompt=True) if stream else None

    with torch.inference_mode():
        return model.generate(
            **inputs,
            max_length=16384,
            streamer=streamer,
            stop_strings=["</think>"],
            tokenizer=tokenizer,
        )
