from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SteeringFlag:
    on: bool
    directional_ablation: bool
    scale: Optional[float] = None


@contextmanager
def add_steering_vector(
    layer: int,
    vector: torch.Tensor,
    model: torch.nn.Module,
    steering: Optional[SteeringFlag] = None,
):
    vector = vector.to("cuda").to(torch.float16)
    unit_vector = vector / vector.norm()

    if steering is None:
        steering = SteeringFlag(True, False)

    def add_to_layer(_model, _args, _kwargs, output: torch.Tensor):
        if not steering.on and not steering.directional_ablation:
            return output
        # layer output is of shape [batch_idx, ctx_idx, hidden_idx]
        if steering.on:
            if steering.scale is not None:
                return (output[0] + unit_vector * steering.scale, *output[1:])
            return (output[0] + vector, *output[1:])

        if steering.directional_ablation:
            return (
                output[0]
                - (torch.dot(output[0], vector) / torch.dot(vector, vector) * vector),
                *output[1:],
            )

    with model.model.layers[layer].register_forward_hook(
        add_to_layer, with_kwargs=True
    ):
        yield
