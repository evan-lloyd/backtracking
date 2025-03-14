from contextlib import ExitStack, contextmanager
from typing import Dict, Iterable

import numpy
import pyarrow
import pyarrow.compute as compute
import pyarrow.dataset as pyarrow_dataset
import torch
from pyarrow import Int64Array, RecordBatch
from transformers import BatchEncoding, PreTrainedTokenizer

from ..dataset_transform import DatasetTransform
from ..numpy_array import NumpyArray, NumpyArrayType
from ..storage import StorageType
from .match_transform import TEXT_MATCH_SEGMENTS
from .token_transform import deserialize_tokenization


@contextmanager
def residuals_for_layers(model: torch.nn.Module, layers: Iterable[int]):
    cached_outputs: Dict[int, torch.Tensor] = {}

    def make_layer_hook(layer: int):
        def hook(_model: torch.nn.Module, _args, _kwargs, output: torch.Tensor):
            cached_outputs[layer] = output[0].to("cpu")
            return output

        return hook

    with ExitStack() as hook_stack:
        for layer in layers:
            hook_stack.enter_context(
                model.model.layers[layer].register_forward_hook(
                    make_layer_hook(layer), with_kwargs=True
                )
            )
        yield cached_outputs
    return


class Activation(DatasetTransform):
    storage_type = StorageType.temporary
    schema: pyarrow.Schema
    batch_size = 1
    max_rows_per_file = 1000
    min_rows_per_group = 100
    max_rows_per_group = 1000

    tokenizer: PreTrainedTokenizer
    match_dataset: pyarrow_dataset.Dataset
    hidden_size: int

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        match_dataset: pyarrow_dataset.Dataset,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.layer_range = range(len(self.model.model.layers))
        self.match_dataset = match_dataset
        self.hidden_size = self.model.config.hidden_size  # type: ignore
        self.activation_pyarrow_type = NumpyArrayType(numpy.float16)
        self.schema = pyarrow.schema(
            {
                "raw_row_id": pyarrow.int64(),
                "token_row_id": pyarrow.int64(),
                "match_row_id": pyarrow.int64(),
                # TODO: actually collect previous/next activations
                # **{
                #     f"previous_activations.{layer}"
                #     for layer in range(len(self.model.model.layers))
                # },
                **{
                    f"{name}_activations.{layer}": self.activation_pyarrow_type
                    for layer in self.layer_range
                    for name in TEXT_MATCH_SEGMENTS
                },
                # **{
                #     f"next_activations.{layer}"
                #     for layer in range(len(self.model.model.layers))
                # },
            }
        )

    projection = {
        "token_row_id",
        "serialized_tokenization",
    }

    def transform(self, in_batch: RecordBatch, row_indices: Int64Array) -> RecordBatch:
        # in_batch: {
        #      # From Match dataset
        #     "raw_row_id": pyarrow.int64(),
        #     "token_row_id": pyarrow.int64(),
        #     "match_type": pyarrow.string(),
        #     "context": pyarrow.string(),
        #     "prefix_text": pyarrow.string(),
        #     "prefix_token_range": pyarrow.list_(pyarrow.int32(), 2),
        #     "match_text": pyarrow.string(),
        #     "match_token_range": pyarrow.list_(pyarrow.int32(), 2),
        #     "suffix_text": pyarrow.string(),
        #     "suffix_token_range": pyarrow.list_(pyarrow.int32(), 2),
        #     }
        in_rows = in_batch.to_pylist()
        out_rows = []

        # Join against Match dataset to get tokenizations for the current batch
        self.mark("join")
        match_rows = self.match_dataset.to_table(
            filter=compute.field("token_row_id").isin(in_batch["token_row_id"]),
            columns=[
                "raw_row_id",
                "token_row_id",
                "match_row_id",
                *(f"{name}_token_range" for name in TEXT_MATCH_SEGMENTS),
            ],
        ).to_pylist()
        self.mark()

        self.mark("load tokenization")
        tokenizations = [
            deserialize_tokenization(in_row["serialized_tokenization"])
            for in_row in in_rows
        ]
        self.mark()
        # Pad (on left side) to length of the longest in the batch, using the beginofsequence token
        # (this doesn't really matter, since we generate an attention mask too)

        # TODO: for some reason we get very slightly different results when padding vs not, but not ENOUGH
        # different to make me think there's an error in calculating the token offset.
        # possibly a known artifact of running LLMs in batch mode? or because we're using quantization?

        self.mark("padding")
        pad_to_length = max(len(t.ids) for t in tokenizations)
        # Keep track of how much we padded each input, because we will need to add it to the token offsets
        pad_length = [pad_to_length - len(t.ids) for t in tokenizations]
        batch_encoding = (
            BatchEncoding(
                {
                    "input_ids": [
                        [t.ids[0]] * p + t.ids
                        for t, p in zip(tokenizations, pad_length)
                    ],
                    "attention_mask": [
                        [0] * p + [1] * len(t.ids)
                        for t, p in zip(tokenizations, pad_length)
                    ],
                }
            )
            .convert_to_tensors("pt")
            .to("cuda")
        )
        self.mark()

        self.mark("inference")
        with (
            torch.inference_mode(),
            residuals_for_layers(self.model, self.layer_range) as residuals,
        ):
            # NB: not storing any info about the logits, because we can easily compute them from
            # the last layer residual!
            self.model(
                **batch_encoding,
                use_cache=False,
            )[0]
        self.mark()

        self.mark("process rows")
        for in_row_index, in_row in enumerate(in_rows):
            for match_row in match_rows:
                if match_row["token_row_id"] != in_row["token_row_id"]:
                    continue
                new_row = {
                    "raw_row_id": match_row["raw_row_id"],
                    "token_row_id": match_row["token_row_id"],
                    "match_row_id": match_row["match_row_id"],
                }
                for segment in TEXT_MATCH_SEGMENTS:
                    # NB: subtracting 1 from the token index, because we want the activation from when
                    # the model was *predicting* the target token. We must also account for left padding,
                    # which we add as an additional offset.
                    token_slice = slice(
                        *(
                            t - 1 + pad_length[in_row_index]
                            for t in match_row[f"{segment}_token_range"]
                        )
                    )
                    for layer in self.layer_range:
                        new_row[f"{segment}_activations.{layer}"] = (
                            NumpyArray.from_numpy(
                                residuals[layer][in_row_index, token_slice, :].numpy()
                            )
                        )
                out_rows.append(new_row)
                # for each match_row

            # NB: just calling out that we de-indented twice before the next actual line
            pass
            # for each in_row
        self.mark()

        return RecordBatch.from_pylist(out_rows, self.schema)
