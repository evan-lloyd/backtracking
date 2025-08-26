import pickle
import re

import pyarrow
from pyarrow import Int64Array, RecordBatch
from transformers import PreTrainedTokenizer

from ..dataset_transform import DatasetTransform
from ..storage import StorageType

STEP_BY_STEP_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
)


def serialize_tokenization(t):
    """This is EXTREMELY hacky, but the only info we need to store about a toknization is the token ids and
    offsets. However, huggingface tokenizers won't let us use the Rust implementation of char_to_token unless
    we have the raw encoding object, which has a bunch of extra fluff like the original tokenized text. We can
    hack into its serialization scheme by monkey-patching __getstate__ and pickling it, since all that does in
    the original source is dump the rust object's data into JSON. We minimize the resulting size by dumping
    a minimal JSON that passes validation, but also has the data we actually need."""

    def _hacked_get_state():
        return (
            f'{{"ids":{t.ids},"type_ids":[],"tokens":[],"words":[],"offsets":{[list(o) for o in t.offsets]},'
            f'"special_tokens_mask":[],"attention_mask":[],"overflowing":[],"sequence_ranges":{{}}}}'
        ).encode()

    t.__getstate__ = _hacked_get_state
    return pickle.dumps(t)


def deserialize_tokenization(s):
    return pickle.loads(s)


class Tokenize(DatasetTransform):
    storage_type = StorageType.persistent
    schema: pyarrow.schema = pyarrow.schema(
        {
            "raw_row_id": pyarrow.int64(),
            "prompt_len": pyarrow.int32(),
            "prefill": pyarrow.string(),
            # TODO: make this an extension type, we should just be able to save/restore a BatchEncoding!
            "serialized_tokenization": pyarrow.binary(),
        }
    )
    _COT_REGEX = re.compile(r"(<think>.*?</think>)", re.DOTALL)
    batch_size = 1_000
    min_rows_per_group = 1_000
    max_rows_per_group = 1_000

    def __init__(self, tokenizer: PreTrainedTokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    projection = {"messages", "raw_row_id"}
    def transform(self, in_batch: RecordBatch, row_indices: Int64Array) -> RecordBatch:
        in_batch = in_batch.to_pydict()
        out_batch = {k: [] for k in self.schema.names}
        # Explode and flatten messages
        for messages, index in zip(in_batch["messages"], in_batch["raw_row_id"]):
            cur_prompt = ""
            for message in messages:
                if message["role"] == "user":
                    cur_prompt = message["content"] + STEP_BY_STEP_SUFFIX
                else:
                    # Only save the chain of thought
                    think_match = self._COT_REGEX.match(message["content"])
                    # Skip any examples where the model didn't use the <think> tag properly
                    if think_match:
                        out_batch["raw_row_id"].append(index)
                        out_batch["prefill"].append(cur_prompt + think_match.group(1))
                        out_batch["prompt_len"].append(len(cur_prompt))

        # TODO: we should save the input_ids and offsets as their own columns, and do the deserialization
        # shenanigans based on those.
        # TODO: we should truncate to the max number of tokens, and truncate the prefill accordingly (guarantees we
        # won't match something that isn't going to yield activations)
        # NB: not converting to Tensor yet, because we'd have to pad, but we won't necessarily batch in
        # the same way when we run inference. We will manually pad in the next pipeline stage.
        out_batch["serialized_tokenization"] = [
            serialize_tokenization(t)
            for t in self.tokenizer(
                out_batch["prefill"],
                padding=False,
            )._encodings
        ]
        return RecordBatch.from_pydict(out_batch)
