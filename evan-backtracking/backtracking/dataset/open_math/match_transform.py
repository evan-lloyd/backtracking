import re
from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

import pyarrow
from pyarrow import Int64Array, RecordBatch

from ..dataset_transform import DatasetTransform
from ..storage import StorageType
from .token_transform import deserialize_tokenization

STEP_BY_STEP_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
)
TEXT_MATCH_SEGMENTS = ("prefix", "match", "suffix")


@dataclass(kw_only=True)
class TextMatchPattern:
    prefix: str
    match: str
    suffix: str
    # Collect activations on this many tokens before the prefix
    previous_tokens_context: int = 5
    # Collect activations on this many tokens after the suffix
    next_tokens_context: int = 5
    # Stop matching this pattern after this many matches within an example
    # TODO: some smarter heuristic than "just the first n" (random?)
    max_per_example: Optional[int] = None
    re_flags: re.RegexFlag = re.DOTALL
    expression: re.Pattern = field(init=False)

    # Do we expect our matches to line up with token boundaries?
    expect_full_token_prefix: bool = False
    expect_full_token_match: bool = False
    expect_full_token_suffix: bool = False
    # Raise an exception, rather than dropping the example, if the boundaries don't match
    validate_token_boundaries: bool = False

    def __post_init__(self):
        self.expression = re.compile(
            "".join(f"(?P<{p}>{getattr(self, p)})" for p in TEXT_MATCH_SEGMENTS),
            self.re_flags,
        )


SENTENCE_PREFIX = r"<think>|[.!?]\n*"

TEXT_MATCH_PATTERNS: Mapping[
    Literal["sentence_start", "backtracking_candidate"], TextMatchPattern
] = {
    "sentence_start": TextMatchPattern(
        prefix=SENTENCE_PREFIX,
        match=r" ?[A-Z][a-z]+",
        suffix=r"",
        max_per_example=10,
    ),
    "backtracking_candidate": TextMatchPattern(
        # Special cases: "...no, .no, .No
        # (these all use unusual tokens, which maybe makes them more important to catch examples of?)
        # The lookahead prevents the prefix from consuming an extra period from the normal sentence pattern.
        prefix=r"(?=\"?\.+no)|" + SENTENCE_PREFIX,
        # NB: the optional weird punctuation in front of some branches is to hit some alternative tokenizations
        # that show up sometimes in our dataset.
        match=r" ?(\.*Wait|\"?\.*No|Nope|Actually|Hold|Hang|Oops|Sorry)\b",
        suffix=r"",
        re_flags=re.DOTALL | re.IGNORECASE,
        expect_full_token_match=True,
        validate_token_boundaries=True,
    ),
}


class Match(DatasetTransform):
    storage_type = StorageType.persistent
    schema = pyarrow.schema(
        {
            "raw_row_id": pyarrow.int64(),
            "token_row_id": pyarrow.int64(),
            "match_type": pyarrow.string(),
            "context": pyarrow.string(),
            "prefix_text": pyarrow.string(),
            "prefix_token_range": pyarrow.list_(pyarrow.int32(), 2),
            "match_text": pyarrow.string(),
            "match_token_range": pyarrow.list_(pyarrow.int32(), 2),
            "suffix_text": pyarrow.string(),
            "suffix_token_range": pyarrow.list_(pyarrow.int32(), 2),
        }
    )
    # Include this many chars before the prefix and after the suffix as context
    _MATCH_CONTEXT_RADIUS = 16
    batch_size = 100_000
    min_rows_per_group = 100_000
    max_rows_per_group = 1_000_000
    max_rows_per_file = 1_000_000

    projection = {
        "raw_row_id",
        "token_row_id",
        "prefill",
        "prompt_len",
        "serialized_tokenization",
    }

    def _extract_match(self, tokenization, match_span):
        match_token_span = (
            tokenization.char_to_token(match_span[0]),
            # By Python convention, the end of a span is *not* included, so we want
            # to find which token our actual last character is in, and then *add one*
            # to that token index, because that is the first token that is *not* included.
            tokenization.char_to_token(match_span[1] - 1) + 1,
        )
        tokenization_char_span = (
            tokenization.token_to_chars(match_token_span[0])[0],
            tokenization.token_to_chars(match_token_span[1] - 1)[1],
        )
        return match_token_span, tokenization_char_span

    def transform(self, in_batch: RecordBatch, row_indices: Int64Array) -> RecordBatch:
        # in_batch: {
        #     "raw_row_id": pyarrow.int64(),
        #     "prompt_len": pyarrow.int32(),
        #     "prefill": pyarrow.string(),
        #     "serialized_tokenization": pyarrow.binary(),
        # }
        in_batch = in_batch.to_pydict()
        out_batch = {k: [] for k in self.schema.names}

        # This is super deeply nested, so for readability use a generator to flatten
        # into a single for loop with the actual transformation logic.
        #
        # for each row:
        #   for each pattern:
        #       for each match:
        #           for (prefix, match, suffix):
        def flat_iterator():
            for batch_index in range(len(row_indices)):
                for pattern_name, pattern in TEXT_MATCH_PATTERNS.items():
                    yield (
                        in_batch["raw_row_id"][batch_index],
                        in_batch["token_row_id"][batch_index],
                        in_batch["prompt_len"][batch_index],
                        in_batch["prefill"][batch_index],
                        deserialize_tokenization(
                            in_batch["serialized_tokenization"][batch_index]
                        ),
                        0,  # init num_matches
                        pattern_name,
                        pattern,
                    )

        for (
            raw_row_id,
            token_row_id,
            prompt_len,
            prefill,
            tokenization,
            num_matches,
            pattern_name,
            pattern,
        ) in flat_iterator():
            for match in pattern.expression.finditer(prefill, prompt_len):
                if (
                    pattern.max_per_example is not None
                    and num_matches >= pattern.max_per_example
                ):
                    break
                new_row = {
                    "raw_row_id": raw_row_id,
                    "token_row_id": token_row_id,
                    "match_type": pattern_name,
                }
                match_failed = False
                for segment in TEXT_MATCH_SEGMENTS:
                    if match_failed:
                        break
                    match_token_span, tokenization_char_span = self._extract_match(
                        tokenization, match.span(segment)
                    )

                    # Optionally, throw an exception if our pattern didn't exactly map to token boundaries,
                    # which could indicate we need to tweak the pattern.
                    if getattr(pattern, f"expect_full_token_{segment}"):
                        if tokenization_char_span != match.span(segment):
                            if pattern.validate_token_boundaries:
                                raise ValueError(
                                    (
                                        f"Match ({segment}) did not line up with token boundaries:\n"
                                        f"Original text: {prefill[slice(*match.span(segment))]}\n"
                                        f"Tokenized text: {prefill[slice(*tokenization_char_span)]}\n"
                                    )
                                )
                            else:
                                # Just skip this example. This isn't perfect, since maybe if eg, we were
                                # less greedy in the regex, we *could* have aligned with token boundaries.
                                match_failed = True

                    new_row["context"] = prefill[
                        match.span("prefix")[0]
                        - self._MATCH_CONTEXT_RADIUS : match.span("suffix")[1]
                        + self._MATCH_CONTEXT_RADIUS
                    ]
                    new_row[f"{segment}_token_range"] = match_token_span
                    new_row[f"{segment}_text"] = match.group(segment)
                    # for prefix/match/suffix
                if not match_failed:
                    num_matches += 1
                    for k, v in new_row.items():
                        out_batch[k].append(v)
                # for match

        return RecordBatch.from_pydict(out_batch)
