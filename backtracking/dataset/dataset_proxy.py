from __future__ import annotations

import re

# Available in python >= 3.12
try:
    from itertools import batched
except ImportError:
    from itertools import islice

    def batched(iterable, n):
        "Batch data into lists of length n. The last batch may be shorter."
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError("n must be >= 1")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch


from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy
import pyarrow
import pyarrow.compute as compute
import pyarrow.dataset as pyarrow_dataset
from pyarrow.dataset import Dataset
from tqdm.auto import tqdm

from .dataset_stage import DatasetStage

if TYPE_CHECKING:
    from .activation_dataset import ActivationDataset


class JoinedDatasetProxy:
    def __init__(
        self,
        self_dataset: DatasetProxy,
        other_dataset: DatasetProxy,
    ):
        self.self_dataset = self_dataset


class DatasetProxy:
    # TODO: might want to eagerly cache the first N rows so this will feel snappy to use
    # if just sanity checking a recent run.
    # TODO: this whole thing is awkward. probably we just want a first-class DatasetStage object.
    def __init__(
        self, dataset: Dataset, stage: DatasetStage, parent: ActivationDataset
    ):
        self.stage = stage
        self.dataset = dataset
        self.parent = parent

    def dataset_join(
        self,
        other: DatasetProxy,
        join_column: str,
        self_filter: Optional[compute.Expression] = None,
        other_filter: Optional[compute.Expression] = None,
        projection: Optional[List[str]] = None,
    ):
        # TODO: do a join in a way that gives us a dataset
        raise NotImplementedError()

    def batched_join(
        self,
        other: DatasetProxy,
        join_column: str,
        self_filter: Optional[compute.Expression] = None,
        other_filter: Optional[compute.Expression] = None,
        batch_size: int = 1_000,
        self_projection: Optional[Sequence[str | re.Pattern]] = None,
        other_projection: Optional[Sequence[str | re.Pattern]] = None,
        limit: Optional[int] = None,
        progress_desc: Optional[str] = None,
    ):
        """projection accepts strs or regex patterns; the latter are expanded to all matching column names."""
        dataset = object.__getattribute__(self, "dataset")

        # Set up column projections; make sure the ones we pass to the dataset methods include the
        # join_column, since we will need it if we're filtering the other dataset.
        def glob_projection(
            projection: Optional[Sequence[str | re.Pattern]],
            for_dataset: DatasetProxy,
            default_value: List[str],
        ) -> List[str]:
            if projection is None:
                return default_value
            globbed_projection: List[str] = []
            for p in projection:
                if isinstance(p, str):
                    globbed_projection.append(p)
                    continue
                globbed_projection.extend(
                    c for c in [f.name for f in for_dataset.schema] if p.match(c)
                )
            # TODO (minor optimization): seems like in some cases (no other filter?) we don't actually need to
            # fetch the join_column, since we could use take() instead of filter() to get the rows we need
            return list(set([*globbed_projection, join_column]))

        self_projection = glob_projection(
            self_projection, self, [f.name for f in self.schema]
        )
        other_projection = glob_projection(other_projection, other, [join_column])

        merged_schema = pyarrow.schema(
            f for f in self.schema if f.name in self_projection
        )
        for field in other.schema:
            if field.name in other_projection and field.name != join_column:
                merged_schema = merged_schema.append(
                    pyarrow.field(f"other.{field.name}", field.type)
                )

        # Finally, get batches from the join
        progress = tqdm(
            total=dataset.count_rows(filter=self_filter),
            desc=progress_desc or f"Joining on {join_column}",
            smoothing=0,
        )
        num_returned = 0
        with progress:
            self_join_ids = dataset.to_table(columns=[join_column], filter=self_filter)
            cur_filter = compute.field(join_column).isin(self_join_ids[join_column])
            if other_filter is not None:
                cur_filter = cur_filter & other_filter
            other_join_ids = other.to_table(columns=[join_column], filter=cur_filter)
            take_index = compute.index_in(
                self_join_ids[join_column], other_join_ids[join_column]
            )
            final_self_ids = compute.filter(
                numpy.arange(len(self_join_ids[join_column])),
                compute.is_valid(take_index),
            )
            final_other_ids = take_index.drop_null()

            if limit is not None:
                final_self_ids = final_self_ids[:limit]
                final_other_ids = final_other_ids[:limit]

            for self_ids, other_ids in zip(
                batched(final_self_ids, batch_size),
                batched(final_other_ids, batch_size),
            ):
                self_batch = dataset.take(
                    self_ids, columns=self_projection, filter=self_filter
                )
                other_batch = other.take(
                    other_ids, columns=other_projection, filter=cur_filter
                )
                yield pyarrow.table(
                    self_batch.columns + other_batch.columns,
                    names=self_projection + [f"other.{n}" for n in other_projection],
                )
                num_returned += len(self_ids)
                if limit is not None and num_returned >= limit:
                    break

                progress.update(len(self_ids))
            progress.update(progress.total - progress.n)

    def take_dataset(self, *args, **kwargs):
        dataset = object.__getattribute__(self, "dataset")
        stage = object.__getattribute__(self, "stage")
        parent = object.__getattribute__(self, "parent")
        return DatasetProxy(
            pyarrow_dataset.dataset(dataset.take(*args, **kwargs)), stage, parent
        )

    def filter(self, *args, **kwargs):
        dataset = object.__getattribute__(self, "dataset")
        stage = object.__getattribute__(self, "stage")
        parent = object.__getattribute__(self, "parent")
        return DatasetProxy(dataset.filter(*args, **kwargs), stage, parent)

    def __getattribute__(self, key):
        if key in (
            "__getitem__",
            "take_dataset",
            "filter",
            "batched_join",
            "dataset_join",
        ):
            return object.__getattribute__(self, key)
        return object.__getattribute__(self, "dataset").__getattribute__(key)

    def __getitem__(self, index):
        """
        Get a row from the dataset by index.

        Args:
            index: The index of the row(s) to retrieve.
                Can be an integer, a slice, a list/tuple, or any iterable of indices.

        Returns:
            The specified row(s) as a list of dicts.
        """
        dataset = object.__getattribute__(self, "dataset")
        stage = object.__getattribute__(self, "stage")
        parent = object.__getattribute__(self, "parent")

        if isinstance(index, slice):
            start, stop, step = index.indices(dataset.count_rows())
            indices = range(start, stop, step)
            result = dataset.take(indices).to_pylist()
        elif isinstance(index, (list, tuple)):
            result = dataset.take(index).to_pylist()
        elif hasattr(index, "__iter__"):
            result = dataset.take(list(index)).to_pylist()
        else:
            result = dataset.take([index]).to_pylist()

        if stage == DatasetStage.activation:
            # Join on match dataset
            match = parent.match[
                slice(result[0]["match_row_id"], result[-1]["match_row_id"] + 1)
            ]
            for i in range(len(result)):
                for k in match[0].keys():
                    result[i][k] = match[i][k]

        if isinstance(index, int):
            return result[0]
        return result
