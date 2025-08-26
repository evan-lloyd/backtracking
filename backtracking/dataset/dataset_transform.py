import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO
from itertools import pairwise
from timeit import default_timer
from typing import (
    DefaultDict,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

import jsonlines
import numpy
import pyarrow
import pyarrow.compute as pyarrow_compute
import pyarrow.dataset as pyarrow_dataset
from pyarrow import Int64Array, RecordBatch, Schema
from pyarrow.dataset import TaggedRecordBatch
from tqdm.auto import tqdm

from .storage import StorageType, _format_storage_path


@dataclass
class ProcessContext:
    scanner: pyarrow_dataset.Scanner
    log_file_path: str
    progress: tqdm
    batch_start_index: List[List[int]]
    total_count: int
    batch_size: int
    buffer: StringIO = field(default_factory=StringIO, init=False)
    flush_count: int = field(default=0, init=False)
    rows_written: int = field(default=0, init=False)

    def pyarrow_file_visitor(self, written_file):
        row_group_meta = [
            written_file.metadata.row_group(i)
            for i in range(written_file.metadata.num_row_groups)
        ]
        self.log_line(
            "Wrote data file",
            {
                "path": written_file.path,
                "metadata": {
                    "num_columns": written_file.metadata.num_columns,
                    "num_row_groups": written_file.metadata.num_row_groups,
                    "num_rows": written_file.metadata.num_rows,
                    "row_groups": [
                        {
                            "num_rows": rg.num_rows,
                            "total_byte_size": rg.total_byte_size,
                        }
                        for rg in row_group_meta
                    ],
                },
            },
        )
        self.flush_log()

    def log_line(self, message: str, meta: dict):
        writer = jsonlines.Writer(self.buffer)
        writer.write(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "message": message,
                "meta": meta,
            }
        )

    def flush_log(self):
        if self.buffer is not None:
            with open(
                self.log_file_path, "w" if self.flush_count == 0 else "a"
            ) as log_file:
                self.buffer.seek(0)
                log_file.write(self.buffer.read())
                self.flush_count += 1
        self.buffer = StringIO()

    def init_processing_log(self):
        self.log_line(
            "Start processing",
            {
                "output_path": self.log_file_path,
                "total_count": self.total_count,
            },
        )

    def finalize_processing_log(self):
        self.log_line(
            "Finished processing",
            {"rows_written": self.rows_written},
        )
        self.flush_log()

    def log_batch(
        self,
        in_batch: TaggedRecordBatch,
        out_batch: RecordBatch,
        timing: Dict[str, List[float]],
        batch_index: int,
    ):
        self.log_line(
            "Finished batch",
            {
                "batch_index": batch_index,
                "num_in_rows": len(in_batch.record_batch),
                "num_out_rows": len(out_batch),
                "timing": timing,
            },
        )
        pass


class DatasetTransform:
    class Default:
        pass

    DEFAULT_PROJECTION = Default()

    schema: Optional[Schema] = None
    noop_transform: bool = False
    use_threads: bool = True
    min_rows_per_group: int = 1_000
    max_rows_per_group: int = 10_000
    max_rows_per_file: int = 10_000
    batch_size: int = 1_000
    projection: Set[str] | Default = DEFAULT_PROJECTION
    existing_data_behavior: str = "delete_matching"
    marks: DefaultDict[str, List[float]]

    def mark(self, key: Optional[str] = None):
        """Add a labeled timestamp to this run's metadata so we can time each sub-operation."""
        if key is None:
            key = next(reversed(self.marks.keys()))

        self.marks[key].append(default_timer())

    def timing(self, num_rows: int):
        return {
            **(
                {
                    "per_row": [
                        (self.marks["all_rows"][1] - self.marks["all_rows"][0])
                        / (num_rows)
                    ]
                }
                if num_rows > 0
                else {}
            ),
            **{
                key: [m2 - m1 for m1, m2 in pairwise(m)]
                for key, m in self.marks.items()
                if len(m) >= 2
            },
        }

    def __init_subclass__(cls) -> None:
        if "storage_type" not in cls.__dict__:
            raise NotImplementedError(
                "Dataset transforms must define a default storage_type"
            )

    def __init__(
        self,
        min_rows_per_group: int | Default = Default(),
        max_rows_per_group: int | Default = Default(),
        max_rows_per_file: int | Default = Default(),
        batch_size: int | Default = Default(),
        storage_type: StorageType | Default = Default(),
        use_threads: bool | Default = Default(),
        existing_data_behavior: str | Default = Default(),
    ):
        if not isinstance(min_rows_per_group, DatasetTransform.Default):
            self.min_rows_per_group = min_rows_per_group
        if not isinstance(max_rows_per_group, DatasetTransform.Default):
            self.max_rows_per_group = max_rows_per_group
        if not isinstance(max_rows_per_file, DatasetTransform.Default):
            self.max_rows_per_file = max_rows_per_file
        if not isinstance(batch_size, DatasetTransform.Default):
            self.batch_size = batch_size
        if not isinstance(storage_type, DatasetTransform.Default):
            self.storage_type = storage_type
        if not isinstance(use_threads, DatasetTransform.Default):
            self.use_threads = use_threads
        if not isinstance(existing_data_behavior, DatasetTransform.Default):
            self.existing_data_behavior = existing_data_behavior

        self.marks = defaultdict(list)

    def run(
        self,
        dataset: pyarrow_dataset.Dataset,
        path: str,
        progress_desc: Optional[str] = None,
        run_id: int = 0,
        resume_from_batch: int = 0,
    ):
        scanner = self.get_scanner(dataset)
        total_count, batch_start_index = self.get_fragment_stats(dataset)
        progress = tqdm(
            desc=progress_desc or f"Writing dataset to {self.storage_type} cache",
            total=total_count,
        )
        output_path = _format_storage_path(self.storage_type, path)
        log_path = f"{output_path}/log-{run_id}.ndjson"
        if resume_from_batch > 0:
            existing_data_behavior = "overwrite_or_ignore"
        else:
            existing_data_behavior = self.existing_data_behavior
        context = ProcessContext(
            scanner,
            log_path,
            progress,
            batch_start_index,
            total_count,
            self.batch_size,
        )
        context.init_processing_log()
        pyarrow_dataset.write_dataset(
            self.transform_batch(context, resume_from_batch),
            output_path,
            schema=self.schema if self.schema is not None else dataset.schema,
            format="parquet",
            max_rows_per_file=self.max_rows_per_file,
            min_rows_per_group=self.min_rows_per_group,
            max_rows_per_group=self.max_rows_per_group,
            existing_data_behavior=existing_data_behavior,
            file_visitor=context.pyarrow_file_visitor,
            basename_template=f"run-{run_id}-part-{{i}}.parquet",
        )
        context.finalize_processing_log()

    def get_fragment_stats(
        self, dataset: pyarrow_dataset.Dataset
    ) -> Tuple[int, List[List[int]]]:
        batch_start_index = []
        total_count = 0

        for fragment in dataset.get_fragments():
            batch_start_within_fragment = []
            # This is a bit silly, but easier than reverse-engineering pyarrow's batching behavior, which makes
            # no effort to consolidate across row groups. By projecting to no columns, this should be
            # reasonably efficient.
            for batch in fragment.to_batches(
                columns=[], batch_size=self.batch_size, use_threads=False
            ):
                batch_start_within_fragment.append(pyarrow.scalar(total_count))
                total_count += len(batch)
            batch_start_index.append(batch_start_within_fragment)

        return (total_count, batch_start_index)

    def get_scanner(self, dataset: pyarrow_dataset.Dataset) -> pyarrow_dataset.Scanner:
        if not isinstance(self.projection, DatasetTransform.Default):
            projection = self.projection
        else:
            projection = {*dataset.schema.names}

        projection.update(("__batch_index", "__fragment_index"))

        return dataset.scanner(
            batch_size=self.batch_size,
            columns=list(projection),
            use_threads=self.use_threads,
            # None of these seem to help with OOMs on tokenization transform, my top guesses are:
            # 1) There's just a really big data point late at row ~90k?
            # 2) The problem is actually with the *writing* of the dataset
            # fragment_scan_options=pyarrow_dataset.ParquetFragmentScanOptions(
            #     use_buffered_stream=True,
            #     pre_buffer=False,
            #     # x10 over default, because we were getting errors
            #     # thrift_string_size_limit=10*100*1000*1000,
            #     # thrift_container_size_limit=10*1000*1000
            # ),
        )

    def transform_batch(
        self,
        context: ProcessContext,
        resume_from_batch: int,
    ) -> Iterator[RecordBatch]:
        rows_written = 0
        num_skipped = 0
        with context.progress:
            # TODO: add parallelism, if self marked as allowed
            for batch_index, batch in enumerate(context.scanner.scan_batches()):
                # No-op until we get to the starting batch index
                if batch_index < resume_from_batch:
                    num_skipped += len(batch.record_batch)
                    yield RecordBatch.from_pylist([], self.schema)
                    continue
                elif batch_index == resume_from_batch:
                    context.progress.update(num_skipped)
                self.marks = defaultdict(list)
                in_batch = batch.record_batch

                batch_id = in_batch["__batch_index"][0].as_py()
                fragment_id = in_batch["__fragment_index"][0].as_py()

                out_batch = in_batch.drop_columns(("__batch_index", "__fragment_index"))

                self.mark("all_rows")
                if not self.noop_transform:
                    row_indices = pyarrow_compute.add(
                        numpy.arange(len(in_batch), dtype=numpy.int64),
                        pyarrow.scalar(
                            context.batch_start_index[fragment_id][batch_id],
                            pyarrow.int64(),
                        ),
                    )

                    out_batch = self.transform(out_batch, row_indices)
                self.mark("all_rows")

                yield out_batch
                rows_written += len(out_batch)

                context.log_batch(
                    batch, out_batch, self.timing(len(in_batch)), batch_index
                )

                context.progress.update(len(in_batch))
        context.rows_written = rows_written

    def transform(self, in_batch: RecordBatch, row_indices: Int64Array) -> RecordBatch:
        raise NotImplementedError("Dataset transforms must implement transform method")


class NoopTransform(DatasetTransform):
    noop_transform = True
    storage_type = StorageType.temporary

    def __init__(self, *args, schema: Optional[Schema] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if schema is not None:
            self.schema = schema


def add_row_id(
    in_transform: DatasetTransform, row_id_name: str, _batch_size: int = 1_000
):
    class AddRowId(DatasetTransform):
        batch_size = _batch_size

        schema = in_transform.schema.append(pyarrow.field(row_id_name, pyarrow.int64()))
        storage_type = in_transform.storage_type
        max_rows_per_file = in_transform.max_rows_per_file
        max_rows_per_group = in_transform.max_rows_per_group
        min_rows_per_group = in_transform.min_rows_per_group
        existing_data_behavior = in_transform.existing_data_behavior

        def transform(
            self, in_batch: RecordBatch, row_indices: Int64Array
        ) -> RecordBatch:
            return in_batch.append_column(row_id_name, row_indices)

    return AddRowId()
