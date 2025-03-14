from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, NamedTuple, Optional

import pyarrow
import pyarrow.compute as compute
import torch
from datasets import load_dataset as load_huggingface_dataset
from pyarrow.dataset import Dataset, dataset
from transformers import PreTrainedTokenizer

from .dataset_proxy import DatasetProxy
from .dataset_stage import DatasetStage
from .dataset_transform import DatasetTransform, NoopTransform, add_row_id
from .numpy_array import NumpyArrayType
from .storage import StorageType, load_dataset


class ProcessInfo(NamedTuple):
    base_dataset: Dataset
    transform: DatasetTransform


class ActivationDataset(ABC):
    datasets: Dict[DatasetStage, Dataset]
    dataset_name: str
    # Pin the revision for reproducibility
    dataset_revision: str
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizer

    LAST_RUN_STORAGE_PATH: str = "last_pipeline_run"

    def _maybe_head_dataset(
        self, arrow_dataset: Dataset, process_head: Optional[int]
    ) -> Dataset:
        if process_head is None:
            return arrow_dataset
        return dataset(arrow_dataset.head(process_head))

    def _process_raw(self) -> ProcessInfo:
        hf_dataset = load_huggingface_dataset(
            self.dataset_name, revision=self.dataset_revision, split="train"
        )
        # Turn the internal HF datasets table (which is already an Arrow table) into a dataset.
        # Note that we have to cast list_(strings) to large string to actually write row groups of a reasonable size,
        # otherwise we get "offset overflow while concatenating arrays".

        arrow_table: pyarrow.Table = hf_dataset.with_format("arrow")[:]
        schema = arrow_table.schema
        for i, field in enumerate(schema):
            if field.type == pyarrow.list_(pyarrow.string()):
                arrow_table = arrow_table.set_column(
                    i,
                    pyarrow.field(field.name, pyarrow.list_(pyarrow.large_string())),
                    compute.cast(
                        arrow_table[field.name], pyarrow.list_(pyarrow.large_string())
                    ),
                )

        return ProcessInfo(
            dataset(arrow_table),
            NoopTransform(
                schema=arrow_table.schema,
                storage_type=StorageType.persistent,
                batch_size=10_000,
                min_rows_per_group=10_000,
                max_rows_per_group=10_000,
                max_rows_per_file=10_000,
            ),
        )

    def _stages(
        self,
    ) -> Dict[DatasetStage, Callable[[], ProcessInfo]]:
        return {
            getattr(DatasetStage, stage): getattr(self, f"_process_{stage}")
            for stage in (
                "raw",
                "token",
                "match",
                # Not yet ready for prime-time
                # "activation",
            )  # "classification" )
        }

    @abstractmethod
    def _process_match(self) -> ProcessInfo:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def _process_token(self) -> ProcessInfo:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def _process_activation(self) -> ProcessInfo:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def _process_classification(self, activation: Dataset):
        return Dataset.from_dict({})

    def __getattr__(self, key: str):
        if key in self.datasets:
            return DatasetProxy(self.datasets[key], key, self)
        raise AttributeError(f"No dataset stage named '{key}'")

    def process_stage(
        self, stage: DatasetStage, process_head: Optional[int] = None
    ) -> Dataset:
        self.logger.info(f"Running pipeline stage: {stage}")
        base_dataset, transform = self._stages()[stage]()

        add_id_transform = add_row_id(transform, f"{stage}_row_id")

        # Really need to refactor this somehow...
        # Run the transform first to ephemeral storage, since it won't have its ID column added yet.
        # We will add the ID and move it over to where it actually wanted to be stored.
        original_storage_type = transform.storage_type
        transform.storage_type = StorageType.ephemeral

        transform.run(
            self._maybe_head_dataset(base_dataset, process_head),
            self.LAST_RUN_STORAGE_PATH,
            f"Running pipeline stage {stage}",
        )
        pre_id_dataset = load_dataset(StorageType.ephemeral, self.LAST_RUN_STORAGE_PATH)
        add_id_transform.storage_type = original_storage_type
        # TODO: remove the need to run this step
        add_id_transform.run(
            pre_id_dataset,
            self._path_for_stage(stage),
            f"Adding row IDs to {stage} data",
        )
        self.datasets[stage] = self.load_stage(
            stage,
            reprocess_on_load_failure=False,
            process_head=process_head,
            storage_type=original_storage_type,
        )

        return self.datasets[stage]

    def _path_for_stage(self, stage: DatasetStage) -> str:
        return str(f"{self.dataset_name.replace('/', '__')}_{stage}/")

    def load_stage(
        self,
        stage: DatasetStage,
        reprocess_on_load_failure=True,
        process_head: Optional[int] = None,
        storage_type: StorageType = StorageType.persistent,
    ) -> Dataset:
        try:
            # TODO: it's awkward that we won't know where a dataset is supposed to live unless we
            # just transformed it, this should definitely be refactored.
            result = self._load_stage(stage, storage_type)
        except Exception:
            if reprocess_on_load_failure:
                result = self.process_stage(stage, process_head=process_head)
            else:
                raise

        return result

    def _load_stage(
        self,
        stage: DatasetStage,
        storage_type: StorageType = StorageType.persistent,
    ) -> Dataset:
        if stage not in self.datasets:
            self.datasets[stage] = load_dataset(
                storage_type, self._path_for_stage(stage)
            )
        return self.datasets[stage]

    def run_pipeline(
        self,
        proceed_after_exception=False,
        reprocess_on_load_failure=True,
        reprocess_stages: Optional[List[DatasetStage]] = None,
        process_head: Optional[int] = None,
    ):
        for stage in self._stages().keys():
            try:
                if stage in (reprocess_stages or []):
                    self.process_stage(stage, process_head)
                else:
                    self.load_stage(
                        stage,
                        reprocess_on_load_failure=reprocess_on_load_failure,
                        process_head=process_head,
                    )
            except Exception:
                import traceback

                if proceed_after_exception:
                    self.logger.warn(
                        "Exception while loading, proceeding with partially initialized dataset..."
                    )
                    self.logger.warn(traceback.format_exc())
                else:
                    raise

    def __init__(
        self,
        dataset_name: str,
        dataset_revision: str,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
    ):
        assert dataset_revision not in (None, "main"), (
            "For reproducibility, you must specify a dataset revision in the form of a git commit SHA or tag"
        )
        self.datasets = {}
        self.dataset_name = dataset_name
        self.dataset_revision = dataset_revision
        self.split = split
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(f"ActivationDataset.{self.dataset_name}")
        # We need to do this before attempting to load any datasets
        NumpyArrayType().ensure_registered()
