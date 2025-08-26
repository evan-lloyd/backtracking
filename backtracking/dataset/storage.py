import os
from enum import StrEnum

import pyarrow.dataset as pyarrow_dataset


class StorageType(StrEnum):
    persistent = "persistent"
    """
    Persistent storage is for things you want to persist after the notebook is closed. For example,
    I set this to my RunPod network volume so I can keep my activations data between sessinos.
    """
    # pod local directory, but don't automatically delete
    temporary = "temporary"
    """
    Temporary storage is for things that should persist through the duration of a notebook session. It
    is expected to lose this when starting a new session, but it won't actively be deleted.
    """
    ephemeral = "ephemeral"
    """
    Ephemeral storage is for things you don't need to access later, and will be automatically
    deleted each time a new pipeline step is run.
    """


_cache_dirs = {
    StorageType.persistent: os.getenv("PERSISTENT_CACHE_DIR", "/workspace"),
    StorageType.temporary: os.getenv("TEMPORARY_CACHE_DIR", "/tmp/spar_data"),
    StorageType.ephemeral: os.getenv("EPHEMERAL_CACHE_DIR", "/tmp/spar_temp"),
}


def set_cache_dirs(dirs: dict[StorageType, str]):
    for k, v in dirs.items():
        _cache_dirs[k] = v


def _format_storage_path(storage_type: StorageType, storage_path: str):
    if storage_type not in _cache_dirs:
        raise ValueError(f"Unknown storage type: {storage_type}")
    return f"{_cache_dirs[storage_type]}/{storage_path}"


def load_dataset(storage_type: StorageType, path: str):
    return pyarrow_dataset.dataset(
        _format_storage_path(storage_type, path),
        format="parquet",
        ignore_prefixes=["log"],
    )
