# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from pyarrow.fs import FSSpecHandler, PyFileSystem
from pyarrow.fs import LocalFileSystem as ArrowLocalFileSystem

from nemo_curator.stages.base import CompositeStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import DocumentBatch, EmptyTask
from nemo_curator.utils.file_utils import FILETYPE_TO_DEFAULT_EXTENSIONS

from .base import BaseFileReader

# The read_kwargs a single pyarrow scan reproduces exactly; anything else keeps the per-file pandas reads.
_SINGLE_SCAN_KWARGS = frozenset({"engine", "dtype_backend", "columns", "storage_options"})


def _read_single_scan(  # noqa: PLR0911 - one return per case that falls back to per-file reads
    paths: list[str], columns: list[str] | None, storage_options: dict[str, Any] | None
) -> pd.DataFrame | None:
    """Read a file group in one pyarrow dataset scan. Returns None when one scan cannot reproduce per-file reads."""
    try:
        resolved = [url_to_fs(path, **(storage_options or {})) for path in paths]
    except (ImportError, ValueError):
        # No fsspec backend for the protocol (e.g. s3fs is not installed); pd.read_parquet can still read an
        # s3:// or gs:// URL through pyarrow's native filesystem.
        return None
    fs = resolved[0][0]
    if any(other != fs for other, _ in resolved):
        return None  # e.g. a local path next to an s3:// one, or zip:// paths in different archives
    paths = [path for _, path in resolved]
    # Exact type: a LocalFileSystem subclass may override how files are opened.
    pa_fs = ArrowLocalFileSystem() if type(fs) is LocalFileSystem else PyFileSystem(FSSpecHandler(fs))
    # Footers are read concurrently. The unified schema keeps a column that only later files have
    # (pyarrow would otherwise take the first file's schema and drop it), as pd.concat does.
    try:
        with ThreadPoolExecutor() as pool:
            schemas = list(pool.map(lambda path: pq.read_schema(path, filesystem=pa_fs), paths))
    except OSError:
        if any(fs.isdir(path) for path in paths):
            return None  # pd.read_parquet reads a directory path as a dataset
        raise
    # Per-file reads raise ArrowInvalid for a requested column that any file lacks.
    if columns is not None and any(name not in schema.names for schema in schemas for name in columns):
        return None
    # to_pandas applies the first file's pandas metadata to the whole table, so a physical index column
    # stored by only some files would come back as a data column (per-file reads drop every index), and
    # a columns name stored by only some files would name the result's columns (pd.concat drops it).
    if len({_pandas_layout(schema) for schema in schemas}) > 1:
        return None
    try:
        schema = pa.unify_schemas(schemas)
    except (pa.ArrowTypeError, pa.ArrowInvalid):
        return None  # conflicting column types: let pd.concat promote them, as before
    # Unification also widens types (null -> string, struct<x> -> struct<x, y>), which pd.concat would turn into
    # different dtypes (string vs large_string, object), so any column whose type changed keeps per-file reads.
    if any(s.field(name).type != schema.field(name).type for s in schemas for name in s.names):
        return None
    table = pq.read_table(paths, filesystem=pa_fs, schema=schema, columns=columns, use_pandas_metadata=True)
    # Same conversion pd.read_parquet does for dtype_backend="pyarrow"; the reset stands in for ignore_index=True.
    return table.to_pandas(types_mapper=pd.ArrowDtype).reset_index(drop=True)


def _pandas_layout(schema: pa.Schema) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    """Index columns a file stores as data (a RangeIndex is stored in the metadata only) and its columns names."""
    metadata = schema.pandas_metadata or {}
    index_columns = tuple(column for column in metadata.get("index_columns", []) if isinstance(column, str))
    column_names = tuple(level.get("name") for level in metadata.get("column_indexes", []))
    # A file without pandas metadata and one with unnamed columns convert the same way.
    return index_columns, column_names if any(name is not None for name in column_names) else ()


@dataclass
class ParquetReaderStage(BaseFileReader):
    """
    Stage that processes a group of Parquet files into a DocumentBatch.
    This stage accepts FileGroupTasks created by FilePartitioningStage
    and reads the actual file contents into DocumentBatches.

    Args:
        fields (list[str], optional): If specified, only read these columns. Defaults to None.
        read_kwargs (dict[str, Any], optional): Keyword arguments for the underlying reader. Defaults to {}.
    """

    name: str = "parquet_reader"

    def read_data(
        self,
        paths: list[str],
        read_kwargs: dict[str, Any] | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read Parquet files into one DataFrame. Raises an exception if reading fails.

        With the default pyarrow engine and dtype backend, a group on one filesystem is read in a single pyarrow
        scan; any other configuration reads file by file with Pandas and concatenates.
        """

        # Normalize read_kwargs to a dict to avoid TypeError when None
        # Work on a copy to avoid mutating caller's dict
        read_kwargs = {} if read_kwargs is None else dict(read_kwargs)

        update_kwargs = {}
        if fields is not None:
            update_kwargs["columns"] = fields
        if "engine" not in read_kwargs:
            update_kwargs["engine"] = "pyarrow"
        if "dtype_backend" not in read_kwargs:
            update_kwargs["dtype_backend"] = "pyarrow"
        read_kwargs.update(update_kwargs)
        if (
            paths
            and read_kwargs["engine"] == "pyarrow"
            and read_kwargs["dtype_backend"] == "pyarrow"
            and read_kwargs.keys() <= _SINGLE_SCAN_KWARGS
        ):
            df = _read_single_scan(paths, read_kwargs.get("columns"), read_kwargs.get("storage_options"))
            if df is not None:
                return df
        return pd.concat(
            (pd.read_parquet(path, **read_kwargs) for path in paths),
            ignore_index=True,
        )


@dataclass
class ParquetReader(CompositeStage[EmptyTask, DocumentBatch]):
    """Composite stage for reading Parquet files.

    This high-level stage decomposes into:
    1. FilePartitioningStage - partitions files into groups
    2. ParquetReaderStage - reads file groups into DocumentBatches
    """

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    fields: list[str] | None = None  # If specified, only read these columns
    read_kwargs: dict[str, Any] | None = None
    file_extensions: list[str] = field(default_factory=lambda: FILETYPE_TO_DEFAULT_EXTENSIONS["parquet"])
    task_type: Literal["document", "image", "video", "audio"] = "document"
    _generate_ids: bool = False
    _assign_ids: bool = False
    name: str = "parquet_reader"

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        if self.read_kwargs is not None:
            self.storage_options = self.read_kwargs.get("storage_options", {})

    def decompose(self) -> list[ParquetReaderStage]:
        """Decompose into file partitioning and processing stages."""
        if self.task_type != "document":
            msg = f"Converting DocumentBatch to {self.task_type} is not supported yet."
            raise NotImplementedError(msg)

        return [
            # First stage: partition files into groups
            FilePartitioningStage(
                file_paths=self.file_paths,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.read_kwargs.get("storage_options", {}) if self.read_kwargs is not None else None,
            ),
            # Second stage: process file groups into document batches
            ParquetReaderStage(
                fields=self.fields,
                read_kwargs=self.read_kwargs or {},
                _generate_ids=self._generate_ids,
                _assign_ids=self._assign_ids,
            ),
        ]

    def get_description(self) -> str:
        """Get a description of this composite stage."""

        parts = [f"Read Parquet files from {self.file_paths}"]

        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")

        if self.fields:
            parts.append(f"reading columns: {self.fields}")

        return ", ".join(parts)
