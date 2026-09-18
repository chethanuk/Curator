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

import threading
import time
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow as pa
import pytest
from fsspec.implementations.memory import MemoryFileSystem

from nemo_curator.stages.text.io.reader.parquet import ParquetReader, ParquetReaderStage
from nemo_curator.tasks import EmptyTask, FileGroupTask
from nemo_curator.tasks.document import DocumentBatch


@pytest.fixture
def sample_parquet_files(tmp_path: Path) -> list[str]:
    """Create multiple Parquet files for testing."""
    files = []
    for i in range(3):
        file_path = tmp_path / f"test_{i}.parquet"
        # Create records with different ranges to ensure variety
        records = _sample_records(start=i * 2, n=2)
        _write_parquet_file(file_path, records)
        files.append(str(file_path))
    return files


@pytest.fixture
def parquet_file_group_tasks(sample_parquet_files: list[str]) -> list[FileGroupTask]:
    """Create multiple FileGroupTasks for parquet files."""
    return [
        FileGroupTask(dataset_name="test_dataset", data=[file_path], _metadata={})
        for i, file_path in enumerate(sample_parquet_files)
    ]


def _write_parquet_file(file_path: Path, records: list[dict]) -> None:
    # Use pandas to write a Parquet file
    df = pd.DataFrame(records)
    df.to_parquet(file_path, index=False)


def _sample_records(start: int = 0, n: int = 2) -> list[dict]:
    return [
        {
            "text": f"doc_{start + i}",
            "category": f"cat_{(start + i) % 3}",
            "score": float(start + i),
        }
        for i in range(n)
    ]


def _make_file_group_task(files: list[str]) -> FileGroupTask:
    return FileGroupTask(
        dataset_name="ds",
        data=files,
        reader_config={},
        _metadata={"source_files": files},
    )


class _SlowMemoryFileSystem(MemoryFileSystem):
    """In-memory filesystem with a fixed delay per read, recording how many files have a read in flight."""

    protocol = ("slowmem",)
    _lock = threading.Lock()
    _in_flight: dict[str, int]  # per-class counters, reset by the test
    peak_files_in_flight = 0

    def _open(self, path: str, mode: str = "rb", **kwargs: object) -> object:
        f = super()._open(path, mode=mode, **kwargs)
        if "r" not in mode:
            return f
        cls, read = type(self), f.read

        def slow_read(*args: object, **kw: object) -> bytes:
            with cls._lock:
                cls._in_flight[path] = cls._in_flight.get(path, 0) + 1
                cls.peak_files_in_flight = max(cls.peak_files_in_flight, len(cls._in_flight))
            try:
                time.sleep(0.02)
                return read(*args, **kw)
            finally:
                with cls._lock:
                    cls._in_flight[path] -= 1
                    if not cls._in_flight[path]:
                        del cls._in_flight[path]

        f.read = slow_read
        return f


def test_parquet_reader_stage_fetches_group_files_concurrently():
    fsspec.register_implementation("slowmem", _SlowMemoryFileSystem, clobber=True)
    files = []
    for i in range(8):
        path = f"slowmem://concurrent/part_{i}.parquet"
        pd.DataFrame(_sample_records(start=i * 2, n=2)).to_parquet(path, index=False)
        files.append(path)
    _SlowMemoryFileSystem._in_flight, _SlowMemoryFileSystem.peak_files_in_flight = {}, 0

    out = ParquetReaderStage(read_kwargs={"storage_options": {}}).process(_make_file_group_task(files))

    assert out.to_pandas()["text"].tolist() == [f"doc_{i}" for i in range(16)]
    # Object-store latency is paid per request; reading one file at a time serialises it across the group.
    assert _SlowMemoryFileSystem.peak_files_in_flight > 1


@pytest.mark.parametrize("read_kwargs", [{}, {"engine": "pyarrow"}], ids=["defaults", "explicit_pyarrow_engine"])
def test_parquet_reader_stage_reads_and_concatenates(sample_parquet_files: list[str], read_kwargs: dict):
    task = _make_file_group_task(sample_parquet_files[:2])

    out = ParquetReaderStage(read_kwargs=read_kwargs, fields=None).process(task)

    assert isinstance(out, DocumentBatch)
    assert out._metadata == {"source_files": sample_parquet_files[:2]}
    df = out.to_pandas()
    assert df["text"].tolist() == ["doc_0", "doc_1", "doc_2", "doc_3"]  # file order, then row order
    assert list(df.columns) == ["text", "category", "score"]
    assert all(isinstance(dtype, pd.ArrowDtype) for dtype in df.dtypes)
    assert isinstance(df.index, pd.RangeIndex)


class TestParquetReaderStorageOptionsAndColumns:
    def test_columns_selection(self, tmp_path: Path):
        f = tmp_path / "a.parquet"
        _write_parquet_file(f, _sample_records(0, 3))
        task = _make_file_group_task([str(f)])
        stage = ParquetReaderStage(fields=["text"])  # select one column
        out = stage.process(task)
        df = out.to_pandas()
        assert list(df.columns) == ["text"]
        assert len(df) == 3

    @pytest.mark.parametrize(
        ("prefix", "storage_options"),
        [
            ("memory://parquet-reader/remote/", {}),
            # dir:// only resolves when storage_options reach the filesystem: they name its root and target.
            ("dir://", {"path": "/parquet-reader/remote", "target_protocol": "memory"}),
        ],
        ids=["memory_url", "storage_options_required"],
    )
    def test_reads_fsspec_urls_with_storage_options(self, prefix: str, storage_options: dict):
        for i in range(2):
            pd.DataFrame(_sample_records(i * 2, 2)).to_parquet(
                f"memory://parquet-reader/remote/{i}.parquet", index=False
            )
        files = [f"{prefix}{i}.parquet" for i in range(2)]
        stage = ParquetReaderStage(read_kwargs={"storage_options": storage_options})

        df = stage.process(_make_file_group_task(files)).to_pandas()

        assert df["text"].tolist() == ["doc_0", "doc_1", "doc_2", "doc_3"]


def test_parquet_reader_stage_pandas_errors_when_some_columns_missing(tmp_path: Path):
    # Prepare files with known columns
    f = tmp_path / "a.parquet"
    _write_parquet_file(f, _sample_records(0, 3))

    task = _make_file_group_task([str(f)])
    stage = ParquetReaderStage(fields=["text", "does_not_exist"])

    with pytest.raises(pa.lib.ArrowInvalid):
        _ = stage.process(task)


def test_parquet_reader_stage_pandas_raises_when_all_columns_missing(tmp_path: Path):
    f = tmp_path / "a.parquet"
    _write_parquet_file(f, _sample_records(0, 2))

    task = _make_file_group_task([str(f)])
    stage = ParquetReaderStage(fields=["missing_only"])

    with pytest.raises(pa.lib.ArrowInvalid):
        _ = stage.process(task)


def test_parquet_reader_stage_empty_file_uses_base_reader_policy(tmp_path: Path):
    f = tmp_path / "empty.parquet"
    pd.DataFrame({"text": pd.Series(dtype="string"), "score": pd.Series(dtype="float64")}).to_parquet(
        f,
        index=False,
    )
    task = _make_file_group_task([str(f)])

    with pytest.raises(ValueError, match="No data read from files"):
        ParquetReaderStage().process(task)

    out = ParquetReaderStage(allow_empty=True).process(task)
    assert isinstance(out, DocumentBatch)
    assert out.num_items == 0
    assert out.to_pandas().columns.tolist() == ["text", "score"]


def _per_file_reference(paths: list[str], **read_kwargs: object) -> pd.DataFrame:
    """What the stage returned before group reads: one pd.read_parquet per file, then concat."""
    kwargs = {"engine": "pyarrow", "dtype_backend": "pyarrow", **read_kwargs}
    return pd.concat((pd.read_parquet(path, **kwargs) for path in paths), ignore_index=True)


_EQUIVALENCE_CASES = {
    # case id: (one DataFrame per file, stage fields, extra read_kwargs)
    "column_only_in_second_file": ([pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2], "b": ["x"]})], None, {}),
    "column_only_in_first_file": ([pd.DataFrame({"a": [1], "b": ["x"]}), pd.DataFrame({"a": [2]})], None, {}),
    "non_range_index": ([pd.DataFrame({"a": [1, 2]}, index=[10, 20]), pd.DataFrame({"a": [3]}, index=[5])], None, {}),
    "int_vs_string_column": ([pd.DataFrame({"a": [1]}), pd.DataFrame({"a": ["x"]})], None, {}),
    "empty_and_nonempty_file": (
        [pd.DataFrame({"a": pd.Series([], dtype="int64")}), pd.DataFrame({"a": [1]})],
        None,
        {},
    ),
    "fields_projection": ([pd.DataFrame(_sample_records(i * 2, 2)) for i in range(3)], ["text"], {}),
    "numpy_nullable_backend": (
        [pd.DataFrame(_sample_records(i * 2, 2)) for i in range(2)],
        None,
        {"dtype_backend": "numpy_nullable"},
    ),
    "filters": ([pd.DataFrame(_sample_records(i * 2, 2)) for i in range(2)], None, {"filters": [("score", ">", 0.5)]}),
}


@pytest.mark.parametrize(
    ("frames", "fields", "read_kwargs"), _EQUIVALENCE_CASES.values(), ids=_EQUIVALENCE_CASES.keys()
)
def test_parquet_reader_stage_matches_per_file_reads(
    tmp_path: Path, frames: list[pd.DataFrame], fields: list[str] | None, read_kwargs: dict
):
    paths = []
    for i, frame in enumerate(frames):
        path = tmp_path / f"{i}.parquet"
        frame.to_parquet(path)
        paths.append(str(path))
    expected = _per_file_reference(paths, **read_kwargs, **({"columns": fields} if fields else {}))

    out = ParquetReaderStage(fields=fields, read_kwargs=read_kwargs, allow_empty=True).process(
        _make_file_group_task(paths)
    )

    pd.testing.assert_frame_equal(out.to_pandas(), expected)


def test_parquet_reader_stage_reads_group_spanning_filesystems(tmp_path: Path):
    local = tmp_path / "local.parquet"
    _write_parquet_file(local, _sample_records(0, 2))
    remote = "memory://parquet-reader/mixed/remote.parquet"
    pd.DataFrame(_sample_records(2, 2)).to_parquet(remote, index=False)
    paths = [str(local), remote]

    out = ParquetReaderStage().process(_make_file_group_task(paths))

    pd.testing.assert_frame_equal(out.to_pandas(), _per_file_reference(paths))


def test_parquet_reader_stage_fields_fill_nulls_for_files_missing_the_column(tmp_path: Path):
    # Per-file reads raised on the first file lacking a requested column; a group read returns nulls for it,
    # as it already does for every column when fields is None.
    paths = [str(tmp_path / "a.parquet"), str(tmp_path / "ab.parquet")]
    pd.DataFrame({"a": [1]}).to_parquet(paths[0], index=False)
    pd.DataFrame({"a": [2], "b": ["x"]}).to_parquet(paths[1], index=False)

    df = ParquetReaderStage(fields=["b"]).process(_make_file_group_task(paths)).to_pandas()

    assert df["b"].isna().tolist() == [True, False]
    assert df["b"].iloc[1] == "x"


def test_parquet_reader_stage_pyarrow_errors_when_some_columns_missing(tmp_path: Path):
    f = tmp_path / "a.parquet"
    _write_parquet_file(f, _sample_records(0, 4))

    task = _make_file_group_task([str(f)])
    stage = ParquetReaderStage(read_kwargs={"engine": "pyarrow"}, fields=["category", "not_there"])

    with pytest.raises(pa.lib.ArrowInvalid):
        _ = stage.process(task)


def test_base_reader_outputs_reflect_columns():
    stage = ParquetReaderStage(fields=["a", "b"])
    inputs, outputs = stage.inputs(), stage.outputs()
    assert inputs == ([], [])
    assert outputs == (["data"], ["a", "b"])


def test_parquet_reader_decompose_configuration(tmp_path: Path):
    reader = ParquetReader(
        file_paths=str(tmp_path),
        files_per_partition=3,
        fields=["text", "score"],
        read_kwargs={"engine": "pyarrow", "storage_options": {"anon": True}},
    )
    stages = reader.decompose()
    assert len(stages) == 2

    # First stage: FilePartitioningStage with parquet extension filter
    first = stages[0]
    from nemo_curator.stages.file_partitioning import FilePartitioningStage

    assert isinstance(first, FilePartitioningStage)
    assert first.file_extensions == [".parquet"]
    assert first.file_paths == str(tmp_path)
    assert first.files_per_partition == 3
    assert first.storage_options == {"anon": True}

    # Second stage: ParquetReaderStage config propagated (includes storage_options)
    second = stages[1]
    assert isinstance(second, ParquetReaderStage)
    assert second.fields == ["text", "score"]
    assert second.read_kwargs == {"engine": "pyarrow", "storage_options": {"anon": True}}


def test_parquet_reader_get_description():
    reader1 = ParquetReader(file_paths="s3://bucket/path", files_per_partition=5, fields=["text"])
    desc1 = reader1.get_description()
    assert "Read Parquet files from s3://bucket/path" in desc1
    assert "with 5 files per partition" in desc1
    assert "reading columns: ['text']" in desc1

    reader2 = ParquetReader(file_paths="/data", blocksize="128MB")
    desc2 = reader2.get_description()
    assert "Read Parquet files from /data" in desc2
    assert "with target blocksize 128MB" in desc2


def test_parquet_reader_non_document_task_type_not_supported():
    reader = ParquetReader(file_paths="/data", task_type="image")  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="Converting DocumentBatch to image is not supported yet"):
        _ = reader.decompose()


def test_parquet_reader_with_file_group_tasks_fixture(parquet_file_group_tasks: list[FileGroupTask]):
    """Demonstrate usage of parquet_file_group_tasks fixture."""
    stage = ParquetReaderStage(read_kwargs={}, fields=None)

    all_results = []
    for task in parquet_file_group_tasks:
        result = stage.process(task)
        df = result.to_pandas()
        assert len(df) == 2  # Each task has 1 file with 2 records
        assert {"text", "category", "score"}.issubset(set(df.columns))
        all_results.append(df)

    # Verify we processed all 3 tasks (files)
    assert len(all_results) == 3

    # Verify each file has unique records based on the start offset
    for i, df in enumerate(all_results):
        expected_texts = [f"doc_{i * 2}", f"doc_{i * 2 + 1}"]
        actual_texts = df["text"].tolist()
        assert actual_texts == expected_texts


def test_parquet_reader_with_blocksize_limit(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    # Storage size is larger than 10_000 bytes
    # In-memory size is larger than 1 billion bytes
    size = 1000
    df = pd.DataFrame({"id": list(range(size)), "text": ["a" * 4000] * size, "other_field": ["b" * 1_000_000] * size})
    df.to_parquet(tmp_path / "test.parquet")

    stage = ParquetReader(file_paths=str(tmp_path), blocksize=10_000)
    assert len(stage.decompose()) == 2

    # Since the storage size is larger than 10_000 bytes, the FilePartitioningStage should warn
    file_partitioning_stage = stage.decompose()[0]
    with caplog.at_level("WARNING"):
        file_partitioning_stage.process(EmptyTask)
    assert "File group task has exceeded the storage limit per partition" in caplog.text
