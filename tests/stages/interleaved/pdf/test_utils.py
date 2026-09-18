# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for the general PDF utility functions."""

from __future__ import annotations

import base64
import io
import json
import zipfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from PIL import Image

from nemo_curator.stages.interleaved.pdf.utils import (
    build_canvas,
    crop_to_bbox,
    extract_pdf_from_jsonl,
    extract_pdf_from_zip,
    extract_pdfs_from_jsonl_batch,
    image_to_bytes,
    resolve_cc_pdf_zip_path,
)


class TestBuildCanvas:
    def test_canvas_has_correct_size(self):
        img = Image.new("RGB", (100, 200), color="red")
        canvas = build_canvas(img, (2048, 1664))
        assert canvas.size == (1664, 2048)

    def test_small_image_gets_padded(self):
        img = Image.new("RGB", (50, 50), color="blue")
        canvas = build_canvas(img, (100, 100))
        assert canvas.size == (100, 100)


class TestCropToBbox:
    def test_none_bbox_returns_full_canvas(self):
        canvas = Image.new("RGB", (100, 100))
        result = crop_to_bbox(canvas, None, (100, 100))
        assert result is not None
        assert result.size == (100, 100)

    def test_valid_bbox_crops(self):
        canvas = Image.new("RGB", (100, 100), color="green")
        result = crop_to_bbox(canvas, [0.0, 0.0, 0.5, 0.5], (100, 100))
        assert result is not None
        assert result.size == (50, 50)

    def test_tiny_bbox_returns_none(self):
        canvas = Image.new("RGB", (100, 100))
        result = crop_to_bbox(canvas, [0.0, 0.0, 0.01, 0.01], (100, 100), min_crop_px=10)
        assert result is None


class TestImageToBytes:
    def test_roundtrip(self):
        img = Image.new("RGB", (10, 10), color="red")
        data = image_to_bytes(img)
        assert len(data) > 0
        restored = Image.open(io.BytesIO(data))
        assert restored.size == (10, 10)


class TestCCPDFZipHelpers:
    def test_resolve_cc_pdf_zip_path(self):
        zip_path, member = resolve_cc_pdf_zip_path("0001234.pdf", "/data/zips")
        assert member == "0001234.pdf"
        assert "0000-0999" in zip_path
        assert "0001.zip" in zip_path

    def test_resolve_higher_numbers(self):
        zip_path, member = resolve_cc_pdf_zip_path("5432100.pdf", "/data/zips")
        assert member == "5432100.pdf"
        assert "5000-5999" in zip_path
        assert "5432.zip" in zip_path

    def test_extract_pdf_from_zip(self, tmp_path: Path):
        zip_dir = tmp_path / "0000-0999"
        zip_dir.mkdir(parents=True)
        zip_path = zip_dir / "0001.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("0001234.pdf", b"%PDF-test-content")

        result = extract_pdf_from_zip("0001234.pdf", str(tmp_path))
        assert result == b"%PDF-test-content"

    def test_extract_missing_returns_none(self, tmp_path: Path):
        result = extract_pdf_from_zip("9999999.pdf", str(tmp_path))
        assert result is None


class TestExtractPdfFromJsonl:
    """Tests for extract_pdf_from_jsonl: byte_offset, line_idx, and neither paths."""

    def _make_jsonl_file(self, tmp_path: Path, pdf_bytes_list: list[bytes]) -> Path:
        lines = []
        for pdf_bytes in pdf_bytes_list:
            content = base64.b64encode(pdf_bytes).decode()
            lines.append(json.dumps({"content": content}) + "\n")
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_bytes("".join(lines).encode())
        return jsonl_file

    def test_byte_offset_path(self, tmp_path: Path):
        pdf_bytes = b"%PDF-fake-content"
        jsonl_file = self._make_jsonl_file(tmp_path, [pdf_bytes])
        result = extract_pdf_from_jsonl(str(jsonl_file), byte_offset=0)
        assert result == pdf_bytes

    def test_line_idx_path(self, tmp_path: Path):
        pdf_bytes_0 = b"%PDF-line0"
        pdf_bytes_1 = b"%PDF-line1"
        jsonl_file = self._make_jsonl_file(tmp_path, [pdf_bytes_0, pdf_bytes_1])
        result = extract_pdf_from_jsonl(str(jsonl_file), line_idx=1)
        assert result == pdf_bytes_1

    def test_neither_returns_none(self, tmp_path: Path):
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_bytes(b"")
        result = extract_pdf_from_jsonl(str(jsonl_file))
        assert result is None

    def test_bad_offset_returns_none(self, tmp_path: Path):
        jsonl_file = self._make_jsonl_file(tmp_path, [b"%PDF-x"])
        # Seeking beyond end of file returns empty line, JSON decode fails
        result = extract_pdf_from_jsonl(str(jsonl_file), byte_offset=99999)
        assert result is None


class TestExtractPdfsFromJsonlBatch:
    """Tests for extract_pdfs_from_jsonl_batch."""

    def test_batch_extraction(self, tmp_path: Path):
        lines = []
        expected: dict[int, bytes] = {}
        offset = 0
        for i in range(3):
            pdf_bytes = f"%PDF-{i}".encode()
            content = base64.b64encode(pdf_bytes).decode()
            line = json.dumps({"content": content}) + "\n"
            expected[offset] = pdf_bytes
            offset += len(line.encode())
            lines.append(line)

        jsonl_file = tmp_path / "batch.jsonl"
        jsonl_file.write_bytes("".join(lines).encode())

        result = extract_pdfs_from_jsonl_batch(str(jsonl_file), list(expected.keys()))
        for off, pdf in expected.items():
            assert result[off] == pdf

    def test_missing_file_returns_none_for_all(self, tmp_path: Path):
        result = extract_pdfs_from_jsonl_batch(str(tmp_path / "nonexistent.jsonl"), [0, 100])
        assert result[0] is None
        assert result[100] is None

    def test_bad_offset_stored_as_none(self, tmp_path: Path):
        """An invalid offset within an existing file should store None, not raise."""
        pdf_bytes = b"%PDF-valid"
        content = base64.b64encode(pdf_bytes).decode()
        line = json.dumps({"content": content}) + "\n"
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_bytes(line.encode())

        offsets = [0, 99999]
        result = extract_pdfs_from_jsonl_batch(str(jsonl_file), offsets)
        assert result[0] == pdf_bytes
        assert result[99999] is None
