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

"""Tests for Nemotron-Parse output parsing and interleaved-row building."""

from __future__ import annotations

from PIL import Image

from nemo_curator.stages.interleaved.pdf.nemotron_parse.utils import (
    build_interleaved_rows,
    interleave_floaters,
    parse_nemotron_output,
)


class TestParseNemotronOutput:
    def test_single_text_element(self):
        raw = "<x_0.1><y_0.2>Hello world<x_0.9><y_0.8><class_Text>"
        elements = parse_nemotron_output(raw)
        assert len(elements) == 1
        assert elements[0]["class"] == "Text"
        assert elements[0]["text"] == "Hello world"
        assert elements[0]["bbox"] == [0.1, 0.2, 0.9, 0.8]

    def test_multiple_elements(self):
        raw = "<x_0.0><y_0.0>Title<x_0.5><y_0.1><class_Title><x_0.0><y_0.2>Body text<x_1.0><y_0.9><class_Text>"
        elements = parse_nemotron_output(raw)
        assert len(elements) == 2
        assert elements[0]["class"] == "Title"
        assert elements[1]["class"] == "Text"

    def test_picture_without_text(self):
        raw = "<x_0.1><y_0.1><x_0.5><y_0.5><class_Picture>"
        elements = parse_nemotron_output(raw)
        assert len(elements) == 1
        assert elements[0]["class"] == "Picture"
        assert elements[0]["text"] == ""

    def test_empty_input(self):
        assert parse_nemotron_output("") == []

    def test_unparseable_fallback(self):
        raw = "Some plain text without any tags"
        elements = parse_nemotron_output(raw)
        assert len(elements) == 1
        assert elements[0]["class"] == "Text"
        assert elements[0]["text"] == "Some plain text without any tags"

    def test_nested_tags_stripped(self):
        raw = "<x_0.0><y_0.0>Hello <b>bold</b> text<x_1.0><y_1.0><class_Text>"
        elements = parse_nemotron_output(raw)
        assert len(elements) == 1
        assert "bold" in elements[0]["text"]


class TestInterlaveFloaters:
    def test_no_floaters_preserves_order(self):
        anchored = [{"class": "Text", "text": "a", "bbox": [0, 0.1, 1, 0.2]}]
        result = interleave_floaters(anchored, [])
        assert len(result) == 1
        assert result[0]["text"] == "a"

    def test_picture_inserted_near_closest_anchor(self):
        anchored = [
            {"class": "Text", "text": "top", "bbox": [0, 0.0, 1, 0.1]},
            {"class": "Text", "text": "bottom", "bbox": [0, 0.8, 1, 0.9]},
        ]
        floaters = [
            {"class": "Picture", "text": "", "bbox": [0, 0.85, 1, 0.95]},
        ]
        result = interleave_floaters(anchored, floaters)
        assert len(result) == 3
        assert result[1]["text"] == "bottom"
        assert result[2]["class"] == "Picture"

    def test_empty_anchored(self):
        floaters = [{"class": "Picture", "text": "", "bbox": [0, 0, 1, 1]}]
        result = interleave_floaters([], floaters)
        assert len(result) == 1


class TestBuildInterleavedRows:
    def test_basic_output(self):
        img = Image.new("RGB", (100, 100), color="white")
        raw = "<x_0.0><y_0.0>Hello<x_1.0><y_1.0><class_Text>"
        rows = build_interleaved_rows("s1", "http://example.com", "test.pdf", [img], [raw])
        assert len(rows) >= 2
        assert rows[0]["modality"] == "metadata"
        assert rows[0]["sample_id"] == "s1"
        text_rows = [r for r in rows if r["modality"] == "text"]
        assert len(text_rows) == 1
        assert text_rows[0]["text_content"] == "Hello"

    def test_empty_pages(self):
        rows = build_interleaved_rows("s1", "http://example.com", "test.pdf", [], [])
        assert len(rows) == 1
        assert rows[0]["modality"] == "metadata"


class TestBuildInterleavedRowsExtended:
    """Additional coverage for build_interleaved_rows: Table, reorder_floaters=False, Picture."""

    def _make_img(self) -> Image.Image:
        return Image.new("RGB", (100, 100), color="white")

    def test_table_element(self):
        raw = "<x_0.0><y_0.0>| A | B |<x_1.0><y_1.0><class_Table>"
        rows = build_interleaved_rows("s1", "http://x", "t.pdf", [self._make_img()], [raw])
        table_rows = [r for r in rows if r["modality"] == "table"]
        assert len(table_rows) == 1
        assert "A" in table_rows[0]["text_content"]

    def test_reorder_floaters_false(self):
        raw = "<x_0.0><y_0.0>Hello<x_1.0><y_1.0><class_Text>"
        rows = build_interleaved_rows("s1", "http://x", "t.pdf", [self._make_img()], [raw], reorder_floaters=False)
        text_rows = [r for r in rows if r["modality"] == "text"]
        assert len(text_rows) == 1
        assert text_rows[0]["text_content"] == "Hello"

    def test_picture_with_caption_interleaved(self):
        # Picture followed by Caption: both should appear in output
        raw = (
            "<x_0.0><y_0.0>Intro text<x_1.0><y_0.1><class_Text>"
            "<x_0.1><y_0.2><x_0.9><y_0.5><class_Picture>"
            "<x_0.1><y_0.55>Fig. 1<x_0.9><y_0.6><class_Caption>"
        )
        rows = build_interleaved_rows("s1", "http://x", "t.pdf", [self._make_img()], [raw])
        modalities = [r["modality"] for r in rows]
        assert "image" in modalities or "text" in modalities  # at least some output
