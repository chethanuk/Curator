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

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.filters import LineLevelQualityFilter, ScoreFilter
from nemo_curator.stages.text.filters.heuristic import WordCountFilter
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.tasks import DocumentBatch
from nemo_curator.tasks.utils import TaskPerfUtils

# Stand-ins for crawl extracts. The stage ships no default nav regex; this one is test-only.
NAV = r"^\s*\w+(\s*\|\s*\w+){2,}\s*$"

NAV_LINE = "Home | News | Sport | Weather | Contact"
COOKIE_LINE = "This website uses cookies, see our privacy policy."
URL_LINE = "https://example.com/share?utm_source=twitter"
SEVEN = "Our shop opens at nine every day."

C1 = "The garden club met on Sunday morning to plant roses along the path."
C2 = "Volunteers brought spades, gloves and forty seedlings from the local nursery."
C3 = "Children from the primary school helped water every bed before lunch."
C4 = "The club secretary thanked the council for donating fresh compost this year."
C5 = "Next month the group will prune the apple trees near the old gate."


def _lines(*lines: str) -> str:
    return "\n".join(lines)


GARDEN = _lines(NAV_LINE, C1, C2, "", C3, URL_LINE, C4, C5, COOKIE_LINE)
SPAM = _lines(NAV_LINE, C1, URL_LINE, COOKIE_LINE, NAV_LINE, URL_LINE, COOKIE_LINE)


def test_cleans_boilerplate_lines_and_keeps_the_document() -> None:
    batch = DocumentBatch(data=pd.DataFrame({"text": [GARDEN], "id": [7]}), dataset_name="test_1")

    result = LineLevelQualityFilter(nav_pattern=NAV).process(batch).to_pandas()

    assert len(result) == 1
    assert result["text"].iloc[0] == _lines(C1, C2, "", C3, C4, C5)
    assert result["id"].iloc[0] == 7


def _batch(text: object, name: str = "t") -> DocumentBatch:
    return DocumentBatch(data=pd.DataFrame({"text": [text]}), dataset_name=name)


GARDEN_CLEAN = _lines(C1, C2, "", C3, C4, C5)
DROPPED = object()


@pytest.mark.parametrize(
    ("text", "config", "expected"),
    [
        pytest.param(GARDEN, {"nav_pattern": NAV}, GARDEN_CLEAN, id="garden"),
        pytest.param(SPAM, {"nav_pattern": NAV}, DROPPED, id="spam"),
        pytest.param(_lines(SEVEN, *[URL_LINE] * 3), {}, SEVEN, id="at-threshold"),
        pytest.param(_lines(SEVEN, *[URL_LINE] * 4), {}, DROPPED, id="over-threshold"),
        pytest.param(_lines(C1, C2, C3, C4, C5), {}, _lines(C1, C2, C3, C4, C5), id="clean-doc"),
        pytest.param("", {}, "", id="empty"),
        pytest.param("  \n ", {}, "  \n ", id="blank"),
        pytest.param(None, {}, None, id="none"),
        pytest.param(f"{C1}\r\n{URL_LINE}\r\n{C2}", {}, f"{C1}\n{C2}", id="crlf"),
        pytest.param(_lines(C1, f"  {URL_LINE}  ", C2), {}, f"{C1}\n{C2}", id="url-padded"),
        pytest.param(GARDEN, {}, _lines(NAV_LINE, C1, C2, "", C3, C4, C5), id="nav-off-default"),
        pytest.param(
            _lines(C1, C2, C3, "buy now buy now buy now buy now"),
            {"max_line_repetition_ratio": 0.5},
            _lines(C1, C2, C3),
            id="repetition",
        ),
        pytest.param(
            _lines(
                C1, C2, C3, "Enable javascript to continue reading.", "Lorem ipsum dolor sit amet today.", COOKIE_LINE
            ),
            {"boilerplate_strings": ("javascript", "lorem ipsum")},
            _lines(C1, C2, C3, COOKIE_LINE),
            id="custom-boilerplate",
        ),
        pytest.param(
            GARDEN,
            {"nav_pattern": NAV, "min_line_words": None, "boilerplate_strings": (), "remove_url_only_lines": False},
            _lines(C1, C2, "", C3, URL_LINE, C4, C5, COOKIE_LINE),
            id="rules-off",
        ),
        pytest.param(
            "首页\n园艺俱乐部周日上午在小路旁种植玫瑰。",
            {"lang": "zh", "min_line_words": 2},
            "园艺俱乐部周日上午在小路旁种植玫瑰。",
            id="zh",
        ),
    ],
)
def test_cleans_lines_and_gates_on_removed_word_ratio(text: object, config: dict[str, Any], expected: object) -> None:
    stage = LineLevelQualityFilter(**config)
    batch = DocumentBatch(data=pd.DataFrame({"text": [text], "id": [7]}), dataset_name="t")

    result = stage.process(batch).to_pandas()

    if expected is DROPPED:
        assert len(result) == 0
        return
    assert len(result) == 1
    assert result["id"].iloc[0] == 7
    assert result["text"].iloc[0] == expected
    if isinstance(expected, str):
        assert stage.process(_batch(expected)).to_pandas()["text"].iloc[0] == expected


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"max_removal_ratio": -0.1}, id="negative-removal-ratio"),
        pytest.param({"max_removal_ratio": 1.5}, id="removal-ratio-above-one"),
        pytest.param({"max_line_repetition_ratio": 1.5}, id="repetition-ratio-above-one"),
        pytest.param({"min_line_words": 0}, id="zero-min-words"),
        pytest.param({"nav_pattern": "("}, id="invalid-nav-regex"),
    ],
)
def test_rejects_bad_config(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        LineLevelQualityFilter(**config)


def test_empty_batch_keeps_its_columns() -> None:
    batch = DocumentBatch(data=pd.DataFrame({"text": [], "id": []}), dataset_name="t")

    result = LineLevelQualityFilter().process(batch).to_pandas()

    assert result.empty
    assert list(result.columns) == ["text", "id"]


@pytest.mark.usefixtures("shared_ray_client")
@pytest.mark.parametrize(
    ("backend_cls", "backend_config"),
    [
        pytest.param(RayDataExecutor, {}, id="ray_data"),
        pytest.param(XennaExecutor, {"execution_mode": "streaming"}, id="xenna_streaming"),
    ],
)
def test_pipeline_cleans_and_reports_rule_counts(
    tmp_path: Path, backend_cls: type, backend_config: dict[str, Any]
) -> None:
    (tmp_path / "a.jsonl").write_text(
        "".join(json.dumps({"text": t}) + "\n" for t in [GARDEN, "Click here."]), encoding="utf-8"
    )
    (tmp_path / "b.jsonl").write_text(json.dumps({"text": SPAM}) + "\n", encoding="utf-8")
    pipeline = Pipeline(
        name="llq",
        stages=[
            JsonlReader(file_paths=str(tmp_path), files_per_partition=1),
            ScoreFilter(WordCountFilter(min_words=20)),
            LineLevelQualityFilter(nav_pattern=NAV),
        ],
    )

    out = pipeline.run(backend_cls(backend_config))

    assert len(out) == 2
    texts = [t for batch in out for t in batch.to_pandas()["text"]]
    assert texts == [GARDEN_CLEAN]
    m = TaskPerfUtils.collect_stage_metrics(out)["line_level_quality_filter"]
    for rule, expected in [
        ("url_only", 3),
        ("nav_pattern", 3),
        ("boilerplate", 3),
        ("repetition", 0),
        ("min_words", 0),
    ]:
        assert m[f"custom.lines_removed_{rule}"].sum() == expected, rule
    assert m["custom.documents_discarded"].sum() == 1


def test_counts_add_up_across_one_call() -> None:
    stage = LineLevelQualityFilter(nav_pattern=NAV)

    stage.process_batch([_batch(GARDEN), _batch(SPAM)])
    metrics = stage._consume_custom_metrics()

    assert metrics["lines_removed_url_only"] == 3
    assert metrics["documents_discarded"] == 1
    assert metrics["words_total"] == 127


def test_failed_call_does_not_leak_counts_into_next_call() -> None:
    stage = LineLevelQualityFilter(nav_pattern=NAV)
    bad = DocumentBatch(data=pd.DataFrame({"body": ["x"]}), dataset_name="t")
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([_batch(GARDEN), bad])

    for _ in range(2):
        stage.process_batch([_batch(GARDEN)])
        metrics = stage._consume_custom_metrics()

        assert metrics["lines_removed_url_only"] == 1
        assert metrics["words_total"] == 78
