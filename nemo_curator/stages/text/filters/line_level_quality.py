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

import re
from dataclasses import dataclass

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.utils.constants import policy_substrings, regex_url
from nemo_curator.stages.text.utils.text_utils import get_word_splitter
from nemo_curator.tasks import DocumentBatch

_RULES = ("url_only", "nav_pattern", "boilerplate", "repetition", "min_words")
_METRIC_KEYS = (*(f"lines_removed_{rule}" for rule in _RULES), "words_removed", "words_total", "documents_discarded")


@dataclass
class LineLevelQualityFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Removes low-quality lines from each document and keeps the rest, in order.
    A document is discarded only when too many of its words were removed.
    Meant to run after document-level filters such as ScoreFilter, not instead of them.

    Each non-blank line is split into words with ``get_word_splitter(lang)`` (whitespace
    for most languages, jieba for "zh", MeCab for "ja") and removed by the first rule it fails,
    checked in this order:

    1. ``url_only``: the stripped line is a single URL (``remove_url_only_lines``).
    2. ``nav_pattern``: the line matches ``nav_pattern`` (``re.search``).
    3. ``boilerplate``: the lowercased line contains one of ``boilerplate_strings``.
    4. ``repetition``: ``1 - unique_words / words > max_line_repetition_ratio``.
    5. ``min_words``: the line has fewer than ``min_line_words`` words.

    Blank lines are kept and not counted. Kept lines are joined with ``"\\n"``; a document with
    no removed line keeps its original text unchanged. The document is discarded when
    ``words_removed / words_total > max_removal_ratio``. Non-string values pass through untouched.

    This is not a full C4 port: the end-mark, long-word, ``javascript``, ``{`` and ``lorem ipsum``
    rules and the minimum-sentence gate are not applied. Add ``"javascript"`` or ``"lorem ipsum"``
    to ``boilerplate_strings`` to remove such lines.

    Custom metrics per ``process_batch`` call: ``lines_removed_<rule>`` for each rule above,
    ``words_removed``, ``words_total`` and ``documents_discarded``. They are summed over every task
    in one call. With ``batch_size > 1`` the executor attaches that call total to every output
    task; ``TaskPerfUtils.collect_stage_metrics`` counts it once in-process, but after
    serialization across workers a sum repeats it per output. At the default ``batch_size = 1``
    the sums are exact.

    Args:
        text_field (str): The field the documents are read from and written to.
        max_removal_ratio (float): Discard a document when the removed word ratio is above this, in [0, 1].
        min_line_words (int | None): Remove lines with fewer words than this. None disables the rule.
        max_line_repetition_ratio (float | None): Remove lines whose repeated word ratio is above this,
            in [0, 1]. None disables the rule.
        remove_url_only_lines (bool): Remove lines that consist of a single URL.
        nav_pattern (str | None): Regex for navigation or menu lines. None disables the rule.
        boilerplate_strings (tuple[str, ...]): Case-insensitive substrings marking boilerplate lines.
            An empty tuple disables the rule.
        lang (str): ISO 639-1 language code used to pick the word splitter.

    """

    text_field: str = "text"
    max_removal_ratio: float = 0.3
    min_line_words: int | None = 5
    max_line_repetition_ratio: float | None = None
    remove_url_only_lines: bool = True
    nav_pattern: str | None = None
    boilerplate_strings: tuple[str, ...] = tuple(policy_substrings)
    lang: str = "en"
    name: str = "line_level_quality_filter"

    def __post_init__(self):
        for field_name in ("max_removal_ratio", "max_line_repetition_ratio"):
            value = getattr(self, field_name)
            if value is not None and not 0 <= value <= 1:
                msg = f"{field_name} must be in [0, 1], got {value}"
                raise ValueError(msg)
        if self.min_line_words is not None and self.min_line_words < 1:
            msg = f"min_line_words must be None or >= 1, got {self.min_line_words}"
            raise ValueError(msg)
        try:
            self._nav_regex = re.compile(self.nav_pattern) if self.nav_pattern is not None else None
        except re.error as e:
            msg = f"nav_pattern is not a valid regex: {self.nav_pattern!r}"
            raise ValueError(msg) from e
        self._boilerplate = tuple(s.lower() for s in self.boilerplate_strings)
        self._split_words = get_word_splitter(self.lang)

    def inputs(self) -> tuple[list[str], list[str]]:
        """Requires the ``data`` attribute with the ``text_field`` column."""
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        """Produces the ``data`` attribute with the cleaned ``text_field`` column."""
        return ["data"], [self.text_field]

    def _failed_rule(self, line: str, words: list[str]) -> str | None:
        if self.remove_url_only_lines and regex_url.fullmatch(line.strip()):
            return "url_only"
        if self._nav_regex is not None and self._nav_regex.search(line):
            return "nav_pattern"
        if self._boilerplate:
            lowered = line.lower()
            if any(s in lowered for s in self._boilerplate):
                return "boilerplate"
        if (
            self.max_line_repetition_ratio is not None
            and words
            and 1 - len(set(words)) / len(words) > self.max_line_repetition_ratio
        ):
            return "repetition"
        if self.min_line_words is not None and len(words) < self.min_line_words:
            return "min_words"
        return None

    def _clean(self, text: object, counts: dict[str, float]) -> tuple[object, bool]:
        if not isinstance(text, str):
            return text, True

        lines = text.splitlines()
        kept_lines = []
        total = removed = 0
        for line in lines:
            if not line.strip():
                kept_lines.append(line)
                continue
            words = self._split_words(line)
            total += len(words)
            rule = self._failed_rule(line, words)
            if rule is None:
                kept_lines.append(line)
            else:
                counts[f"lines_removed_{rule}"] += 1
                removed += len(words)

        counts["words_total"] += total
        counts["words_removed"] += removed
        if total > 0 and removed / total > self.max_removal_ratio:
            counts["documents_discarded"] += 1
            return text, False
        if len(kept_lines) == len(lines):
            return text, True
        return "\n".join(kept_lines), True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """
        Removes failing lines from every document and drops documents over the removal ratio.

        Args:
            batch (DocumentBatch): The batch to clean.

        Returns:
            DocumentBatch: The kept rows with cleaned text; other columns and the index are unchanged.
                Empty when every row is discarded, so the metrics still reach the executor.

        """
        df = batch.to_pandas()
        counts = dict.fromkeys(_METRIC_KEYS, 0.0)

        cleaned, keep = [], []
        for text in df[self.text_field]:
            new_text, kept = self._clean(text, counts)
            cleaned.append(new_text)
            keep.append(kept)
        df = df.assign(**{self.text_field: cleaned}).loc[keep]

        if len(df) == 0:
            logger.info(f"All documents filtered out for batch {batch.task_id}")

        prev = getattr(self, "_custom_metrics", None) or {}
        self._log_metrics({k: prev.get(k, 0.0) + v for k, v in counts.items()})

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def process_batch(self, tasks: list[DocumentBatch]) -> list[DocumentBatch]:
        """
        Cleans each task and sums the metrics over this call only.

        Metrics left over from an earlier call that raised before being consumed are cleared first.

        Args:
            tasks (list[DocumentBatch]): The batches to clean.

        Returns:
            list[DocumentBatch]: One cleaned batch per input task.

        """
        self._custom_metrics = {}
        return super().process_batch(tasks)
