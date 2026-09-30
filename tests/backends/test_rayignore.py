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

from pathlib import Path

import pytest

# Private Ray API: the same matcher `ray job submit --working-dir .` uses to skip files.
from ray._private.runtime_env.packaging import get_excludes_from_ignore_files

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("relative_path", "excluded"),
    [
        ("docs/README.md", True),
        ("fern/docs.yml", True),
        ("nemo_curator/__init__.py", False),
        ("nemo_curator/eval/__init__.py", False),
        ("tests/backends/test_utils.py", False),
        ("benchmarking/run.py", False),
        # Files that tests/ and benchmarking/ import or read, so they must ship with them.
        ("eval/video/caption_clipscore.py", False),  # tests/eval/video/test_caption_clipscore.py
        ("tutorials/audio/nemo_fastconformer/pipeline.yaml", False),  # tests/config/test_run.py
        ("tutorials/eval/llm_judge/cc_extract_example/pipeline.yaml", False),  # tests/eval/llm_judge
        ("tutorials/video/getting-started/video_split_clip_example.py", False),  # video_pipeline_benchmark.py
        ("tutorials/interleaved/nemotron_parse_pdf/pipeline_utils.py", False),  # nemotron_parse_pdf_benchmark.py
    ],
)
def test_ray_working_dir_upload_skips_only_docs_and_fern(relative_path: str, excluded: bool) -> None:
    assert (REPO_ROOT / ".rayignore").is_file()
    excludes = get_excludes_from_ignore_files(REPO_ROOT, include_gitignore=True)
    assert any(match(REPO_ROOT / relative_path) for match in excludes) is excluded
