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
        ("tutorials/x.ipynb", True),
        ("fern/docs.yml", True),
        ("eval/run.py", True),
        ("nemo_curator/__init__.py", False),
        ("nemo_curator/eval/__init__.py", False),
        ("tests/backends/test_utils.py", False),
        ("benchmarking/run.py", False),
    ],
)
def test_ray_working_dir_upload_skips_only_docs_tutorials_fern_and_eval(relative_path: str, excluded: bool) -> None:
    assert (REPO_ROOT / ".rayignore").is_file()
    excludes = get_excludes_from_ignore_files(REPO_ROOT, include_gitignore=True)
    assert any(match(REPO_ROOT / relative_path) for match in excludes) is excluded
