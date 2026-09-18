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
import subprocess

import pytest
import ray

from nemo_curator.core.utils import ignore_ray_head_node, init_cluster


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("no", False),
        *[(v, True) for v in ("1", "true", "TRUE", "yes", " 1 ")],
    ],
)
def test_ignore_ray_head_node_env_parsing(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool) -> None:
    if value is None:
        monkeypatch.delenv("CURATOR_IGNORE_RAY_HEAD_NODE", raising=False)
    else:
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", value)
    assert ignore_ray_head_node() is expected


@pytest.mark.parametrize("enable_object_spilling", [True, False])
def test_init_cluster_never_names_a_shared_spill_directory(
    monkeypatch: pytest.MonkeyPatch,
    enable_object_spilling: bool,
) -> None:
    """Spilled objects must land under ``--temp-dir``, which is per-user, not a shared path.

    ``ray start --system-config`` takes JSON, so an unparseable value stops the cluster from
    starting at all. Both properties are asserted on the command Curator actually builds.
    """
    captured: list[list[str]] = []

    def fake_popen(args: list[str], **_kwargs: object) -> object:
        captured.append(args)
        return object()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ray.util, "register_serializer", lambda *_a, **_kw: None)

    init_cluster(
        ray_port=6379,
        ray_temp_dir="/tmp/ray_temp_dir_under_test",  # noqa: S108
        ray_dashboard_port=8265,
        ray_metrics_port=8080,
        ray_client_server_port=10001,
        ray_dashboard_host="127.0.0.1",
        enable_object_spilling=enable_object_spilling,
        block=False,
    )

    (ray_command,) = captured
    assert "/tmp/ray_spill" not in " ".join(ray_command)  # noqa: S108

    if not enable_object_spilling:
        assert "--system-config" not in ray_command
        return

    system_config = ray_command[ray_command.index("--system-config") + 1]
    parsed = json.loads(system_config)
    assert parsed == {"local_fs_capacity_threshold": 0.95}
    assert "object_spilling_config" not in parsed
