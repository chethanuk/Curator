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

from unittest import mock

import pytest

from nemo_curator.core.serve import InferenceServer, RayServeModelConfig
from nemo_curator.core.serve.ray_serve.backend import RayServeBackend

ApplicationStatus = pytest.importorskip("ray.serve.schema", reason="ray[serve] not installed").ApplicationStatus


class TestRayServeBackend:
    def test_to_llm_config_reads_typed_model_config(self) -> None:
        model = RayServeModelConfig(
            model_identifier="google/gemma-3-27b-it",
            model_name="gemma-27b",
            deployment_config={"autoscaling_config": {"min_replicas": 1}},
            engine_kwargs={"tensor_parallel_size": 4},
            runtime_env={
                "pip": ["my-package"],
                "env_vars": {"MY_VAR": "1", "VLLM_LOGGING_LEVEL": "DEBUG"},
            },
        )

        llm = pytest.importorskip("ray.serve.llm", reason="ray[serve] LLM extras not installed", exc_type=ImportError)

        quiet_env = RayServeBackend._quiet_runtime_env()
        result = RayServeBackend._to_llm_config(model, quiet_runtime_env=quiet_env)

        assert isinstance(result, llm.LLMConfig)
        assert result.model_loading_config.model_id == "gemma-27b"
        assert result.model_loading_config.model_source == "google/gemma-3-27b-it"
        assert result.deployment_config == {"autoscaling_config": {"min_replicas": 1}}
        assert result.engine_kwargs == {"tensor_parallel_size": 4}
        assert result.runtime_env["pip"] == ["my-package"]
        assert result.runtime_env["env_vars"]["MY_VAR"] == "1"
        assert result.runtime_env["env_vars"]["VLLM_LOGGING_LEVEL"] == "WARNING"
        assert result.runtime_env["env_vars"]["RAY_SERVE_LOG_TO_STDERR"] == "0"

    @pytest.mark.parametrize("status", [ApplicationStatus.DEPLOY_FAILED, ApplicationStatus.UNHEALTHY])
    def test_raise_if_app_failed_surfaces_ray_serve_message(self, status: ApplicationStatus) -> None:
        backend = RayServeBackend(InferenceServer(models=[], name="curator-app"))
        serve_status = mock.Mock(
            applications={"curator-app": mock.Mock(status=status, message="Replica died: CUDA out of memory")}
        )

        with (
            mock.patch("ray.serve.status", return_value=serve_status),
            pytest.raises(RuntimeError, match="Replica died: CUDA out of memory"),
        ):
            backend._raise_if_app_failed()

    @pytest.mark.parametrize(
        "applications",
        [
            pytest.param({}, id="application-not-registered-yet"),
            pytest.param(
                {"curator-app": mock.Mock(status=ApplicationStatus.RUNNING, message="")},
                id="application-running",
            ),
        ],
    )
    def test_raise_if_app_failed_ignores_non_terminal_status(self, applications: dict[str, mock.Mock]) -> None:
        backend = RayServeBackend(InferenceServer(models=[], name="curator-app"))

        with mock.patch("ray.serve.status", return_value=mock.Mock(applications=applications)):
            backend._raise_if_app_failed()
