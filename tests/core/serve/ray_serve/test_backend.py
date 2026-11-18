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

import sys
import threading
import time
from unittest import mock

import pytest

from nemo_curator.core.serve import InferenceServer, RayServeModelConfig
from nemo_curator.core.serve.ray_serve.backend import RayServeBackend

schema = pytest.importorskip("ray.serve.schema", reason="ray[serve] not installed")
ApplicationStatusOverview = schema.ApplicationStatusOverview
ServeStatus = schema.ServeStatus


def _app(
    status: str, deployment: tuple[str, str] | None = None, message: str = "", deployment_message: str = ""
) -> dict[str, ApplicationStatusOverview]:
    deployments = {}
    if deployment is not None:
        deployments["LLMServer"] = schema.DeploymentStatusOverview(
            status=schema.DeploymentStatus(deployment[0]),
            status_trigger=schema.DeploymentStatusTrigger(deployment[1]),
            replica_states={},
            message=deployment_message,
        )
    return {
        "curator-app": ApplicationStatusOverview(
            status=schema.ApplicationStatus(status),
            message=message,
            last_deployed_time_s=0.0,
            deployments=deployments,
        )
    }


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

    @pytest.mark.parametrize(
        ("applications", "error"),
        [
            pytest.param({}, None, id="application-not-registered-yet"),
            pytest.param(_app("RUNNING", ("HEALTHY", "CONFIG_UPDATE_COMPLETED")), None, id="running"),
            pytest.param(
                _app("UNHEALTHY", ("UNHEALTHY", "HEALTH_CHECK_FAILED")), None, id="health-check-failed-recovers"
            ),
            pytest.param(
                _app("DEPLOY_FAILED", ("DEPLOY_FAILED", "HEALTH_CHECK_FAILED")),
                None,
                id="deploy-failed-health-check-recovers",
            ),
            pytest.param(
                _app("UNHEALTHY", ("UNHEALTHY", "REPLICA_STARTUP_FAILED"), deployment_message="CUDA out of memory"),
                "is UNHEALTHY: LLMServer: CUDA out of memory",
                id="replica-startup-retries-exhausted",
            ),
            pytest.param(
                _app(
                    "DEPLOY_FAILED",
                    ("DEPLOY_FAILED", "REPLICA_STARTUP_FAILED"),
                    deployment_message="CUDA out of memory",
                ),
                "is DEPLOY_FAILED: LLMServer: CUDA out of memory",
                id="deploy-failed",
            ),
            pytest.param(
                _app("DEPLOY_FAILED", ("DEPLOY_FAILED", "DEPLOYMENT_ACTOR_FAILED"), deployment_message="actor died"),
                "is DEPLOY_FAILED: LLMServer: actor died",
                id="deployment-actor-failed",
            ),
            pytest.param(
                _app("DEPLOY_FAILED", message="Failed to build app"),
                "is DEPLOY_FAILED: Failed to build app",
                id="deploy-failed-without-deployments",
            ),
        ],
    )
    def test_raise_if_app_failed_only_on_unrecoverable_status(
        self, applications: dict[str, ApplicationStatusOverview], error: str | None
    ) -> None:
        backend = RayServeBackend(InferenceServer(models=[], name="curator-app"))

        with mock.patch("ray.serve.status", return_value=ServeStatus(applications=applications)):
            if error is None:
                backend._raise_if_app_failed()
            else:
                with pytest.raises(RuntimeError, match=error):
                    backend._raise_if_app_failed()

    def test_raise_if_app_failed_ignores_status_errors(self) -> None:
        backend = RayServeBackend(InferenceServer(models=[], name="curator-app"))

        with mock.patch("ray.serve.status", side_effect=ConnectionError("controller restarting")):
            backend._raise_if_app_failed()

    def test_deploy_cleans_up_and_raises_when_app_fails(self) -> None:
        server = InferenceServer(models=[], name="curator-app", health_check_timeout_s=30)
        backend = RayServeBackend(server)
        failed = ServeStatus(applications=_app("DEPLOY_FAILED", message="Failed to build app"))

        with (
            mock.patch.dict(sys.modules, {"ray.serve.llm": mock.MagicMock()}),
            mock.patch("ray.serve.start"),
            mock.patch("ray.serve.run"),
            mock.patch("ray.serve.status", return_value=failed),
            mock.patch("ray.serve.shutdown") as shutdown,
            pytest.raises(RuntimeError, match="is DEPLOY_FAILED: Failed to build app"),
        ):
            backend._deploy()

        # The failed deploy must not leave Serve running.
        shutdown.assert_called_once()

    def test_wait_for_healthy_times_out_when_serve_status_hangs(self) -> None:
        server = InferenceServer(models=[], name="curator-app", port=19879, health_check_timeout_s=2)
        backend = RayServeBackend(server)
        release = threading.Event()

        started = time.monotonic()
        try:
            # serve.status() blocks on the controller with no timeout of its own.
            with (
                mock.patch("ray.serve.status", side_effect=lambda: release.wait(20)),
                pytest.raises(TimeoutError, match="did not become ready within 2s"),
            ):
                server._wait_for_healthy(status_check=backend._raise_if_app_failed)
        finally:
            release.set()

        assert time.monotonic() - started < 10
