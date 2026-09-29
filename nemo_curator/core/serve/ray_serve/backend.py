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

import os
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.core.serve.base import BaseModelConfig, InferenceBackend
from nemo_curator.core.serve.ray_serve.config import RayServeModelConfig
from nemo_curator.core.utils import get_free_port

if TYPE_CHECKING:
    from ray.serve.llm import LLMConfig

    from nemo_curator.core.serve.server import InferenceServer


class RayServeBackend(InferenceBackend):
    """Ray Serve backend for ``InferenceServer``."""

    def __init__(self, server: "InferenceServer") -> None:
        self._server = server

    def start(self) -> None:
        """Connect to Ray, deploy the models, and detach the driver."""
        self._configure_ray_serve_haproxy()

        import ray

        with ray.init(ignore_reinit_error=True):
            self._deploy()

    def stop(self) -> None:
        """Reconnect to Ray and tear down Ray Serve."""
        logger.info("Shutting down Ray Serve")
        try:
            import ray
            from ray import serve

            with ray.init(ignore_reinit_error=True):
                serve.shutdown()
        except Exception:  # noqa: BLE001
            logger.debug("serve.shutdown() failed (cluster may already be gone)")

        logger.info("Ray Serve stopped")

    def _deploy(self) -> None:
        """Deploy models onto the connected Ray cluster."""
        server = self._server
        server.port = get_free_port(server.port)

        model_names = [model.resolved_model_name for model in server.models]
        logger.info(f"Starting Ray Serve with models: {model_names} on port {server.port}")

        quiet_env = self._quiet_runtime_env() if not server.verbose else None
        llm_configs = [self._to_llm_config(model, quiet_runtime_env=quiet_env) for model in server.models]

        build_args: dict[str, Any] = {"llm_configs": llm_configs}
        if quiet_env:
            # Suppress access logs on the OpenAI ingress deployment too.
            build_args["ingress_deployment_config"] = {
                "ray_actor_options": {"runtime_env": quiet_env},
            }

        from ray import serve
        from ray.serve.llm import build_openai_app
        from ray.serve.schema import LoggingConfig

        logging_config = None
        if not server.verbose:
            logging_config = LoggingConfig(
                log_level="WARNING",
                enable_access_log=False,
            )

        app = build_openai_app(build_args)
        # ``serve.run()`` does not accept ``http_options`` and would otherwise
        # default to port 8000, so the controller must be started explicitly.
        serve.start(http_options={"port": server.port}, logging_config=logging_config)

        try:
            serve.run(app, name=server.name, blocking=False, logging_config=logging_config)
            server._wait_for_healthy(status_check=self._raise_if_app_failed)
        except Exception:
            self._cleanup_failed_deploy()
            raise

    def _raise_if_app_failed(self) -> None:
        """Raise with Ray Serve's own diagnosis once the application cannot come up."""
        from ray import serve
        from ray.serve.schema import ApplicationStatus, DeploymentStatus, DeploymentStatusTrigger

        try:
            app_status = serve.status().applications.get(self._server.name)
        except Exception:  # noqa: BLE001
            # A controller hiccup mid-load is not a deploy failure; keep polling.
            logger.debug("serve.status() failed while waiting for the application", exc_info=True)
            return
        if app_status is None:
            return
        # DEPLOY_FAILED/UNHEALTHY alone is not final: after a failed health check Serve
        # replaces the replica and the deployment can go back to HEALTHY. The triggers
        # below mean replicas (or the deployment actor) failed to start through Serve's
        # whole retry budget in a row (e.g. GPU OOM). If no replica ever started, Serve
        # stops retrying; otherwise it keeps trying, but we treat that as failed rather
        # than wait out health_check_timeout_s.
        fatal_triggers = {
            DeploymentStatusTrigger.REPLICA_STARTUP_FAILED,
            DeploymentStatusTrigger.DEPLOYMENT_ACTOR_FAILED,
        }
        failed = [
            f"{name}: {deployment.message}"
            for name, deployment in app_status.deployments.items()
            if deployment.status in {DeploymentStatus.DEPLOY_FAILED, DeploymentStatus.UNHEALTHY}
            and deployment.status_trigger in fatal_triggers
        ]
        # With no deployments listed, DEPLOY_FAILED means the app itself failed to build.
        if failed or (not app_status.deployments and app_status.status == ApplicationStatus.DEPLOY_FAILED):
            details = "; ".join(failed) or app_status.message
            msg = f"Ray Serve application {self._server.name!r} is {app_status.status.value}: {details}"
            raise RuntimeError(msg)

    @staticmethod
    def _quiet_runtime_env() -> dict[str, Any]:
        """Return a ``runtime_env`` dict that suppresses per-request logs."""
        return {
            "env_vars": {
                "VLLM_LOGGING_LEVEL": "WARNING",
                "RAY_SERVE_LOG_TO_STDERR": "0",
            },
        }

    @staticmethod
    def _configure_ray_serve_haproxy() -> None:
        """Set Ray Serve HAProxy defaults before importing Ray Serve."""
        os.environ.setdefault("RAY_SERVE_ENABLE_HA_PROXY", "1")

    @staticmethod
    def _to_llm_config(model: RayServeModelConfig, quiet_runtime_env: dict[str, Any] | None = None) -> "LLMConfig":
        """Translate a typed Ray Serve model config into ``LLMConfig``."""
        from ray.serve.llm import LLMConfig

        merged_env = BaseModelConfig.merge_runtime_envs(model.runtime_env, quiet_runtime_env)

        return LLMConfig(
            model_loading_config={
                "model_id": model.resolved_model_name,
                "model_source": model.model_identifier,
            },
            deployment_config=model.deployment_config,
            engine_kwargs=model.engine_kwargs,
            runtime_env=merged_env or None,
        )

    @staticmethod
    def _cleanup_failed_deploy() -> None:
        """Best-effort cleanup after a failed deploy."""
        from ray import serve

        try:
            serve.shutdown()
        except Exception:  # noqa: BLE001
            logger.debug("Cleanup: serve.shutdown() failed after failed deploy")
