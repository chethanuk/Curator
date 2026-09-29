"""Repro for NVIDIA-NeMo/Curator#1620 on a real local Ray Serve cluster (no GPU needed).

The LLM deployment starts once, then its replica dies (failed health check) and every
restart fails in the constructor with a simulated GPU OOM, like the nightly benchmark.
The ingress keeps answering /v1/models, but the model never shows up.
"""

import os
import sys
import tempfile
import time

import ray
from ray import serve
from starlette.responses import JSONResponse

from nemo_curator.core.serve import InferenceServer, RayServeModelConfig
from nemo_curator.core.serve.ray_serve.backend import RayServeBackend
from nemo_curator.core.utils import get_free_port

MARKER = os.path.join(tempfile.mkdtemp(), "started")


@serve.deployment(health_check_period_s=1, health_check_timeout_s=5)
class LLMServer:
    def __init__(self, marker: str) -> None:
        if os.path.exists(marker):
            raise ValueError("Free memory on device cuda:0 (27.44/79.15 GiB) is less than desired (71.24 GiB)")
        open(marker, "w").close()
        self.started = time.monotonic()

    def check_health(self) -> None:
        if time.monotonic() - self.started > 5:
            raise RuntimeError("engine died")


@serve.deployment
class Ingress:
    def __init__(self, llm) -> None:
        self.llm = llm

    async def __call__(self, request) -> JSONResponse:
        return JSONResponse({"data": []})


ray.init(address="local", num_cpus=4, object_store_memory=200 * 1024**2, include_dashboard=False, log_to_driver=False)
server = InferenceServer(models=[RayServeModelConfig(model_identifier="my-model")], health_check_timeout_s=60)
server.port = get_free_port(18000)
serve.start(http_options={"port": server.port})
serve.run(Ingress.bind(LLMServer.bind(MARKER)), name=server.name, route_prefix="/v1", blocking=False)
print(f"serve.run returned; application {server.name!r} is {serve.status().applications[server.name].status.value}")

backend = RayServeBackend(server)
started = time.monotonic()
try:
    if hasattr(backend, "_raise_if_app_failed"):
        server._wait_for_healthy(status_check=backend._raise_if_app_failed)
    else:
        server._wait_for_healthy()
except Exception as e:  # noqa: BLE001
    lines = str(e).splitlines()
    print(f"after {time.monotonic() - started:.0f}s -> {type(e).__name__}: {lines[0]}")
    if len(lines) > 1:
        print(f"    [{len(lines) - 2} traceback lines elided] ... {lines[-1]}")
    print(f"serve.status() at that point: {serve.status().applications[server.name].status.value}")
    sys.exit(1)
finally:
    serve.shutdown()
    ray.shutdown()
