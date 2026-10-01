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

"""Readers must be scheduled as Ray Data tasks (not actors), including when IDs are enabled."""

import pytest

from nemo_curator.backends.ray_data.utils import is_actor_stage
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage
from nemo_curator.stages.text.io.reader.lance import LanceReaderStage
from nemo_curator.stages.text.io.reader.parquet import ParquetReaderStage

ID_KWARGS = {"none": {}, "generate": {"_generate_ids": True}, "assign": {"_assign_ids": True}}


@pytest.mark.parametrize("reader_cls", [ParquetReaderStage, JsonlReaderStage, LanceReaderStage])
@pytest.mark.parametrize("id_mode", ["none", "generate", "assign"])
def test_reader_is_ray_data_task(reader_cls: type, id_mode: str) -> None:
    stage = reader_cls(**ID_KWARGS[id_mode])
    # Mirrors RayDataStageAdapter's scheduling rule.
    assert stage.ray_stage_spec().get(RayStageSpecKeys.IS_ACTOR_STAGE, is_actor_stage(stage)) is False


def test_no_ids_cpu_gpu_reader_stays_task() -> None:
    stage = ParquetReaderStage().with_(resources=Resources(cpus=1, gpus=1))
    assert stage.ray_stage_spec().get(RayStageSpecKeys.IS_ACTOR_STAGE, is_actor_stage(stage)) is False
