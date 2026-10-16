# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh parametrization for bring-up op tests (from mstaletovic/mimo-v2-dp models/demos/mimo_v2_d_p/tests/mesh.py): SP on rows, TP on cols.

QuietBox bringup runs 2x2 (SP2 x TP2); the same code targets a BH Galaxy as 8x4 (SP8 x TP4, EP32).
"""

import os

import pytest

from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_y_device_params

# BRINGUP_TEST_TRACE_REGION (bytes): a trace region for traced runs
_TRACE = (
    {"trace_region_size": int(os.environ["BRINGUP_TEST_TRACE_REGION"])}
    if os.environ.get("BRINGUP_TEST_TRACE_REGION")
    else {}
)
QB_MESHES = [pytest.param((2, 2), fabric2d_device_params(**_TRACE), id="2x2")]
# BRINGUP_TEST_MESH=RxC (e.g. 8x4): run the MESH_PARAMS tests on that mesh instead (FABRIC_2D), e.g. the BH Galaxy layout
# under tt-emule
if os.environ.get("BRINGUP_TEST_MESH"):
    _shape = tuple(int(v) for v in os.environ["BRINGUP_TEST_MESH"].split("x"))
    QB_MESHES = [pytest.param(_shape, fabric2d_device_params(**_TRACE), id=os.environ["BRINGUP_TEST_MESH"])]

MESH_PARAMS = pytest.mark.parametrize(
    "mesh_device, device_params", QB_MESHES, indirect=["mesh_device", "device_params"]
)


# SP generalization (op tests only, not the model): 4x1 = a 4-long SP ring (2D torus along the rows), TP 1.
SP4_MESHES = QB_MESHES + [pytest.param((4, 1), torus_y_device_params(**_TRACE), id="4x1")]
SP_MESH_PARAMS = pytest.mark.parametrize(
    "mesh_device, device_params", SP4_MESHES, indirect=["mesh_device", "device_params"]
)


def sp_tp(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    return rows, cols, 0, 1


def mesh_id(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    return f"{rows}x{cols}"
