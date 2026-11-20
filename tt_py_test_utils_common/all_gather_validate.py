# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn


def ti_cond_skip(condition, reason):
    if condition:
        pytest.skip("Skipping unsupported case: " + reason)


def validate_test(num_devices, topology, shape, cluster_axis, mesh_device=None):
    if topology is None:
        assert mesh_device is not None, "validate_test needs mesh_device to read the fabric topology"
        fabric_config = ttnn.get_fabric_config(mesh_device)
        topology = ttnn.Topology.Ring if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING else ttnn.Topology.Linear
    ti_cond_skip((1 == num_devices), "Can't run a CCL test on 1 device")
    ti_cond_skip(
        ((shape[cluster_axis] != num_devices) and (topology == ttnn.Topology.Ring)),
        "Ring configuration requires the entire row or column so it loops around",
    )
    ti_cond_skip((shape[cluster_axis] < num_devices), "Test requires more devices than are available on this platform")
