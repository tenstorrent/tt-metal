# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

Topology = ttnn._ttnn.operations.ccl.Topology

get_usable_topology = ttnn._ttnn.operations.ccl.get_usable_topology

# Experimental CCL enums for all_to_all_dispatch_metadata operation
DispatchAlgorithm = ttnn._ttnn.operations.experimental.ccl_experimental.DispatchAlgorithm
WorkerMode = ttnn._ttnn.operations.experimental.ccl_experimental.WorkerMode

# Experimental CCL enum for moe_compute operation
MoEActivationFunction = ttnn._ttnn.operations.experimental.ccl_experimental.MoEActivationFunction


def _golden_composed_identity(input_tensor, **_):
    """Global tensor view for inverse partition/gather collectives.

    Comparison-mode preprocessing composes mesh shards into a logical CPU tensor.
    Both operations preserve that composed value: all_gather changes shard placement
    to replication, while mesh_partition changes replication to shard placement.
    """

    return input_tensor.clone()


ttnn.attach_golden_function(ttnn.all_gather, golden_function=_golden_composed_identity)
ttnn.attach_golden_function(ttnn.mesh_partition, golden_function=_golden_composed_identity)

# The remaining planned CCL operations intentionally have no golden attachment:
# - all_broadcast returns device-source tensors whose grouping depends on cluster_axis.
# - all_reduce and reduce_scatter require preserving every device shard and independent
#   reduction groups; one composed CPU tensor loses that placement information.
# - point_to_point and reduce_to_root define values only on the receiver/root device.
# - all_to_all_dispatch and all_to_all_combine explicitly leave non-routed rows as
#   unspecified placeholder data.
# Comparison mode compares complete returned tensors, so attaching a dense or identity
# placeholder for any of these operations would assert semantics the operation does not have.

__all__ = ["Topology", "get_usable_topology", "DispatchAlgorithm", "WorkerMode", "MoEActivationFunction"]
