# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

__all__ = []

Topology = ttnn._ttnn.operations.ccl.Topology
initialize_edm_fabric = ttnn._ttnn.operations.ccl.initialize_edm_fabric
teardown_edm_fabric = ttnn._ttnn.operations.ccl.teardown_edm_fabric

# TODO: Add golden functions (#12747)


__all__ = []
