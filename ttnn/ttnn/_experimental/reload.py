# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host support for the runtime binary reload (Blaze). Experimental; may change.

``set_configure_only`` puts a mesh's slow-dispatch path into configure-without-launch mode;
``read_core_l1`` / ``write_core_l1`` are raw per-core L1 access; ``read_kernel_config`` reads a
core's launch message back field by field. All take the MeshDevice first.
"""

from ttnn._ttnn.multi_device.experimental import (
    read_core_l1,
    read_kernel_config,
    set_configure_only,
    write_core_l1,
)

__all__ = ["read_core_l1", "read_kernel_config", "set_configure_only", "write_core_l1"]
