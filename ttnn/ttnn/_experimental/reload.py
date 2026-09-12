# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host support for runtime binary reload. Experimental; may change.

``set_configure_only`` puts a mesh's slow-dispatch path into configure-without-launch mode;
``read_core_l1`` / ``write_core_l1`` are raw per-core L1 access; ``capture_kernel_config`` captures
a core's relocatable kernel-config block and opaque launch configuration. All take the MeshDevice first.
"""

from ttnn._ttnn.multi_device.experimental import (
    capture_kernel_config,
    read_core_l1,
    set_configure_only,
    write_core_l1,
)

__all__ = ["capture_kernel_config", "read_core_l1", "set_configure_only", "write_core_l1"]
