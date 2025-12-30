# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unified Kernel Python Wrapper

Provides an MPI-like API for writing kernels that automatically handles:
- 3-way kernel split (Reader/Compute/Writer) for local-only operations
- Role-aware multicast/unicast with automatic semaphore management
- Cross-core synchronization primitives
"""

from .builder import UnifiedKernelBuilder
from .primitives import Role, McastGroup, BufferMode

__all__ = ["UnifiedKernelBuilder", "Role", "McastGroup", "BufferMode"]
