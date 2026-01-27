# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.experimental module for experimental features.

This module provides experimental APIs that may change in future releases.
"""

from typing import List

import ttnn

# Import ProgramBranch for type hints
from ttnn.experimental.programs import ProgramBranch


def launch_composite(branches: List[ProgramBranch]) -> List["ttnn.Tensor"]:
    """
    Launch a composite operation from multiple branches.

    Merges the program descriptors from each branch and executes them as a single
    program. Each branch operates on non-overlapping cores.

    The device program cache (enabled via device.enable_program_cache()) uses
    GenericOpDeviceOperation::compute_program_hash to cache the compiled program,
    so subsequent calls with structurally identical descriptors reuse the compiled
    program automatically.

    Args:
        branches: A list of ProgramBranch objects created by programs.rms_norm(),
            programs.layer_norm(), etc.

    Returns:
        A list of output tensors, one per branch, in the same order as the input branches.

    Raises:
        ValueError: If branches is empty.
        RuntimeError: If any branches have overlapping core ranges.

    Example:
        >>> left = ttnn.experimental.programs.rms_norm(input1, weight=w1)
        >>> right = ttnn.experimental.programs.rms_norm(input2, weight=w2)
        >>> for _ in range(100):
        ...     left_out, right_out = ttnn.experimental.launch_composite([left, right])
        >>> assert left_out is left.output
    """
    if not branches:
        raise ValueError("branches cannot be empty")

    # Extract program descriptors
    descriptors = [b.descriptor for b in branches]

    # Merge descriptors (or use single descriptor directly)
    if len(descriptors) == 1:
        merged = descriptors[0]
    else:
        # Merge all descriptors - validates no overlapping core ranges
        merged = ttnn.ProgramDescriptor.merge_descriptors(descriptors)

    # Build io_tensors list
    io_tensors = [t for b in branches for t in b.io_tensors]

    # Execute the merged program
    # Device program cache (via compute_program_hash) handles compiled program caching
    ttnn.generic_op(io_tensors, merged)

    # Return output tensors from each branch
    return [b.output for b in branches]


# Import submodules
from ttnn.experimental import programs

__all__ = ["launch_composite", "programs"]
