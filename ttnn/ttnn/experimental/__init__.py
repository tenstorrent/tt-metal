# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.experimental module for experimental features.

This module provides experimental APIs that may change in future releases.
"""

from typing import List, Tuple

import ttnn


def launch_composite(
    descriptors_with_outputs: List[Tuple["ttnn.ProgramDescriptor", "ttnn.Tensor"]],
    io_tensors: List["ttnn.Tensor"],
) -> None:
    """
    Launch a composite operation made of multiple program descriptors.

    This function merges multiple program descriptors into a single program and executes it
    using generic_op. Each program descriptor should operate on non-overlapping core ranges.

    Args:
        descriptors_with_outputs: A list of (ProgramDescriptor, output_tensor) tuples.
            These are the return values from programs.rms_norm, programs.layer_norm, etc.
        io_tensors: A list of all tensors referenced by the program descriptors (inputs and outputs).
            These tensors are passed to generic_op for runtime arg resolution.

    Raises:
        RuntimeError: If any program descriptors have overlapping core ranges.
        ValueError: If descriptors_with_outputs is empty.

    Example:
        >>> import ttnn
        >>> # Create tensors
        >>> input1 = ttnn.from_torch(torch_input1, device=device, ...)
        >>> input2 = ttnn.from_torch(torch_input2, device=device, ...)
        >>>
        >>> # Create program descriptors (outputs are created internally)
        >>> desc1, out1 = ttnn.experimental.programs.rms_norm(input1, core_range_1)
        >>> desc2, out2 = ttnn.experimental.programs.rms_norm(input2, core_range_2)
        >>>
        >>> # Launch composite - outputs are written in-place
        >>> ttnn.experimental.launch_composite(
        ...     [(desc1, out1), (desc2, out2)],
        ...     io_tensors=[input1, out1, input2, out2]
        ... )
        >>>
        >>> # Now out1 and out2 contain the results
    """
    if not descriptors_with_outputs:
        raise ValueError("descriptors_with_outputs cannot be empty")

    # Extract just the descriptors
    program_descriptors = [desc for desc, _ in descriptors_with_outputs]

    if len(program_descriptors) == 1:
        # Single descriptor - no need to merge
        merged = program_descriptors[0]
    else:
        # Merge all descriptors - this validates no overlapping core ranges
        merged = ttnn.ProgramDescriptor.merge_descriptors(program_descriptors)

    # Execute the merged program with all io tensors
    ttnn.generic_op(io_tensors, merged)


# Import submodules
from ttnn.experimental import programs

__all__ = ["launch_composite", "programs"]
