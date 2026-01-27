# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.parallel - Parallel execution of multiple operations on disjoint core ranges.

This module provides utilities for running multiple operations in parallel within
a single fused program dispatch. Each operation runs on a disjoint set of cores,
enabling efficient hardware utilization without kernel launch overhead.

Example:
    >>> import ttnn
    >>>
    >>> # Define disjoint core ranges
    >>> cores_a = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (3, 3))])
    >>> cores_b = ttnn.CoreRangeSet([ttnn.CoreRange((4, 0), (7, 3))])
    >>>
    >>> # Create branches - each branch is an operation bound to a core range
    >>> branch_a = ttnn.parallel.branch(ttnn.rms_norm, input_a, cores_a, epsilon=1e-5, weight=w_a)
    >>> branch_b = ttnn.parallel.branch(ttnn.rms_norm, input_b, cores_b, epsilon=1e-5, weight=w_b)
    >>>
    >>> # Execute in parallel
    >>> results = ttnn.parallel([branch_a, branch_b])
    >>> output_a = results[0][0]
    >>> output_b = results[1][0]
"""

from typing import Any, List
import ttnn._ttnn.operations.experimental as _experimental


def branch(operation, *args, cores, **kwargs):
    """
    Create a branch descriptor for parallel execution.

    A branch binds an operation to a specific set of cores. Multiple branches
    can be executed in parallel using ttnn.parallel().

    Args:
        operation: A ttnn operation (e.g., ttnn.rms_norm, ttnn.matmul).
            The operation must support parallel execution (have a branch() method).
        *args: Positional arguments to pass to the operation.
        cores (ttnn.CoreRangeSet): The cores this branch should execute on.
            Must be disjoint from other branches in the same parallel call.
        **kwargs: Keyword arguments to pass to the operation.

    Returns:
        BranchDescriptor: A branch descriptor for use with ttnn.parallel().

    Raises:
        AttributeError: If the operation does not support parallel execution.
        TypeError: If cores is not provided as a keyword argument.

    Example:
        >>> cores = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (3, 3))])
        >>> branch = ttnn.parallel.branch(ttnn.rms_norm, input_tensor, cores, epsilon=1e-5)
    """
    # Get the underlying C++ operation
    if hasattr(operation, "function"):
        # FastOperation wrapper - get the C++ registered operation
        cpp_op = operation.function
        op_name = operation.python_fully_qualified_name
    else:
        # Already the C++ operation
        cpp_op = operation
        op_name = getattr(operation, "python_fully_qualified_name", str(operation))

    # Check if the operation supports parallel execution
    if not hasattr(cpp_op, "branch"):
        raise AttributeError(
            f"{op_name} does not support parallel execution. "
            "The operation must implement a branch() method to be used with ttnn.parallel.branch()."
        )

    # Call the operation's branch method
    return cpp_op.branch(*args, cores, **kwargs)


def __call__(branches: List[Any]):
    """
    Execute multiple branches in parallel as a single fused program.

    Args:
        branches (list[BranchDescriptor]): List of branch descriptors created
            via ttnn.parallel.branch().

    Returns:
        list[list[Tensor]]: Nested list where results[i] contains the output
            tensors from the i-th branch.

    Example:
        >>> results = ttnn.parallel([branch_a, branch_b])
        >>> output_a = results[0][0]
        >>> output_b = results[1][0]
    """
    return _experimental.parallel(branches)


# Make the module callable so ttnn.parallel([...]) works
class _ParallelModule:
    """Module wrapper that allows ttnn.parallel to be both callable and have attributes."""

    def __init__(self):
        self.branch = branch
        self.__doc__ = __doc__

    def __call__(self, branches: List[Any]):
        """Execute branches in parallel."""
        return _experimental.parallel(branches)

    def __repr__(self):
        return "<module 'ttnn.parallel'>"


# Replace this module with the callable wrapper
import sys

sys.modules[__name__] = _ParallelModule()
