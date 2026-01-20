# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.sequential - Execute operations in sequence.

This module provides utilities for running operations in sequence. It supports:
1. Simple Python-level sequential execution (for operations that don't support add_to)
2. C++ device-level sequential execution (for operations that support add_to)
3. Creating sequential branches for use with ttnn.parallel

Each step specifies its own core range when created via the operation's .step() method.

Example (Python-level):
    >>> import ttnn
    >>>
    >>> # Execute operations in sequence (Python dispatch)
    >>> results = ttnn.sequential([
    ...     (ttnn.rms_norm, input1, {"epsilon": 1e-5}),
    ...     (ttnn.layer_norm, input2, {"epsilon": 1e-6, "weight": gamma}),
    ... ])
    >>> rms_output = results[0]
    >>> ln_output = results[1]

Example (C++ device-level with cores per step):
    >>> import ttnn
    >>>
    >>> # Define cores for the steps
    >>> cores = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (3, 3))])
    >>>
    >>> # Create steps using .step() methods - each step specifies its cores
    >>> step1 = ttnn.rms_norm.step(input1, cores, epsilon=1e-5)
    >>> step2 = ttnn.layer_norm.step(input2, cores, epsilon=1e-6)
    >>>
    >>> # Execute as fused program
    >>> results = ttnn.sequential([step1, step2])

Example (As a branch in parallel):
    >>> # Create steps with their cores
    >>> cores_seq = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (3, 3))])
    >>> step1 = ttnn.rms_norm.step(input1, cores_seq, epsilon=1e-5)
    >>> step2 = ttnn.layer_norm.step(input2, cores_seq, epsilon=1e-6)
    >>>
    >>> # Create a sequential branch
    >>> seq_branch = ttnn.sequential.branch([step1, step2])
    >>>
    >>> # Use with parallel - dispatched as single fused program
    >>> cores_other = ttnn.CoreRangeSet([ttnn.CoreRange((4, 0), (7, 3))])
    >>> other_branch = ttnn.rms_norm.branch(input_b, cores_other, epsilon=1e-5)
    >>> results = ttnn.parallel([seq_branch, other_branch])
"""

from typing import Any, List, Tuple, Optional, Union
import ttnn._ttnn.operations.experimental as _experimental


def _is_step_descriptor(obj) -> bool:
    """Check if an object is a StepDescriptor."""
    return hasattr(obj, "__class__") and "StepDescriptor" in obj.__class__.__name__


def _execute_python_steps(steps: List[Tuple]) -> List[Any]:
    """
    Execute a sequence of operations one after another (Python dispatch).

    Each operation is executed independently with its own arguments.
    Results from one operation are NOT automatically fed to the next.

    Args:
        steps: List of tuples, where each tuple is either:
               - (operation, arg1, arg2, ...) - positional args only
               - (operation, arg1, arg2, ..., kwargs_dict) - with keyword args as last element

    Returns:
        List of results, one per operation.
    """
    results = []

    for step in steps:
        if not isinstance(step, (list, tuple)) or len(step) < 2:
            raise ValueError(
                f"Each step must be a tuple of (operation, arg1, ...) or "
                f"(operation, arg1, ..., kwargs_dict). Got: {step}"
            )

        operation = step[0]
        remaining = step[1:]

        # Check if the last element is a dict (kwargs)
        if remaining and isinstance(remaining[-1], dict):
            args = remaining[:-1]
            kwargs = remaining[-1]
        else:
            args = remaining
            kwargs = {}

        # Execute the operation
        result = operation(*args, **kwargs)
        results.append(result)

    return results


def _execute_cpp_steps(steps: List[Any]) -> List[Any]:
    """
    Execute a sequence of step descriptors as a fused program (C++ dispatch).

    Each step carries its own core range (specified when the step was created).

    Args:
        steps: List of StepDescriptor objects created via op.step() methods.

    Returns:
        List of output tensors from the last step.
    """
    # Use the C++ sequential device operation
    return _experimental.sequential(steps)


# Make the module callable so ttnn.sequential([...]) works
class _SequentialModule:
    """Module wrapper that allows ttnn.sequential to be callable and have attributes."""

    def __init__(self):
        self.__doc__ = __doc__

    def __call__(
        self,
        steps: List[Union[Tuple, Any]],
    ) -> List[Any]:
        """
        Execute a sequence of operations.

        If steps are StepDescriptor objects (created via op.step()), executes as a
        fused C++ program. Each step carries its own core range.

        If steps are (op, args...) tuples, executes at Python level.

        Args:
            steps: Either:
                   - List of StepDescriptors: fused C++ execution (each step has its own cores)
                   - List of (operation, arg1, ...) tuples: Python dispatch

        Returns:
            List of results, one per operation (Python) or outputs from last step (C++).

        Example (Python dispatch):
            >>> results = ttnn.sequential([
            ...     (ttnn.rms_norm, input1, {"epsilon": 1e-5}),
            ...     (ttnn.layer_norm, input2),
            ... ])

        Example (C++ fused execution):
            >>> cores = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (3, 3))])
            >>> step1 = ttnn.rms_norm.step(input1, cores, epsilon=1e-5)
            >>> step2 = ttnn.layer_norm.step(input2, cores, epsilon=1e-6)
            >>> results = ttnn.sequential([step1, step2])
        """
        if not steps:
            return []

        # Check if this is a list of StepDescriptors
        if _is_step_descriptor(steps[0]):
            return _execute_cpp_steps(steps)
        else:
            # Python-level dispatch
            return _execute_python_steps(steps)

    def branch(self, steps: List[Any]):
        """
        Create a sequential branch descriptor for use with ttnn.parallel.

        This allows running a sequence of operations as a single branch in a parallel
        execution context. The branch's core range is taken from the first step.
        All steps should use the same core range for proper sequential execution.

        Args:
            steps: List of StepDescriptor objects created via op.step() methods.
                   Each step carries its core range (specified when created).

        Returns:
            BranchDescriptor: A branch descriptor for use with ttnn.parallel.

        Example:
            >>> # Create steps with their cores
            >>> cores_seq = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (3, 3))])
            >>> step1 = ttnn.rms_norm.step(input1, cores_seq, epsilon=1e-5)
            >>> step2 = ttnn.layer_norm.step(input2, cores_seq, epsilon=1e-6)
            >>>
            >>> # Create a sequential branch
            >>> seq_branch = ttnn.sequential.branch([step1, step2])
            >>>
            >>> # Use with parallel - dispatched as single fused program
            >>> cores_other = ttnn.CoreRangeSet([ttnn.CoreRange((4, 0), (7, 3))])
            >>> other_branch = ttnn.rms_norm.branch(input_b, cores_other, epsilon=1e-5)
            >>> results = ttnn.parallel([seq_branch, other_branch])
        """
        return _experimental.sequential_branch(steps)

    def __repr__(self):
        return "<module 'ttnn.sequential'>"


# Replace this module with the callable wrapper
import sys

sys.modules[__name__] = _SequentialModule()
