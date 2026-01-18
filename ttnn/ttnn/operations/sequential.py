# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.sequential - Execute operations in sequence.

This module provides utilities for running operations in sequence. Currently
execution happens at the Python level, with C++ infrastructure reserved for
future CB chaining optimizations.

Example:
    >>> import ttnn
    >>>
    >>> # Execute operations in sequence
    >>> results = ttnn.sequential([
    ...     (ttnn.rms_norm, input1, {"epsilon": 1e-5}),
    ...     (ttnn.layer_norm, input2, {"epsilon": 1e-6, "weight": gamma}),
    ... ])
    >>> rms_output = results[0]
    >>> ln_output = results[1]
"""

from typing import Any, List, Tuple, Dict, Union


def _execute_steps(steps: List[Tuple]) -> List[Any]:
    """
    Execute a sequence of operations one after another.

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


# Make the module callable so ttnn.sequential([...]) works
class _SequentialModule:
    """Module wrapper that allows ttnn.sequential to be callable."""

    def __init__(self):
        self.__doc__ = __doc__

    def __call__(self, steps: List[Tuple]) -> List[Any]:
        """
        Execute a sequence of operations one after another.

        Args:
            steps: List of tuples, where each tuple is either:
                   - (operation, arg1, arg2, ...) - positional args only
                   - (operation, arg1, arg2, ..., kwargs_dict) - with kwargs as last element

        Returns:
            List of results, one per operation.

        Example:
            >>> results = ttnn.sequential([
            ...     (ttnn.rms_norm, input1, {"epsilon": 1e-5}),
            ...     (ttnn.layer_norm, input2),
            ... ])
        """
        return _execute_steps(steps)

    def __repr__(self):
        return "<module 'ttnn.sequential'>"


# Replace this module with the callable wrapper
import sys

sys.modules[__name__] = _SequentialModule()
