# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Install measured auto-configuration over the public ``ttnn.matmul`` / ``ttnn.linear`` entrypoints.

The selector, tuning, caching, and dispatch logic all live in
:mod:`ttnn._experimental.auto_config.matmul`. This module is the thin
integration layer that rebinds the public entrypoints onto those wrappers.

It is invoked once from ``ttnn.operations.matmul`` (via
:func:`install_public_wrappers`) after the underlying C++ operations and their
golden functions have been registered, so the wrappers can reuse the existing
golden functions rather than redefining them.
"""

import inspect

import ttnn
from ttnn.decorators import get_golden_function

# Kept as a self-contained reStructuredText ``.. note::`` directive so it appends cleanly to the
# base op docstrings without forming a definition list or colliding with their existing
# ``Keyword Args:`` section (which would break the Sphinx docs build under ``-W``).
_AUTO_CONFIG_DOC_SUFFIX = """.. note::
   This entrypoint also accepts an ``auto_config`` keyword argument (``bool``, default ``True``)
   that enables measured auto-selection and persistent caching of matmul recipes. When explicit
   low-level placement or program-configuration arguments (``program_config``, ``core_grid``,
   ``compute_kernel_config`` or ``output_tile``) are supplied, auto-configuration is bypassed and
   the requested configuration is used directly.
"""


def _compose_doc(base_doc: str | None) -> str:
    # ``inspect.cleandoc`` strips the base docstring's common leading indentation (the C++ op
    # docstrings are indented) so the appended column-0 suffix shares a consistent indent and the
    # combined text dedents cleanly for Sphinx.
    base = inspect.cleandoc(base_doc or "")
    return f"{base}\n\n{_AUTO_CONFIG_DOC_SUFFIX}"


# Populated by ``install_public_wrappers`` with the original C++ operations so
# the wrappers can dispatch to the kernel-backed base op.
_CPP_MATMUL = None
_CPP_LINEAR = None
_installed = False


def _matmul_wrapper_impl(
    input_tensor_a,
    input_tensor_b,
    *,
    transpose_a=False,
    transpose_b=False,
    memory_config=None,
    dtype=None,
    program_config=None,
    activation=None,
    compute_kernel_config=None,
    core_grid=None,
    output_tile=None,
    optional_output_tensor=None,
    global_cb=None,
    sub_device_id=None,
    auto_config=True,
    queue_id=None,
    cq_id=None,
):
    from ttnn._experimental.auto_config.matmul import dispatch_matmul

    dispatch_kwargs = {}
    if queue_id is not None:
        dispatch_kwargs["queue_id"] = queue_id
    elif cq_id is not None:
        dispatch_kwargs["cq_id"] = cq_id

    return dispatch_matmul(
        base_operation=_CPP_MATMUL,
        input_tensor_a=input_tensor_a,
        input_tensor_b=input_tensor_b,
        bias=None,
        is_linear=False,
        auto_config=auto_config,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        memory_config=memory_config,
        dtype=dtype,
        program_config=program_config,
        activation=activation,
        compute_kernel_config=compute_kernel_config,
        core_grid=core_grid,
        output_tile=output_tile,
        optional_output_tensor=optional_output_tensor,
        global_cb=global_cb,
        sub_device_id=sub_device_id,
        **dispatch_kwargs,
    )


def _linear_wrapper_impl(
    input_tensor_a,
    input_tensor_b,
    *,
    bias=None,
    transpose_a=False,
    transpose_b=False,
    memory_config=None,
    dtype=None,
    program_config=None,
    activation=None,
    compute_kernel_config=None,
    core_grid=None,
    output_tile=None,
    optional_output_tensor=None,
    global_cb=None,
    sub_device_id=None,
    auto_config=True,
    queue_id=None,
    cq_id=None,
):
    from ttnn._experimental.auto_config.matmul import dispatch_matmul

    dispatch_kwargs = {}
    if queue_id is not None:
        dispatch_kwargs["queue_id"] = queue_id
    elif cq_id is not None:
        dispatch_kwargs["cq_id"] = cq_id

    return dispatch_matmul(
        base_operation=_CPP_LINEAR,
        input_tensor_a=input_tensor_a,
        input_tensor_b=input_tensor_b,
        bias=bias,
        is_linear=True,
        auto_config=auto_config,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        memory_config=memory_config,
        dtype=dtype,
        program_config=program_config,
        activation=activation,
        compute_kernel_config=compute_kernel_config,
        core_grid=core_grid,
        output_tile=output_tile,
        optional_output_tensor=optional_output_tensor,
        global_cb=global_cb,
        sub_device_id=sub_device_id,
        **dispatch_kwargs,
    )


def _install_passthrough_wrapper(function, *, doc, golden_function):
    function.__doc__ = doc
    ttnn.attach_golden_function(function, golden_function=golden_function)
    return function


def install_public_wrappers():
    """Rebind ``ttnn.matmul`` / ``ttnn.linear`` onto the auto-config wrappers.

    Idempotent: safe to call more than once. Must be called after the C++
    matmul/linear operations and their golden functions have been registered.
    """
    global _CPP_MATMUL, _CPP_LINEAR, _installed
    if _installed:
        return

    _CPP_MATMUL = ttnn.matmul
    _CPP_LINEAR = ttnn.linear

    # Reuse the golden functions already attached to the C++ operations so the
    # wrappers keep identical reference semantics without redefining them here.
    matmul_golden = get_golden_function(_CPP_MATMUL)
    linear_golden = get_golden_function(_CPP_LINEAR)

    matmul_doc = _compose_doc(_CPP_MATMUL.__doc__)
    linear_doc = _compose_doc(_CPP_LINEAR.__doc__)

    # Install the public entrypoints as plain passthrough functions rather than registering them
    # as new TTNN operations. This leaves the global operation registry completely untouched (no
    # discard / re-register of the C++ ops), so importing ttnn has no global side effect on the
    # rest of the op set, and the kernel-backed C++ op remains the tracked operation. The public
    # matmul/linear API, auto_config support, host RHS/bias staging, queue-id passthrough, and
    # golden functions are all preserved. (This is what the slow-dispatch path already did; using
    # it unconditionally also keeps Tracy slow-dispatch profiling correct.)
    ttnn.matmul = _install_passthrough_wrapper(
        _matmul_wrapper_impl,
        doc=matmul_doc,
        golden_function=matmul_golden,
    )
    ttnn.linear = _install_passthrough_wrapper(
        _linear_wrapper_impl,
        doc=linear_doc,
        golden_function=linear_golden,
    )

    _installed = True
