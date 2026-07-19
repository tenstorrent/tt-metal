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

import os

import ttnn
from ttnn.decorators import REGISTERED_OPERATIONS, get_golden_function

_AUTO_CONFIG_DOC_SUFFIX = """

Additional Keyword Args:
    auto_config (bool, optional): Enables measured auto-selection and persistent caching for matmul recipes.
        Defaults to `True`. If explicit low-level placement/program configuration arguments such as
        `program_config`, `core_grid`, `compute_kernel_config`, or `output_tile` are supplied,
        auto-configuration is bypassed and the requested configuration is used directly.
"""

# Populated by ``install_public_wrappers`` with the original C++ operations so
# the wrappers can dispatch to the kernel-backed base op.
_CPP_MATMUL = None
_CPP_LINEAR = None
_installed = False


def _slow_dispatch_mode_enabled() -> bool:
    return os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") == "1"


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


def _install_slow_dispatch_wrapper(function, *, doc, golden_function):
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

    matmul_doc = f"{_CPP_MATMUL.__doc__}{_AUTO_CONFIG_DOC_SUFFIX}"
    linear_doc = f"{_CPP_LINEAR.__doc__}{_AUTO_CONFIG_DOC_SUFFIX}"

    if _slow_dispatch_mode_enabled():
        # Keep the public matmul/linear API available, but avoid wrapping the
        # call in another registered TTNN operation. Tracy slow-dispatch
        # profiling joins host/device data using the kernel-backed op identity,
        # so the underlying C++ op must remain the top-level tracked operation.
        ttnn.matmul = _install_slow_dispatch_wrapper(
            _matmul_wrapper_impl,
            doc=matmul_doc,
            golden_function=matmul_golden,
        )
        ttnn.linear = _install_slow_dispatch_wrapper(
            _linear_wrapper_impl,
            doc=linear_doc,
            golden_function=linear_golden,
        )
    else:
        REGISTERED_OPERATIONS.operations.discard(_CPP_MATMUL)
        REGISTERED_OPERATIONS.operations.discard(_CPP_LINEAR)
        ttnn.register_python_operation(
            name="ttnn.matmul",
            golden_function=matmul_golden,
            doc=matmul_doc,
        )(_matmul_wrapper_impl)
        ttnn.register_python_operation(
            name="ttnn.linear",
            golden_function=linear_golden,
            doc=linear_doc,
        )(_linear_wrapper_impl)

    _installed = True
