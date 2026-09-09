// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Nanobind boundary for the native `migrated_run1000_groupnorm` operation.
//
// Everything here is thin conversion plus boundary reads/writes of Python-owned
// state:
//
//   * `num_groups` keeps the source's `isinstance(num_groups, int)` check, whose
//     ValueError message embeds the argument's repr.
//   * `compute_kernel_config` keeps the source's duck-typed `getattr` projection
//     onto the four exposed knobs, so WormholeComputeKernelConfig,
//     GrayskullComputeKernelConfig and duck-typed stand-ins all work.
//   * host refusals are re-raised as the SAME Python exception classes the
//     source raised, including
//     `ttnn.operations._op_contract.UnsupportedAxisValue` / `ExcludedCell`.
//
// `ttnn::bind_function` releases the GIL around the invocation, so every access
// to a Python object below explicitly reacquires it.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <optional>
#include <string>

#include "migrated_run1000_groupnorm.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::migration::generated_migrated_run1000_groupnorm {

namespace nb = nanobind;

namespace {

// Re-raise a host refusal with the Python exception identity the source used.
[[noreturn]] void raise_host_error(const HostError& error) {
    nb::gil_scoped_acquire gil;
    switch (error.kind()) {
        case PyErrKind::ValueError: PyErr_SetString(PyExc_ValueError, error.what()); break;
        case PyErrKind::NotImplementedError: PyErr_SetString(PyExc_NotImplementedError, error.what()); break;
        case PyErrKind::ZeroDivisionError: PyErr_SetString(PyExc_ZeroDivisionError, error.what()); break;
        case PyErrKind::UnsupportedAxisValue:
        case PyErrKind::ExcludedCell: {
            const char* class_name =
                error.kind() == PyErrKind::UnsupportedAxisValue ? "UnsupportedAxisValue" : "ExcludedCell";
            // Both subclass NotImplementedError; the identity is what the eval
            // harness recognizes, so it must be the existing ttnn class.
            nb::object contract = nb::module_::import_("ttnn.operations._op_contract");
            nb::object exception_class = contract.attr(class_name);
            nb::str message(error.what());
            PyErr_SetObject(exception_class.ptr(), message.ptr());
            break;
        }
    }
    throw nb::python_error();
}

[[noreturn]] void raise_value_error(const std::string& message) {
    nb::gil_scoped_acquire gil;
    PyErr_SetString(PyExc_ValueError, message.c_str());
    throw nb::python_error();
}

// `bool(value)` semantics -- truthiness, not a strict bool cast.
bool python_truth(nb::handle value) {
    const int truth = PyObject_IsTrue(value.ptr());
    if (truth == -1) {
        throw nb::python_error();
    }
    return truth == 1;
}

// `_compute_config_fields()` -- project a config object onto the four exposed
// knobs; a missing or None attribute keeps the Phase-0 default for that knob.
ComputeConfigFields read_compute_config(nb::handle compute_kernel_config) {
    ComputeConfigFields fields;  // == _DEFAULT_COMPUTE_CONFIG
    if (!compute_kernel_config.is_valid() || compute_kernel_config.is_none()) {
        return fields;
    }

    nb::gil_scoped_acquire gil;

    nb::object math_fidelity = nb::getattr(compute_kernel_config, "math_fidelity", nb::none());
    if (!math_fidelity.is_none()) {
        fields.math_fidelity = nb::cast<tt::tt_metal::MathFidelity>(math_fidelity);
    }
    nb::object fp32_dest_acc_en = nb::getattr(compute_kernel_config, "fp32_dest_acc_en", nb::none());
    if (!fp32_dest_acc_en.is_none()) {
        fields.fp32_dest_acc_en = python_truth(fp32_dest_acc_en);
    }
    nb::object math_approx_mode = nb::getattr(compute_kernel_config, "math_approx_mode", nb::none());
    if (!math_approx_mode.is_none()) {
        fields.math_approx_mode = python_truth(math_approx_mode);
    }
    nb::object dst_full_sync_en = nb::getattr(compute_kernel_config, "dst_full_sync_en", nb::none());
    if (!dst_full_sync_en.is_none()) {
        fields.dst_full_sync_en = python_truth(dst_full_sync_en);
    }
    return fields;
}

// `_validate_args()`'s `isinstance(num_groups, int)` gate. The message embeds
// the argument's repr, so the check lives at the boundary.
std::int64_t read_num_groups(nb::handle num_groups) {
    bool is_int = false;
    std::string repr;
    std::int64_t value = 0;
    {
        nb::gil_scoped_acquire gil;
        is_int = PyLong_Check(num_groups.ptr()) != 0;
        if (is_int) {
            value = nb::cast<std::int64_t>(num_groups);
        } else {
            repr = nb::cast<std::string>(nb::repr(num_groups));
        }
    }
    if (!is_int) {
        raise_value_error("groupnorm_sc_N_1_HW_C: num_groups must be a positive int, got " + repr);
    }
    return value;
}

// `bind_function` releases the GIL around this body, so the two Python-owned
// arguments are taken by CONST REFERENCE (never by value): a by-value nb::object
// parameter would inc_ref/dec_ref with the GIL released. The casters that own
// them live in nanobind's dispatcher frame, which outlives the release guard.
// Same shape as the target's own `from_buffer_impl`.
Tensor migrated_run1000_groupnorm(
    const Tensor& input_tensor,
    const nb::object& num_groups,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    double eps,
    const nb::object& compute_kernel_config) {
    try {
        const std::int64_t groups = read_num_groups(num_groups);
        const ComputeConfigFields compute = read_compute_config(compute_kernel_config);
        return invoke(input_tensor, groups, gamma, beta, eps, compute);
    } catch (const HostError& error) {
        raise_host_error(error);
    }
}

constexpr auto kDoc = R"doc(
GroupNorm over a channel-last ``(N, 1, H*W, C)`` tensor.

Native migration of the registry-model operation ``groupnorm_sc_N_1_HW_C``
(regime ``cluster_parallel_two_pass``). Non-tile-aligned channels-per-group
(``(C / num_groups) % 32 != 0``) is supported through the *channel cluster*
construction: a cluster is a whole number of groups AND a whole number of
32-channel tiles, so the ``(batch n, cluster k)`` work split is
communication-free. ``HW % 32 != 0`` is handled by a row mask applied to the
trailing HW tile in the moment pass, and ``C % 32 != 0`` collapses the cluster
construction to the degenerate single cluster.

Args:
    input_tensor (ttnn.Tensor): rank-4 ``(N, 1, HW, C)`` input, ``dim[1] == 1``.
    num_groups (int): positive group count; must divide ``C``.

Keyword args:
    gamma (ttnn.Tensor, optional): ``(1, 1, 1, C)`` scale. Defaults to `None`.
    beta (ttnn.Tensor, optional): ``(1, 1, 1, C)`` shift. Defaults to `None`.
    eps (float): numerical-stability epsilon. Defaults to ``1e-5``.
    compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): projected
        onto ``math_fidelity`` / ``fp32_dest_acc_en`` / ``math_approx_mode`` /
        ``dst_full_sync_en``. Unset keeps the HiFi4 + fp32-DEST default.

Returns:
    ttnn.Tensor: the normalized tensor, input dtype, ALWAYS TILE_LAYOUT,
    interleaved DRAM.

Note:
    Supported input dtypes: BFLOAT16, FLOAT32, BFLOAT8_B.
    Supported input layouts: TILE, ROW_MAJOR.
    Supported gamma/beta dtypes: BFLOAT16, FLOAT32, BFLOAT8_B (or absent).
    Supported gamma/beta layouts: TILE, ROW_MAJOR (or absent).

Raises:
    ValueError: on a malformed rank/shape/num_groups argument.
    ttnn.operations._op_contract.UnsupportedAxisValue: when an axis value falls
        outside the operation's declared support set.
    NotImplementedError: when one channel cluster does not fit L1.
)doc";

}  // namespace

void bind_operation(nb::module_& mod) {
    ttnn::bind_function<"migrated_run1000_groupnorm">(
        mod,
        kDoc,
        &migrated_run1000_groupnorm,
        nb::arg("input_tensor"),
        nb::arg("num_groups"),
        nb::kw_only(),
        nb::arg("gamma") = nb::none(),
        nb::arg("beta") = nb::none(),
        nb::arg("eps") = 1e-5,
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::migration::generated_migrated_run1000_groupnorm
