// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_norm_ttnn_nanobind.hpp"

#include <cstdint>
#include <exception>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "rms_norm_ttnn.hpp"
#include "device/rms_norm_ttnn_device_operation_types.hpp"
#include "device/rms_norm_ttnn_program_factory.hpp"

namespace ttnn::operations::bringup::rms_norm_ttnn {

// What the binding hands back: the INPUT tensor object itself under `inplace` (the caller reads
// their own tensor back, so identity is the contract), else the new output.  Under
// `return_residual_sum` that output is the first member of a tuple whose second is t = x + r.
struct RmsNormPyResult {
    const ttnn::Tensor* alias = nullptr;
    std::optional<ttnn::Tensor> value;
    std::optional<ttnn::Tensor> residual_sum;
};

}  // namespace ttnn::operations::bringup::rms_norm_ttnn

namespace nanobind::detail {

// `program_config` is read FIELD BY FIELD off whatever object is passed -- the op's own dataclasses
// (ttnn.bringup.rms_norm_ttnn.RMSNorm*ProgramConfig), ttnn.LayerNorm*ProgramConfig, or any stand-in
// with the same fields -- exactly as rms_norm_ttnn.py's resolve_program_config() does.
template <>
struct type_caster<ttnn::operations::bringup::rms_norm_ttnn::ProgramConfigArg> {
    NB_TYPE_CASTER(ttnn::operations::bringup::rms_norm_ttnn::ProgramConfigArg, const_name("RMSNormProgramConfig"))

    static bool truthy(const object& o) {
        const int r = PyObject_IsTrue(o.ptr());
        if (r < 0) {
            throw python_error();
        }
        return r == 1;
    }
    static int64_t as_int(const object& o) { return cast<int64_t>(int_(o)); }

    bool from_python(handle src, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept {
        if (src.is_none()) {
            return false;
        }
        try {
            value = {};
            value.use_welford = truthy(getattr(src, "use_welford", bool_(false)));
            value.sharded_variant = hasattr(src, "compute_with_storage_grid_size");
            if (value.sharded_variant) {
                object grid = getattr(src, "compute_with_storage_grid_size", none());
                if (!grid.is_none()) {
                    if (hasattr(grid, "x")) {
                        value.grid = std::make_pair(as_int(grid.attr("x")), as_int(grid.attr("y")));
                    } else {
                        value.grid = std::make_pair(as_int(grid[0]), as_int(grid[1]));
                    }
                }
                if (hasattr(src, "block_h")) {
                    value.block_h = as_int(src.attr("block_h"));
                }
                if (hasattr(src, "block_w")) {
                    value.block_w = as_int(src.attr("block_w"));
                }
                object sb = getattr(src, "subblock_w", int_(0));
                value.subblock_w = truthy(sb) ? as_int(sb) : 0;
                value.inplace = truthy(getattr(src, "inplace", bool_(false)));
            }
            return true;
        } catch (...) {
            PyErr_Clear();
            return false;
        }
    }

    static handle from_cpp(
        const ttnn::operations::bringup::rms_norm_ttnn::ProgramConfigArg&, rv_policy, cleanup_list*) noexcept {
        return none().release();
    }
};

template <>
struct type_caster<ttnn::operations::bringup::rms_norm_ttnn::RmsNormPyResult> {
    NB_TYPE_CASTER(ttnn::operations::bringup::rms_norm_ttnn::RmsNormPyResult, const_name("ttnn.Tensor"))

    bool from_python(handle, uint8_t, cleanup_list*) noexcept { return false; }

    static handle output_of(
        const ttnn::operations::bringup::rms_norm_ttnn::RmsNormPyResult& v, cleanup_list* cleanup) noexcept {
        if (v.alias != nullptr) {
            // rv_policy::reference looks the pointer up among the live instances first, so this returns the
            // caller's own Python object (the input) rather than a new wrapper.
            return nb_type_put(
                &typeid(ttnn::Tensor), const_cast<ttnn::Tensor*>(v.alias), rv_policy::reference, cleanup, nullptr);
        }
        return make_caster<ttnn::Tensor>::from_cpp(*v.value, rv_policy::copy, cleanup);
    }

    static handle from_cpp(
        const ttnn::operations::bringup::rms_norm_ttnn::RmsNormPyResult& v, rv_policy, cleanup_list* cleanup) noexcept {
        if (!v.residual_sum.has_value()) {
            return output_of(v, cleanup);
        }
        // return_residual_sum: (output, residual_sum).
        object y = steal(output_of(v, cleanup));
        if (!y.is_valid()) {
            return handle();
        }
        object t = steal(make_caster<ttnn::Tensor>::from_cpp(*v.residual_sum, rv_policy::copy, cleanup));
        if (!t.is_valid()) {
            return handle();
        }
        return PyTuple_Pack(2, y.ptr(), t.ptr());
    }
};

}  // namespace nanobind::detail

namespace ttnn::operations::bringup::rms_norm_ttnn::detail {

namespace {

RmsNormPyResult rms_norm_py(
    const ttnn::Tensor& input_tensor,
    double epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ProgramConfigArg>& program_config,
    const std::optional<ComputeConfigArg>& compute_kernel_config,
    bool return_residual_sum,
    const std::optional<ttnn::MemoryConfig>& residual_sum_memory_config) {
    auto r = rms_norm_full(
        input_tensor,
        epsilon,
        weight,
        bias,
        residual_input_tensor,
        memory_config,
        program_config,
        compute_kernel_config,
        return_residual_sum,
        residual_sum_memory_config);
    if (r.inplace) {
        return RmsNormPyResult{
            .alias = &input_tensor, .value = std::nullopt, .residual_sum = std::move(r.residual_sum)};
    }
    return RmsNormPyResult{.alias = nullptr, .value = std::move(r.output), .residual_sum = std::move(r.residual_sum)};
}

tt::tt_metal::ProgramDescriptor program_descriptor_py(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    const std::optional<ttnn::Tensor>& weight,
    const std::optional<ttnn::Tensor>& bias,
    const std::optional<ttnn::Tensor>& residual,
    double epsilon,
    const std::optional<ComputeConfigArg>& compute_kernel_config,
    uint32_t subblock_w,
    const std::optional<ttnn::Tensor>& residual_sum) {
    return create_program_descriptor(
        input_tensor,
        output_tensor,
        weight,
        bias,
        residual,
        epsilon,
        normalize_compute_kernel_config(compute_kernel_config),
        subblock_w,
        residual_sum);
}

}  // namespace

void bind_rms_norm_ttnn(nb::module_& mod) {
    // The refusals keep the Python op's exception types (see rms_norm_ttnn.cpp).
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void*) {
            try {
                std::rethrow_exception(p);
            } catch (const UnsupportedAxisError& e) {
                PyObject* cls = PyExc_NotImplementedError;
                nb::object contract;
                try {
                    contract = nb::module_::import_("ttnn.operations._op_contract").attr("UnsupportedAxisValue");
                    cls = contract.ptr();
                } catch (...) {
                    PyErr_Clear();
                }
                PyErr_SetString(cls, e.what());
            } catch (const NotImplementedErrorCpp& e) {
                PyErr_SetString(PyExc_NotImplementedError, e.what());
            }
        },
        nullptr);

    ttnn::bind_function<"rms_norm", "ttnn.bringup.">(
        mod,
        R"doc(
        RMSNorm over the last dimension -- the AI-generated, perf-optimized drop-in for ``ttnn.rms_norm``
        (ttnn/ttnn/bringup/rms_norm_ttnn).  C++ host side; the device kernels are the fork's.

            t = input_tensor + residual_input_tensor                (optional)
            y = t * rsqrt(mean(t^2 over the last dim) + epsilon)
            y = y * weight                                          (optional, per-channel)
            y = y + bias                                            (optional, per-channel)

        Args:
            input_tensor (ttnn.Tensor): on-device, any rank 0..5, TILE or ROW_MAJOR, INTERLEAVED or any *_SHARDED.

        Keyword Args:
            epsilon (float): added to the mean square. Defaults to 1e-12. 0.0 is accepted.
            weight (ttnn.Tensor, optional): per-channel scale, flat (1, 1, 1, Wg >= W) or blocked (Wt, 32) ROW_MAJOR.
            bias (ttnn.Tensor, optional): per-channel shift after the scale; same layout as weight when both given.
            residual_input_tensor (ttnn.Tensor, optional): added before the statistics; matches the input exactly.
            memory_config (ttnn.MemoryConfig, optional): output placement. Defaults to the input's.
            program_config (optional): ttnn.bringup.rms_norm_ttnn.RMSNormDefaultProgramConfig or
                RMSNormShardedMultiCoreProgramConfig (or any object with their fields); ``subblock_w`` and
                ``inplace`` are honoured, ``inplace`` returns the input tensor object itself.
            compute_kernel_config (optional): ttnn.ComputeConfigDescriptor or a device compute-kernel config.
                Defaults to HiFi4 / approx / fp32_dest_acc_en=False.
            return_residual_sum (bool): also return t = input_tensor + residual_input_tensor, the sum the
                statistics are taken over, at the input's dtype and layout -- the next residual add's input in a
                transformer's residual stream, so no separate ttnn.add is needed. Needs residual_input_tensor and
                a TILE input. Defaults to False.
            residual_sum_memory_config (ttnn.MemoryConfig, optional): t's placement (any TILE placement,
                interleaved or sharded). Defaults to the input's. Only with return_residual_sum=True.

        Returns:
            ttnn.Tensor: the normalized tensor (the input tensor itself under ``inplace``); with
            ``return_residual_sum=True`` a tuple ``(normalized, residual_sum)``.
        )doc",
        &rms_norm_py,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("epsilon") = 1e-12,
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("residual_input_tensor") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("return_residual_sum") = false,
        nb::arg("residual_sum_memory_config") = nb::none());

    // For the parity test: the C++ builder's ProgramDescriptor, without dispatching.
    mod.def(
        "_rms_norm_ttnn_program_descriptor",
        &program_descriptor_py,
        nb::arg("input_tensor"),
        nb::arg("output_tensor"),
        nb::kw_only(),
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("residual") = nb::none(),
        nb::arg("epsilon") = 1e-12,
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("subblock_w") = 0,
        nb::arg("residual_sum") = nb::none(),
        R"doc(Build ttnn.bringup.rms_norm's ProgramDescriptor with the C++ builder (no dispatch). Test-only.)doc");
}

}  // namespace ttnn::operations::bringup::rms_norm_ttnn::detail
