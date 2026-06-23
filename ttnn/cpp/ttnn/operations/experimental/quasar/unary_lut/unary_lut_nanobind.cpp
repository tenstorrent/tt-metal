// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_lut_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "unary_lut.hpp"
#include "device/unary_lut_device_operation.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_unary_lut(nb::module_& mod) {
    using ttnn::operations::experimental::quasar::unary_lut::LutConfig;

    // Per-activation LUT config: the GENERIC DFB eltwise flow's bridge from the fitter
    // coefficient CSVs to the baked compute kernel. eval_method 0 = POLY_CASCADE,
    // 1 = RATIONAL. `data` = boundaries (num_segments+1) followed by per-segment
    // coefficients (POLY: poly_degree+1 each; RATIONAL: (num_degree+1)+(den_degree+1)).
    nb::class_<LutConfig>(mod, "LutConfig")
        .def(
            // Default-construct, then set fields via def_rw / keyword args below. RR
            // fields are set through def_rw (kept off the positional ctor to preserve
            // the original 6-arg signature used by the no-RR driver path).
            "__init__",
            [](LutConfig* self,
               uint32_t eval_method,
               uint32_t poly_degree,
               uint32_t num_segments,
               uint32_t num_degree,
               uint32_t den_degree,
               std::vector<float> data,
               uint32_t rr_method,
               float rr_log_ln2,
               float rr_exp_mult,
               float rr_exp_const,
               float rr_scale0,
               float rr_scale1,
               float rr_scale2,
               float rr_exp2_mult,
               uint32_t rr_compose,
               float rr_log2_scale,
               uint32_t rr_log2_basis_mminus1,
               float rr_input_offset,
               uint32_t rr_pow_n,
               uint32_t rr_pow_recip) {
                new (self) LutConfig{eval_method,
                                     poly_degree,
                                     num_segments,
                                     num_degree,
                                     den_degree,
                                     std::move(data),
                                     rr_method,
                                     rr_log_ln2,
                                     rr_exp_mult,
                                     rr_exp_const,
                                     rr_scale0,
                                     rr_scale1,
                                     rr_scale2,
                                     rr_exp2_mult,
                                     rr_compose,
                                     rr_log2_scale,
                                     rr_log2_basis_mminus1,
                                     rr_input_offset,
                                     rr_pow_n,
                                     rr_pow_recip};
            },
            nb::arg("eval_method") = 0,
            nb::arg("poly_degree") = 2,
            nb::arg("num_segments") = 4,
            nb::arg("num_degree") = 0,
            nb::arg("den_degree") = 0,
            nb::arg("data") = std::vector<float>{},
            nb::arg("rr_method") = 0,
            nb::arg("rr_log_ln2") = 1.0f,
            nb::arg("rr_exp_mult") = 1.4426950408889634f,
            nb::arg("rr_exp_const") = 0.6931471805599453f,
            nb::arg("rr_scale0") = 1.0f,
            nb::arg("rr_scale1") = 1.0f,
            nb::arg("rr_scale2") = 1.0f,
            nb::arg("rr_exp2_mult") = 1.0f,
            nb::arg("rr_compose") = 0,
            nb::arg("rr_log2_scale") = 1.0f,
            nb::arg("rr_log2_basis_mminus1") = 0,
            nb::arg("rr_input_offset") = 0.0f,
            nb::arg("rr_pow_n") = 2,
            nb::arg("rr_pow_recip") = 0)
        .def_rw("eval_method", &LutConfig::eval_method)
        .def_rw("poly_degree", &LutConfig::poly_degree)
        .def_rw("num_segments", &LutConfig::num_segments)
        .def_rw("num_degree", &LutConfig::num_degree)
        .def_rw("den_degree", &LutConfig::den_degree)
        .def_rw("data", &LutConfig::data)
        .def_rw("rr_method", &LutConfig::rr_method)
        .def_rw("rr_log_ln2", &LutConfig::rr_log_ln2)
        .def_rw("rr_exp_mult", &LutConfig::rr_exp_mult)
        .def_rw("rr_exp_const", &LutConfig::rr_exp_const)
        .def_rw("rr_scale0", &LutConfig::rr_scale0)
        .def_rw("rr_scale1", &LutConfig::rr_scale1)
        .def_rw("rr_scale2", &LutConfig::rr_scale2)
        .def_rw("rr_exp2_mult", &LutConfig::rr_exp2_mult)
        .def_rw("rr_compose", &LutConfig::rr_compose)
        .def_rw("rr_log2_scale", &LutConfig::rr_log2_scale)
        .def_rw("rr_log2_basis_mminus1", &LutConfig::rr_log2_basis_mminus1)
        .def_rw("rr_input_offset", &LutConfig::rr_input_offset)
        .def_rw("rr_pow_n", &LutConfig::rr_pow_n)
        .def_rw("rr_pow_recip", &LutConfig::rr_pow_recip);

    const auto* doc = R"doc(
            Applies an embedded piecewise-LUT activation to a fully height/block-sharded
            bf16 L1 input, through the Metal 2.0 / DataflowBuffer (DFB) path.

            With ``lut_config`` set, the op bakes that per-activation LUT (boundaries +
            per-segment POLY or RATIONAL coefficients, parsed from a fitter coefficient
            CSV) into the compute kernel at JIT time — the GENERIC DFB eltwise flow over
            (activation, eval_method). Without it, the kernel's compile-time default (the
            proven deg-2 / 4-segment sigmoid, no range reduction) is used.

            Args:
                input_tensor (ttnn.Tensor): bf16, TILE layout, height/block-sharded L1.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): output memory config.
                output (ttnn.Tensor, optional): preallocated output tensor.
                sub_device_id (ttnn.SubDeviceId, optional): sub-device to run on.
                lut_config (ttnn.experimental.quasar.LutConfig, optional): per-activation
                    LUT to bake into the kernel.

            Returns:
                ttnn.Tensor: the activation applied element-wise.
        )doc";

    ttnn::bind_function<"unary_lut", "ttnn.experimental.quasar.">(
        mod,
        doc,
        &ttnn::operations::experimental::quasar::unary_lut::unary_lut,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output") = nb::none(),
        nb::arg("sub_device_id") = nb::none(),
        nb::arg("lut_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::quasar::detail
