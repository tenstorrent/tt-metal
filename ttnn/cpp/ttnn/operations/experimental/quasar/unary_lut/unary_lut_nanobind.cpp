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
            nb::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, std::vector<float>>(),
            nb::arg("eval_method") = 0,
            nb::arg("poly_degree") = 2,
            nb::arg("num_segments") = 4,
            nb::arg("num_degree") = 0,
            nb::arg("den_degree") = 0,
            nb::arg("data") = std::vector<float>{})
        .def_rw("eval_method", &LutConfig::eval_method)
        .def_rw("poly_degree", &LutConfig::poly_degree)
        .def_rw("num_segments", &LutConfig::num_segments)
        .def_rw("num_degree", &LutConfig::num_degree)
        .def_rw("den_degree", &LutConfig::den_degree)
        .def_rw("data", &LutConfig::data);

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
