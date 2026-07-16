// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "yuv_conversion_nanobind.hpp"

#include <array>
#include <optional>
#include <tuple>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ttnn/tensor/tensor.hpp"
#include "yuv_conversion.hpp"
#include "device/yuv_conversion_device_op_types.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace ttnn::experimental::detail {

void bind_yuv_conversion(nb::module_& mod) {
    // Expose the YUVCoefficients struct to Python so callers can construct it.
    nb::class_<prim::YUVCoefficients>(mod, "YUVCoefficients")
        .def(nb::init<>())
        .def(
            "__init__",
            [](prim::YUVCoefficients* self, std::array<float, 4> y, std::array<float, 4> cb, std::array<float, 4> cr) {
                new (self) prim::YUVCoefficients{y, cb, cr};
            },
            "y"_a,
            "cb"_a,
            "cr"_a)
        .def_rw("y", &prim::YUVCoefficients::y)
        .def_rw("cb", &prim::YUVCoefficients::cb)
        .def_rw("cr", &prim::YUVCoefficients::cr);

    // Expose helper coefficient factories.
    mod.def(
        "yuv_bt601_coefficients",
        &ttnn::experimental::yuv_bt601_coefficients,
        "Return BT.601 YUV coefficients for bf16 input in [-1, 1].");
    mod.def(
        "yuv_bt709_coefficients",
        &ttnn::experimental::yuv_bt709_coefficients,
        "Return BT.709 YUV coefficients for bf16 input in [-1, 1].");

    // Main op: CHWT bf16 → (Y, U, V) uint8 tuple.
    mod.def(
        "yuv_conversion",
        [](const Tensor& input,
           const prim::YUVCoefficients& coefficients,
           const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
            return ttnn::experimental::yuv_conversion(input, coefficients, memory_config);
        },
        "input"_a,
        "coefficients"_a,
        nb::kw_only(),
        "memory_config"_a = nb::none(),
        R"doc(
Convert a CHWT bfloat16 tensor (C=3, values in [-1, 1]) to YUV 4:2:0 uint8.

Returns a tuple of three row-major uint8 tensors:
  (Y, U, V)  with shapes (1,H,W,T), (1,H/2,W/2,T), (1,H/2,W/2,T).

Args:
    input: CHWT bfloat16 tensor on device, C=3.  Must be interleaved.
    coefficients: YUVCoefficients with per-channel [wR, wG, wB, offset] rows.

Keyword Args:
    memory_config: Output memory configuration.  Defaults to the input memory
        config.  Must be interleaved -- sharded output is not supported.
)doc");
}

}  // namespace ttnn::experimental::detail
