// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rgb_to_yuv_nanobind.hpp"

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
#include "rgb_to_yuv.hpp"
#include "device/rgb_to_yuv_device_op_types.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace ttnn::experimental::detail {

void bind_rgb_to_yuv(nb::module_& mod) {
    // Standard conversion selectors.
    nb::enum_<YUVColorSpace>(mod, "YUVColorSpace")
        .value("BT601", YUVColorSpace::BT601)
        .value("BT709", YUVColorSpace::BT709)
        .value("BT2020", YUVColorSpace::BT2020);
    nb::enum_<RGBRange>(mod, "RGBRange")
        .value("MinusOneToOne", RGBRange::MinusOneToOne)
        .value("ZeroToOne", RGBRange::ZeroToOne);
    nb::enum_<YUVRange>(mod, "YUVRange").value("Full", YUVRange::Full).value("Limited", YUVRange::Limited);
    nb::enum_<prim::YUVFormat>(mod, "YUVFormat").value("YUV420Planar", prim::YUVFormat::YUV420Planar);

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
        "rgb_to_yuv",
        [](const Tensor& input,
           prim::YUVFormat format,
           YUVColorSpace color_space,
           RGBRange input_range,
           YUVRange output_range,
           const std::optional<prim::YUVCoefficients>& coefficients,
           const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
            return ttnn::experimental::rgb_to_yuv(
                input, format, color_space, input_range, output_range, coefficients, memory_config);
        },
        "input"_a,
        nb::kw_only(),
        "format"_a = prim::YUVFormat::YUV420Planar,
        "color_space"_a = YUVColorSpace::BT601,
        "input_range"_a = RGBRange::MinusOneToOne,
        "output_range"_a = YUVRange::Limited,
        "coefficients"_a = nb::none(),
        "memory_config"_a = nb::none(),
        R"doc(
Convert a CHWT bfloat16 tensor (C=3, RGB) to YUV 4:2:0 uint8.

Returns a tuple of three row-major uint8 tensors:
  (Y, U, V)  with shapes (1,H,W,T), (1,H/2,W/2,T), (1,H/2,W/2,T).

Args:
    input: CHWT bfloat16 tensor on device, C=3.  Must be interleaved.

Keyword Args:
    format: YUVFormat output pixel format. Only YUV420Planar is supported today.
    color_space: YUVColorSpace (BT601/BT709/BT2020). Default BT601.
    input_range: RGBRange of the input samples (MinusOneToOne or ZeroToOne).
        Default MinusOneToOne.
    output_range: YUVRange of the output (Full or Limited). Default Limited.
    coefficients: Optional explicit YUVCoefficients (3x4 [wR, wG, wB, offset]
        rows). If given, overrides color_space/input_range/output_range — for
        power users needing a custom matrix.
    memory_config: Output memory configuration.  Defaults to the input memory
        config.  Must be interleaved -- sharded output is not supported.
)doc");
}

}  // namespace ttnn::experimental::detail
