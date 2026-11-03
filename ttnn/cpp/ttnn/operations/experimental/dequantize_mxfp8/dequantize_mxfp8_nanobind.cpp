// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dequantize_mxfp8_nanobind.hpp"
#include "dequantize_mxfp8.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental {
void bind_dequantize_mxfp8(nanobind::module_& mod) {
    namespace nb = nanobind;
    ttnn::bind_function<"dequantize_mxfp8", "ttnn.experimental.">(
        mod,
        R"doc(
Decode MXFP8 (OCP MX, E4M3 elements) on Blackhole: output = E4M3 * 2^(E8M0 - 127).

Prepare inputs with ttnn.experimental.mxfp8.pack. Each block of 32 horizontally
adjacent values (one row of one 32x32 output tile) shares one E8M0 scale.

Packed UINT8/UINT16 words contain 1/2 E4M3 bytes, least significant byte first.
Each packed 32x32 tile expands into 1/2 adjacent output tiles; byte k of every
word supplies the same position in output tile k. For UINT8 this is the identity:
the packed tensor holds the E4M3 bytes in their natural positions. Input height
and width must be multiples of 32, without padding; rank 2 to 4 and interleaved
DRAM/L1 are supported. Input and output may each be TILE or ROW_MAJOR.

Scales are UINT8 ROW_MAJOR raw E8M0 codes with shape
[packed_tile_count, 32 * values_per_word]. For each packed tile: order by byte
plane, then the 32 rows of the output tile.

Numerical contract: E4M3 NaN codes (0x7F, 0xFF) and the E8M0 NaN code (0xFF)
are not supported; their outputs are unspecified. Every other result is exact.
Results with magnitude at or above 2^128 become signed infinity; results below
the smallest normal FP32/BF16 magnitude (2^-126) may flush to zero. BF16 and
FLOAT32 outputs represent the same values; FLOAT32 output and UINT8 input
require fp32_dest_acc_en=True.
)doc",
        &dequantize_mxfp8,
        nb::arg("packed").noconvert(),
        nb::arg("scales").noconvert(),
        nb::kw_only(),
        nb::arg("dtype") = DataType::BFLOAT16,
        nb::arg("output_layout") = Layout::TILE,
        nb::arg("fp32_dest_acc_en") = false,
        nb::arg("memory_config") = nb::none());
}
}  // namespace ttnn::operations::experimental
