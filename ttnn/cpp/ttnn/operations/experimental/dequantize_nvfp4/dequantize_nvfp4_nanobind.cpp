// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dequantize_nvfp4_nanobind.hpp"
#include "dequantize_nvfp4.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental {
void bind_dequantize_nvfp4(nanobind::module_& mod) {
    namespace nb = nanobind;
    nb::enum_<Nvfp4ScaleFormat>(mod, "Nvfp4ScaleFormat")
        .value("E4M3", Nvfp4ScaleFormat::E4M3)
        .value("E2M1", Nvfp4ScaleFormat::E2M1);
    ttnn::bind_function<"dequantize_nvfp4", "ttnn.experimental.">(
        mod,
        R"doc(
Decode pre-permuted E2M1 data on Blackhole: output = E2M1 * block_scale * global_scale.

Packed UINT8/UINT16 words contain 2/4 nibbles, least significant nibble first.
Each packed 32x32 tile expands into 2/4 adjacent output tiles; nibble k of every
word supplies the same position in output tile k. This is a tile-plane packing,
not consecutive-element nibble packing. Input height and width must be multiples
of 32, without padding; rank 2 to 4 and interleaved DRAM/L1 are supported.

Scales are UINT8 ROW_MAJOR with shape [packed_tile_count, 64 * values_per_word].
For each packed tile: order by nibble plane, then face (TL, TR, BL, BR), then
16 face rows. Each byte scales the 16 elements in that face row. E4M3 uses
finite nonnegative E4M3FN codes (0..126); E2M1 uses nonnegative codes (0..7)
in one byte per scale and is a custom variant, rather than standard NVFP4.
The caller must supply scale codes in these ranges. The global scale is a
finite nonnegative dequantization multiplier, not its reciprocal.

The output last dimension is 2/4 times the packed last dimension. Output can be
TILE or ROW_MAJOR and BFLOAT16 or FLOAT32; FLOAT32 requires fp32_dest_acc_en=True.
UINT8 requires fp32_dest_acc_en=True because its integer unpack path uses 32-bit
DEST. Use UINT16 for BF16 DEST. BF16 output rounds to nearest even once after both scale multiplications, in
either DEST mode. FLOAT32 output retains FP32 values. SFPU arithmetic flushes FP32 subnormals.
experimental_fast=True requires UINT16 data, BFLOAT16 output, and BF16 DEST.
Prepare both inputs with nvfp4.pack(..., experimental_fast=True). In this mode
scales are compact BFLOAT16 rows of exactly decoded E4M3/E2M1 scales. Each
32x32 data tile is transposed offline. A compact scale tile is unpacked
and expanded into column scale vectors in DEST. SFPU uses native nearest
rounding (ties away), and an in-DEST transpose restores ordinary output tiles.
The tensor scale is rounded to BF16.
fpu_tensor_scale=True applies it using HiFi4 FPU DEST reuse; False applies it
in the SFPU and avoids the FPU multiply/reconfiguration. Both are experimental
relaxed-precision paths; the original behavior remains the default.

transpose_output=True requires experimental_fast=True and TILE output. It returns
the full last-two-dimension transpose of the decoded tensor, including swapping
the tile grid independently within each batch. Prepare the original data/scales
with the same nvfp4.pack call. This skips the in-DEST transpose and writes tiles
to transposed grid coordinates, without a separate transpose operation. For a
weight W with shape [K,N], the result has shape [N,K] and can be consumed by
linear(X, decoded_W_transposed, transpose_b=True). Use a matmul program config
with native RHS transposition; some program configs materialize a transpose.
This option changes the returned tensor shape and defaults to False.
See the operation README for packing examples, accuracy, and fusion considerations.
)doc",
        &dequantize_nvfp4,
        nb::arg("packed").noconvert(),
        nb::arg("scales").noconvert(),
        nb::kw_only(),
        nb::arg("global_scale") = 1.0f,
        nb::arg("scale_format") = Nvfp4ScaleFormat::E4M3,
        nb::arg("dtype") = DataType::BFLOAT16,
        nb::arg("output_layout") = Layout::TILE,
        nb::arg("fp32_dest_acc_en") = false,
        nb::arg("memory_config") = nb::none(),
        nb::arg("experimental_fast") = false,
        nb::arg("fpu_tensor_scale") = true,
        nb::arg("transpose_output") = false);
}
}  // namespace ttnn::operations::experimental
