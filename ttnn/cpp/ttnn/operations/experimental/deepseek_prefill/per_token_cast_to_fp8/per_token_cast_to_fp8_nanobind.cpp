// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_to_fp8_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "per_token_cast_to_fp8.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8::detail {

void bind_experimental_per_token_cast_to_fp8_operation(nb::module_& mod) {
    ttnn::bind_function<"per_token_cast_to_fp8", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Per-token quantization to FP8_E4M3 with per-128-element scale factors.

            Mirrors DeepEP's per_token_cast_to_fp8 (see deepseek-ai/DeepEP math.py).
            For input shape [..., M, H], produces:
              * an FP8_E4M3 tensor of the same shape, and
              * a FLOAT32 scale tensor of shape [..., M, H/128].

            Per 128-element block of each token: scale = clamp(max(|x|), 1e-4) / 448 and the
            e4m3 output is round(x / scale). The e4m3 packer rounds toward zero (truncates the
            mantissa), so results are within one e4m3 ULP of a round-to-nearest reference.

            With ``round_scale_to_power_of_two=True``, the scale is rounded upward to a power of two
            after division by the E4M3FN maximum finite magnitude, 448. Sparse MLA KV caching uses
            this mode for UE8M0-style scaling.

            Args:
                * :attr:`input_tensor`: BFLOAT16 or FLOAT32 ROW_MAJOR or TILE tensor of shape [..., M, H].
                  Requires H % 128 == 0, DRAM interleaved memory, and Blackhole hardware.
                * :attr:`memory_config`: optional DRAM interleaved output memory config
                  (default: same as input).
                * :attr:`round_scale_to_power_of_two`: round each scale upward to a power of two.
                * :attr:`scale_block_w`: elements per scale (32, 64 or 128; default 128). The scale
                  tensor's last dim is H / scale_block_w. 32 matches MXFP8 block granularity.
                * :attr:`round_to_nearest_even`: round the scaled values to e4m3 with round-to-nearest-even
                  (SFPU) before packing, instead of the packer's truncation.

            Returns:
                Tuple (e4m3, scale_fp32). Both are ROW_MAJOR. FP8_E4M3 can only be in ROW_MAJOR layout.
        )doc",
        &per_token_cast_to_fp8,
        nb::arg("input_tensor").noconvert(),
        nb::arg("memory_config") = std::nullopt,
        nb::kw_only(),
        nb::arg("round_scale_to_power_of_two") = false,
        nb::arg("scale_block_w") = DEFAULT_SCALE_BLOCK_W,
        nb::arg("round_to_nearest_even") = false,
        nb::arg("token_count_aware") = false,
        nb::arg("expert_region_offsets") = nb::none(),
        nb::arg("expert_token_counts") = nb::none(),
        nb::arg("global_expert_idx_table") = nb::none(),
        nb::arg("experts_per_chip") = 0,
        nb::arg("pack_scales_in_row") = false,
        nb::arg("packed_row_align") = 0);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8::detail
