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
            e4m3 output is round(x / scale). By default the e4m3 packer rounds toward zero (truncates
            the mantissa), so results are within one e4m3 ULP of a round-to-nearest reference.

            With ``round_to_nearest=True``, half an e4m3 step of each value's own binade (2^(e-4), or
            2^-10 in the subnormal range below 2^-6) is added to its magnitude before the truncating
            pack, so the output rounds to nearest with ties away from zero. This is bit-identical to
            the Hy4 decode KV append's ``nearest`` mode. Magnitudes stay within 448 (no saturation
            change): the scale bounds |x / scale| by 448, and 448 + 16 still truncates to 448.

            With ``round_scale_to_power_of_two=True``, the scale is rounded upward to a power of two
            after division by the E4M3FN maximum finite magnitude, 448. Sparse MLA KV caching uses
            this mode for UE8M0-style scaling.

            Args:
                * :attr:`input_tensor`: BFLOAT16 or FLOAT32 ROW_MAJOR or TILE tensor of shape [..., M, H].
                  Requires H % 128 == 0, DRAM interleaved memory, and Blackhole hardware.
                * :attr:`memory_config`: optional DRAM interleaved output memory config
                  (default: same as input).
                * :attr:`round_scale_to_power_of_two`: round each scale upward to a power of two.
                * :attr:`round_to_nearest`: round the e4m3 values to nearest (ties away from zero)
                  instead of truncating. Default False (truncate).

            Returns:
                Tuple (e4m3, scale_fp32). Both are ROW_MAJOR. FP8_E4M3 can only be in ROW_MAJOR layout.
        )doc",
        &per_token_cast_to_fp8,
        nb::arg("input_tensor").noconvert(),
        nb::arg("memory_config") = std::nullopt,
        nb::kw_only(),
        nb::arg("round_scale_to_power_of_two") = false,
        nb::arg("round_to_nearest") = false);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8::detail
