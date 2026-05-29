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

            v0 currently emits scale = 1.0 for every group; real per-row amax
            quantization is deferred.

            Args:
                * :attr:`input_tensor`: BFLOAT16 or FLOAT32 ROW_MAJOR tensor of shape [..., M, H].
                  Requires H % 128 == 0. Blackhole only.
                * :attr:`memory_config`: optional output memory config (default: same as input).

            Returns:
                Tuple (e4m3, scale_fp32). Both are ROW_MAJOR.
        )doc",
        &per_token_cast_to_fp8,
        nb::arg("input_tensor").noconvert(),
        nb::arg("memory_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8::detail
