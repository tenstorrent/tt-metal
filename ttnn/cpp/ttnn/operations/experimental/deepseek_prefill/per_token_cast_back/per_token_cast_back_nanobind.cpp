// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_back_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "per_token_cast_back.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back::detail {

void bind_experimental_per_token_cast_back_operation(nb::module_& mod) {
    ttnn::bind_function<"per_token_cast_back", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Inverse of per_token_cast_to_fp8: take an FP8_E4M3 tensor + a FLOAT32 scale tensor and
            recover a BFLOAT16/FLOAT32 tensor: out = decode(e4m3) * scale, where each scale applies
            to its 128-element block of the token (the scale's last dim is H/128).

            Args:
                * :attr:`input_e4m3`: FP8_E4M3 ROW_MAJOR tensor of shape [..., M, H]. Requires
                  H % 128 == 0, DRAM interleaved memory, and Blackhole hardware.
                * :attr:`input_scale`: FLOAT32 ROW_MAJOR DRAM interleaved tensor of shape [..., M, H/128].
                * :attr:`output_dtype`: BFLOAT16 (default) or FLOAT32.
                * :attr:`memory_config`: optional DRAM interleaved output memory config
                  (default: same as input_e4m3).
                * :attr:`narrow_scales_to_bf16`: when True, narrow the fp32 scale to bf16 on-device and run the
                  broadcast multiply in bf16 (HiFi2); when False (default), keep the fp32 (HiFi4) datapath.

            Returns:
                A ROW_MAJOR tensor of the same logical shape as input_e4m3, dtype = output_dtype.
        )doc",
        &per_token_cast_back,
        nb::arg("input_e4m3").noconvert(),
        nb::arg("input_scale").noconvert(),
        nb::arg("output_dtype") = std::nullopt,
        nb::arg("memory_config") = std::nullopt,
        nb::arg("narrow_scales_to_bf16") = false);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back::detail
