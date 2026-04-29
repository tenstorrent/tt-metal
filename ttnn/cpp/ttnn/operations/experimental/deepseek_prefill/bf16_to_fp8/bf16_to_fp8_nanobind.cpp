// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bf16_to_fp8_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "bf16_to_fp8.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8::detail {

void bind_bf16_to_fp8(nb::module_& mod) {
    ttnn::bind_function<"bf16_to_fp8", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Test op: convert a BF16 TILE-layout tensor to Fp8_e4m3 via the packer.

            Returns a UINT8 tensor whose bytes are Fp8_e4m3-encoded values; same shape
            and TILE layout as the input. Single core, one tile at a time, no
            pack_untilize. Intended only for verifying BF16->FP8 conversion accuracy.

            Args:
                input_tensor (ttnn.Tensor): 2D BFLOAT16 TILE-layout DRAM-interleaved tensor.
                    Both dims must be multiples of 32.

            Returns:
                ttnn.Tensor: UINT8 TILE-layout DRAM-interleaved tensor with Fp8_e4m3 bytes.
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8::bf16_to_fp8,
        nb::arg("input_tensor").noconvert());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_bf16_to_fp8(::nanobind::module_& mod) { bf16_to_fp8::detail::bind_bf16_to_fp8(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
