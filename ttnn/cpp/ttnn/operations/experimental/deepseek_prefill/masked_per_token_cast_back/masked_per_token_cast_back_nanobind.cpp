// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "masked_per_token_cast_back_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "masked_per_token_cast_back.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::masked_per_token_cast_back::detail {

void bind_experimental_masked_per_token_cast_back_operation(nb::module_& mod) {
    ttnn::bind_function<"masked_per_token_cast_back", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Masked inverse of per_token_cast_to_fp8 for MoE dispatch buffers: take an FP8_E4M3 tensor +
            a FLOAT32 scale tensor and recover a BFLOAT16/FLOAT32 tensor (out = decode(e4m3) * scale),
            but only dequantize the valid, contiguously-packed prefix of the buffer. The number of valid
            rows is computed device-side from the per-expert token counts and region offsets, and the work
            is balanced across the whole Tensix grid (no host<->device sync).

            The valid prefix length is:
                total_valid_rows = max over local expert slots s of (
                    expert_region_offsets[g] + ceil_tile(expert_token_counts[g]) ),
                where g = global_expert_idx_table[s].
            Rows beyond total_valid_rows are left untouched (garbage tail).

            Args:
                * :attr:`input_e4m3`: FP8_E4M3 ROW_MAJOR tensor of shape [..., M, H]. Requires
                  H % 128 == 0, DRAM interleaved memory, and Blackhole hardware.
                * :attr:`input_scale`: FLOAT32 ROW_MAJOR DRAM interleaved tensor of shape [..., M, H/128].
                  Provide either this OR `metadata` (exactly one).
                * :attr:`metadata`: optional dispatch metadata tensor [..., M, metadata_len] (int32/uint32),
                  whose row tail (columns [metadata_len - H/128, metadata_len)) holds the per-token fp32
                  scales bit-stored as int32. When given, scales are read from here instead of `input_scale`.
                * :attr:`expert_region_offsets`: UINT32 ROW_MAJOR DRAM interleaved, (1, num_routed_experts).
                * :attr:`expert_token_counts`: UINT32 ROW_MAJOR DRAM interleaved, (1, num_routed_experts).
                * :attr:`global_expert_idx_table`: UINT32 ROW_MAJOR DRAM interleaved, (experts_per_chip,).
                * :attr:`experts_per_chip`: number of local experts hosted on this device.
                * :attr:`output_dtype`: BFLOAT16 (default) or FLOAT32.
                * :attr:`memory_config`: optional DRAM interleaved output memory config
                  (default: same as input_e4m3).

            Returns:
                A ROW_MAJOR tensor of the same logical shape as input_e4m3, dtype = output_dtype. Only the
                valid prefix rows are written.
        )doc",
        &masked_per_token_cast_back,
        nb::arg("input_e4m3").noconvert(),
        nb::arg("input_scale").noconvert(),  // pass a FLOAT32 (M,H/128) scale tensor, or None when using `metadata`
        nb::arg("expert_region_offsets").noconvert(),
        nb::arg("expert_token_counts").noconvert(),
        nb::arg("global_expert_idx_table").noconvert(),
        nb::arg("experts_per_chip"),
        nb::arg("output_dtype") = std::nullopt,
        nb::arg("memory_config") = std::nullopt,
        nb::arg("metadata").noconvert() = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::masked_per_token_cast_back::detail
