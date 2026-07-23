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

            ``token_count_aware`` selects between two behaviors of this single op:

            * ``False`` (default): dequantize the whole [..., M, H] buffer. Only ``input_scale`` is used.
            * ``True``: dequantize only the valid, contiguously-packed prefix of a MoE dispatch buffer.
              The number of valid rows is computed device-side from the per-expert token counts and region
              offsets, and the work is balanced across the whole Tensix grid (no host<->device sync). The
              valid prefix length is
              ``max over local expert slots s of (expert_region_offsets[g] + ceil_tile(expert_token_counts[g]))``
              where ``g = global_expert_idx_table[s]``; rows beyond it are left untouched (garbage tail).

            Args:
                * :attr:`input_e4m3`: FP8_E4M3 ROW_MAJOR tensor of shape [..., M, H]. Requires
                  H % 128 == 0, DRAM interleaved memory, and Blackhole hardware.
                * :attr:`input_scale`: FLOAT32 ROW_MAJOR DRAM interleaved tensor of shape [..., M, H/128].
                  Required on the plain path; on the token-count-aware path provide either this OR
                  `metadata` (exactly one).
                * :attr:`output_dtype`: BFLOAT16 (default) or FLOAT32.
                * :attr:`memory_config`: optional DRAM interleaved output memory config
                  (default: same as input_e4m3).
                * :attr:`token_count_aware`: enable the token-count-aware MoE-dispatch prefix path (default False).
                * :attr:`expert_region_offsets`: UINT32 ROW_MAJOR DRAM interleaved, (1, num_routed_experts).
                  Required when token_count_aware=True.
                * :attr:`expert_token_counts`: UINT32 ROW_MAJOR DRAM interleaved, (1, num_routed_experts).
                  Required when token_count_aware=True.
                * :attr:`global_expert_idx_table`: UINT32 ROW_MAJOR DRAM interleaved, (experts_per_chip,).
                  Required when token_count_aware=True.
                * :attr:`experts_per_chip`: number of local experts hosted on this device
                  (token_count_aware=True).
                * :attr:`metadata`: optional dispatch metadata tensor [..., M, metadata_len] (int32/uint32)
                  whose row tail (columns [metadata_len - H/128, metadata_len)) holds the per-token fp32
                  scales bit-stored as int32. When given (token_count_aware=True), scales are read from
                  here instead of `input_scale`.

            Returns:
                A ROW_MAJOR tensor of the same logical shape as input_e4m3, dtype = output_dtype. On the
                token-count-aware path only the valid prefix rows are written.
        )doc",
        &per_token_cast_back,
        nb::arg("input_e4m3").noconvert(),
        nb::arg("input_scale").noconvert() = std::nullopt,
        nb::arg("output_dtype") = std::nullopt,
        nb::arg("memory_config") = std::nullopt,
        nb::arg("token_count_aware") = false,
        nb::arg("expert_region_offsets").noconvert() = std::nullopt,
        nb::arg("expert_token_counts").noconvert() = std::nullopt,
        nb::arg("global_expert_idx_table").noconvert() = std::nullopt,
        nb::arg("experts_per_chip") = 0,
        nb::arg("metadata").noconvert() = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back::detail
