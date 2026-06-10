// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_back_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

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

            Masked decompress mode (optional):
                When :attr:`expert_token_counts`, :attr:`expert_region_offsets`, and :attr:`metadata`
                are all supplied (together with :attr:`experts_per_chip` and :attr:`dispatch_group_size`),
                input_e4m3 is treated as a per-device dispatch buffer of shape (1, 1, T, H), where
                T = max_dispatch_buffer_token_size. Only the rows covered by this device's expert
                regions are decompressed, defined by expert_token_counts / expert_region_offsets and
                this device's mesh-coordinate window.
                For each valid row r, the scale used is input_scale[ metadata[r][1] ] (the token_idx
                field). Rows outside any valid expert region (garbage) are left untouched.

                * :attr:`expert_token_counts`: INT32/UINT32 ROW_MAJOR, shape (1, 1, num_routed_experts).
                * :attr:`expert_region_offsets`: INT32/UINT32 ROW_MAJOR, same shape as expert_token_counts.
                  Produced by ttnn.experimental.deepseek_prefill.offset_cumsum.
                * :attr:`metadata`: INT32 ROW_MAJOR, shape (1, 1, T, 5). Field [1] is token_idx.
                * :attr:`experts_per_chip`: experts hosted on each device.
                * :attr:`dispatch_group_size`: devices per dispatch group.

            Returns:
                A ROW_MAJOR tensor of the same logical shape as input_e4m3, dtype = output_dtype.
                In masked mode, only valid rows are written; garbage rows are uninitialized.
        )doc",
        &per_token_cast_back,
        nb::arg("input_e4m3").noconvert(),
        nb::arg("input_scale").noconvert(),
        nb::arg("output_dtype") = std::nullopt,
        nb::arg("memory_config") = std::nullopt,
        nb::arg("expert_token_counts").noconvert() = std::nullopt,
        nb::arg("expert_region_offsets").noconvert() = std::nullopt,
        nb::arg("metadata").noconvert() = std::nullopt,
        nb::arg("experts_per_chip") = std::nullopt,
        nb::arg("dispatch_group_size") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back::detail
