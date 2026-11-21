// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_bh_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "sdpa_decode_bh.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::experimental::llama31_8b_v3 {

void bind_sdpa_decode_bh(nb::module_& llama31_8b_v3) {
    const auto* doc = R"doc(
        Paged causal scaled dot product attention decode for Llama-3.1-8B on Blackhole.

        Output is bit-identical to ``ttnn.transformer.paged_scaled_dot_product_attention_decode`` with
        is_causal=True and the same page table, cur_pos tensor, scale, program config and compute kernel config.

        Args:
            input_tensor_q (ttnn.Tensor): [1, B, 32, 128].
            input_tensor_k (ttnn.Tensor): paged K cache [blocks, 8, 32, 128] bfloat8_b.
            input_tensor_v (ttnn.Tensor): paged V cache [blocks, 8, 32, 128] bfloat8_b.
            page_table_tensor (ttnn.Tensor): [B, blocks per user] int32.
            cur_pos_tensor (ttnn.Tensor): [B] int32.
            program_config (ttnn.SDPAProgramConfig): k_chunk_size 0.

        Keyword Args:
            scale (float, optional): defaults to 1 / sqrt(128).
            memory_config (ttnn.MemoryConfig, optional): output memory config.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): defaults to HiFi2 with approx.
            multi_pos (bool, optional): rows of Q are positions cur_pos[b] of one user (same page table rows,
                ascending); one KV pass per head; row b equals the single-row call at cur_pos[b] when it falls in
                the same k-chunk as row 0, otherwise row b is left unwritten.

        Returns:
            ttnn.Tensor: [1, B, 32, 128].
        )doc";

    ttnn::bind_function<"sdpa_decode_bh", "ttnn.experimental.llama31_8b_v3.">(
        llama31_8b_v3,
        doc,
        &sdpa_decode_bh,
        nb::arg("input_tensor_q"),
        nb::arg("input_tensor_k"),
        nb::arg("input_tensor_v"),
        nb::arg("page_table_tensor"),
        nb::arg("cur_pos_tensor"),
        nb::arg("program_config"),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("multi_pos") = false);
}

}  // namespace ttnn::experimental::llama31_8b_v3
