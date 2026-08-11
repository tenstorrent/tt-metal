// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prepare_chunk_recurrence_nanobind.hpp"

#include "prepare_chunk_recurrence.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::prepare_chunk_recurrence::detail {

void bind_prepare_chunk_recurrence(nb::module_& mod) {
    ttnn::bind_function<"prepare_chunk_recurrence", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Convert flat KDA Q, K, V, gate, and beta tensors into the seven chunk-local
        terms consumed by ``recurrent_chunk_scan``.

        The operation partitions the sequence into 32-token chunks, normalizes Q and
        K, accumulates the log-decay gate, constructs causal within-chunk interactions,
        and computes the triangular correction used by the recurrence.

        Args:
            q (ttnn.Tensor): Flat queries ``[1, T, H*K]`` in BFLOAT16.
            k (ttnn.Tensor): Flat keys ``[1, T, H*K]`` in BFLOAT16.
            v (ttnn.Tensor): Flat values ``[1, T, H*V]`` in BFLOAT16.
            g (ttnn.Tensor): Flat per-key log decays ``[1, T, H*K]`` in BFLOAT16.
            beta (ttnn.Tensor): Per-token update strengths ``[H, N, 32, 1]`` in
                FLOAT32, where ``N = T / 32``.
            eye (ttnn.Tensor): Identity constant ``[1, 1, 32, 32]`` in FLOAT32.
            tril (ttnn.Tensor): Lower-triangular constant ``[1, 1, 32, 32]`` in
                FLOAT32.
            ones (ttnn.Tensor): Ones constant ``[1, 1, 32, 32]`` in FLOAT32.
            masks (ttnn.Tensor): Packed mask constants ``[1, 1, 32, 96]`` in
                FLOAT32.
            num_heads (int): Number of heads ``H``. Flat Q/K/G and V widths must be
                divisible by ``H``.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory
                configuration. Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Compute-kernel configuration.
            output_bf16_mask (int): Bit mask selecting BFLOAT16 storage for outputs.
                Bits 0, 1, 2, 4, and 5 are supported; unselected outputs use FLOAT32.
                Defaults to 0.

        Returns:
            list[ttnn.Tensor]: Seven new TILE-layout tensors, in order:

                * ``v_beta[H,N,32,V]``
                * ``kd[H,N,32,K]``
                * ``q_decay[H,N,32,K]``
                * ``intra[H,N,32,32]``
                * ``k_dec_t[H,N,K,32]``
                * ``final_decay[H,N,K,1]``
                * ``t_inv[H,N,32,32]``

        Note:
            ``T`` must be positive and divisible by 32; ``K`` and ``V`` must be
            positive and tile-aligned. All inputs must be interleaved TILE-layout
            device tensors on the same device and are not modified.
        )doc",
        &ttnn::experimental::kda::prepare_chunk_recurrence,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("eye").noconvert(),
        nb::arg("tril").noconvert(),
        nb::arg("ones").noconvert(),
        nb::arg("masks").noconvert(),
        nb::arg("num_heads"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_bf16_mask") = 0);
}

}  // namespace ttnn::operations::experimental::kda::prepare_chunk_recurrence::detail
