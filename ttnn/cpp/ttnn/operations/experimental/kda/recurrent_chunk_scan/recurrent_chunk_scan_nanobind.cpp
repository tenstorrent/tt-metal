// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_chunk_scan_nanobind.hpp"

#include "recurrent_chunk_scan.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::recurrent_chunk_scan::detail {

void bind_recurrent_chunk_scan(nb::module_& mod) {
    ttnn::bind_function<"recurrent_chunk_scan", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Apply the KDA recurrence over an ordered sequence of prepared 32-token chunks.

        For each chunk ``n``:

            U_n     = t_inv_n @ (v_beta_n - kd_n @ S_n)
            Y_n     = q_decay_n @ S_n + intra_n @ U_n
            S_{n+1} = final_decay_n * S_n + k_dec_t_n @ U_n

        Args:
            v_beta (ttnn.Tensor): Prepared values ``[B*H, N, 32, V]``.
            kd (ttnn.Tensor): Prepared decayed keys ``[B*H, N, 32, K]``.
            q_decay (ttnn.Tensor): Prepared decayed queries ``[B*H, N, 32, K]``.
            intra (ttnn.Tensor): Causal within-chunk interactions
                ``[B*H, N, 32, 32]`` in FLOAT32.
            k_dec_t (ttnn.Tensor): Prepared transposed key term
                ``[B*H, N, K, 32]``.
            final_decay (ttnn.Tensor): End-of-chunk state decay
                ``[B*H, N, K, 1]``.
            t_inv (ttnn.Tensor): Triangular correction inverse
                ``[B*H, N, 32, 32]`` in FLOAT32.
            initial_state (ttnn.Tensor): Initial recurrent state ``[B*H, K, V]``
                in FLOAT32.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory
                configuration. Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Compute-kernel configuration.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: New tensors containing BFLOAT16 token
                outputs ``Y[B*H,N,32,V]`` and FLOAT32 final state ``S[B*H,K,V]``.

        Note:
            ``v_beta``, ``kd``, ``q_decay``, ``k_dec_t``, and ``final_decay`` may be
            FLOAT32 or BFLOAT16. ``intra``, ``t_inv``, and ``initial_state`` must be
            FLOAT32. ``K`` and ``V`` must be positive and tile-aligned. All inputs
            must be interleaved TILE-layout tensors on the same device and are not
            modified.
        )doc",
        &ttnn::experimental::kda::recurrent_chunk_scan,
        nb::arg("v_beta").noconvert(),
        nb::arg("kd").noconvert(),
        nb::arg("q_decay").noconvert(),
        nb::arg("intra").noconvert(),
        nb::arg("k_dec_t").noconvert(),
        nb::arg("final_decay").noconvert(),
        nb::arg("t_inv").noconvert(),
        nb::arg("initial_state").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"summarize_chunk_recurrence", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Summarize each group of prepared chunks into one affine state transition.

        In grouped execution, groups are flattened into the leading dimension:

            leading = B * H * G

        The operation applies the same state update as ``recurrent_chunk_scan`` without
        producing token outputs and returns one transform per batch-head-group:

            S_after = A @ S_before + B

        It derives the transform through two parallel recurrence evaluations:

            B = F(0)
            A = F(I) - B

        Args:
            v_beta (ttnn.Tensor): Prepared values ``[B*H*G, N, 32, V]``.
            kd (ttnn.Tensor): Prepared decayed keys ``[B*H*G, N, 32, K]``.
            q_decay (ttnn.Tensor): Prepared decayed queries ``[B*H*G, N, 32, K]``.
            intra (ttnn.Tensor): Causal within-chunk interactions
                ``[B*H*G, N, 32, 32]`` in FLOAT32.
            k_dec_t (ttnn.Tensor): Prepared transposed key term
                ``[B*H*G, N, K, 32]``.
            final_decay (ttnn.Tensor): End-of-chunk state decay
                ``[B*H*G, N, K, 1]``.
            t_inv (ttnn.Tensor): Triangular correction inverse
                ``[B*H*G, N, 32, 32]`` in FLOAT32.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration.
                Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Compute-kernel configuration.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: New FLOAT32 TILE-layout tensors
                ``A[B*H*G,K,K]`` and ``B[B*H*G,K,V]``.

        Note:
            The current summary path requires ``K=V``. ``q_decay`` and ``intra`` are
            accepted as part of the shared prepared-chunk protocol but do not contribute
            to the state-only summary. All inputs must be interleaved TILE-layout tensors
            on the same device and are not modified.
        )doc",
        &ttnn::experimental::kda::summarize_chunk_recurrence,
        nb::arg("v_beta").noconvert(),
        nb::arg("kd").noconvert(),
        nb::arg("q_decay").noconvert(),
        nb::arg("intra").noconvert(),
        nb::arg("k_dec_t").noconvert(),
        nb::arg("final_decay").noconvert(),
        nb::arg("t_inv").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::recurrent_chunk_scan::detail
