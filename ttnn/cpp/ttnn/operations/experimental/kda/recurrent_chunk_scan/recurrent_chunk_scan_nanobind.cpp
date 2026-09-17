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

        Optional ``actual_end`` is a replicated UINT32 row-major scalar, with
        the same lifetime as ``actual_start``. It defines a nonempty 32-aligned
        interval within physical capacity; omission uses the full capacity.
        Bounds may change during trace replay. Padded group outputs are unspecified.
        Bounds are caller preconditions and are not read back on the host.

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
            group_entry_states (ttnn.Tensor): Initial recurrent state ``[B*H*G, K, V]``
                in FLOAT32, with group folded into the leading dimension.
                Tail state is unfolded: one ``[K,V]`` matrix per ``B*H``.

        Keyword Args:
            actual_start (ttnn.Tensor): Replicated UINT32 row-major scalar
                containing the absolute position of the chunk's first token; pass [0]
                for zero-offset execution. Its
                value must be nonnegative and 32-aligned. Keep its address stable
                and update its contents before replay of a captured trace.
            sequence_parallel_axis (int, optional): Mesh axis partitioning the
                sequence. Native mesh coordinates supply each device's rank.
            tail_entry_states (ttnn.Tensor): FLOAT32 carry ``[B*H,K,V]``
                to reload at the locally derived split. Ignored when unsplit.
                No input tensor is modified.
            groups_per_head (int, optional): Groups folded into the leading
                dimension. Defaults to 1.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory
                configuration. Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Compute-kernel configuration.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: New tensors containing BFLOAT16 token
                outputs ``Y[B*H*G,N,32,V]`` and FLOAT32 final state ``S[B*H*G,K,V]``.

        Note:
            ``v_beta``, ``kd``, ``q_decay``, ``k_dec_t``, and ``final_decay`` may be
            FLOAT32 or BFLOAT16. ``intra``, ``t_inv``, and ``group_entry_states`` must be
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
        nb::arg("group_entry_states").noconvert(),
        nb::kw_only(),
        nb::arg("actual_start").noconvert(),
        nb::arg("tail_entry_states").noconvert(),

        nb::arg("groups_per_head") = 1,

        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0,
        nb::arg("actual_end") = nb::none());

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

        Optional ``actual_end`` is a replicated UINT32 row-major scalar, with
        the same lifetime as ``actual_start``. It defines a nonempty 32-aligned
        interval within physical capacity; omission uses the full capacity.
        Bounds may change during trace replay. Padded group outputs are unspecified.
        Bounds are caller preconditions and are not read back on the host.

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
            actual_start (ttnn.Tensor): Replicated UINT32 row-major scalar
                containing the absolute position of the chunk's first token; pass [0]
                for zero-offset execution. Its
                value must be nonnegative and 32-aligned. Keep its address stable
                and update its contents before replay of a captured trace.
            sequence_parallel_axis (int, optional): Mesh axis partitioning the
                sequence. Native mesh coordinates supply each device's rank.
            groups_per_head (int, optional): Groups folded into the leading
                dimension. Defaults to 1.
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration.
                Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Compute-kernel configuration.

        Returns:
            tuple[ttnn.Tensor, ...]: Four BFLOAT16 TILE tensors: head A/B followed by
                tail A/B. A has shape ``[B*H*G,K,K]`` and B ``[B*H*G,K,V]``.

        Note:
            Summaries accumulate in FLOAT32 and pack directly to BFLOAT16 for transport.
            Unsplit execution defines only the head pair. Inactive head/tail slots
            are unspecified; consumers must use the same chronology and geometry.

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
        nb::arg("actual_start").noconvert(),

        nb::arg("groups_per_head") = 1,

        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0,
        nb::arg("actual_end") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::recurrent_chunk_scan::detail
