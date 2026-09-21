// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fused_recurrent_gated_delta_rule_nanobind.hpp"
#include "fused_recurrent_gated_delta_rule.hpp"

#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

namespace ttnn::operations::transformer {

void bind_fused_recurrent_gated_delta_rule(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Fused recurrent Gated Delta Rule forward (flash-linear-attention algorithm).

        Single-token decode (T=1) and K+1-token speculative verify are the same op: one Tensix core
        per head walks the T token axis sequentially, holding the recurrent state on-core. Matches
        FLA `naive_recurrent_gated_delta_rule` (fp32/HiFi4). q/k must be L2-normalized over K on host.

        Args:
            q (ttnn.Tensor):    [B, T, H,  K]  (L2-normalized over K)
            k (ttnn.Tensor):    [B, T, H,  K]  (L2-normalized over K)
            v (ttnn.Tensor):    [B, T, HV, V]
            g (ttnn.Tensor):    [B, T, HV]     log-space decay (exp(g) applied internally)
            beta (ttnn.Tensor): [B, T, HV]     gate (already sigmoid'd)

        Keyword Args:
            scale (float, optional): query scale, defaults to K**-0.5.
            initial_state (ttnn.Tensor, optional): [B, HV, K, V]; in "ring" mode instead the
                [T*BH, K, V] ring (see initial_state_block_idx).
            initial_state_block_idx (ttnn.Tensor, optional): "ring" mode -- deferred per-head
                initial-state select, IN PLACE. [BH] (or [1, BH]) uint32/int32, ROW_MAJOR, on
                device, interleaved, with BH = B*HV and h = b*HV + hv (b-major). Requires
                output_per_token_state=True and an fp32 TILE interleaved `initial_state` of EXACT
                shape [T*BH, K, V] (the ring; block t*BH + h is head h's state after token t).
                Head h then starts from ring block idx[h] instead of block h, and its per-token
                states are written back into the SAME buffer. The returned state IS that tensor
                (same buffer, shape [T*BH, K, V]), not a copy or a reshape.
                CALLER CONTRACT: idx[h] % BH == h for every h. Each core then only reads a block it
                wrote itself, and it reads its whole initial state before its first per-token
                write, so no cross-core ordering is needed (there are no semaphores).
            output_final_state (bool): default False.
            output_per_token_state (bool): default False. When True, returns the state AFTER every
                token as [B, T, HV, K, V] (speculative-verify slots); overrides output_final_state.
            use_qk_l2norm (bool): must be False (L2-norm is done on host).
            memory_config (ttnn.MemoryConfig, optional).
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional).

        Returns:
            tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
                o [B, T, HV, V],
                state [B, T, HV, K, V] (per-token) or [B, HV, K, V] (final), if requested.
        )doc";

    ttnn::bind_function<"fused_recurrent_gated_delta_rule", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::fused_recurrent_gated_delta_rule,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("beta").noconvert(),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("initial_state_block_idx") = nb::none(),
        nb::arg("output_final_state") = false,
        nb::arg("output_per_token_state") = false,
        nb::arg("use_qk_l2norm") = false,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
