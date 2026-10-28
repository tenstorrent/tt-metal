// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fused_recurrent_gated_delta_rule_nanobind.hpp"
#include "fused_recurrent_gated_delta_rule.hpp"
#include "fused_recurrent_gated_delta_rule_config.hpp"

#include "ttnn-nanobind/bind_function.hpp"

#include <fmt/format.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <string>

namespace ttnn::operations::transformer {

void bind_fused_recurrent_gated_delta_rule(nb::module_& mod) {
    using ttnn::transformer::FusedRecurrentGdnProgramConfig;
    auto py_opt = [](const std::optional<uint32_t>& v) { return v ? std::to_string(*v) : std::string("None"); };

    nb::class_<FusedRecurrentGdnProgramConfig>(
        mod,
        "FusedRecurrentGdnProgramConfig",
        R"doc(Geometry of ttnn.transformer.fused_recurrent_gated_delta_rule: how the (batch, value head) work is cut
        into units and placed on the grid. A field left None is chosen by the op. This build runs one whole head
        per core and accepts only the defaults; stage 2 of the v2 plan fills them from a cost model and accepts
        the rest.

        Keyword Args:
            num_slices (int, optional): NV, the V-slices per head: 1, 2 or 4 (V / 32 / NV tile columns per unit).
            units_per_core (int, optional): units (b, hv, V-slice) looped per core.)doc")
        .def(
            nb::init<std::optional<uint32_t>, std::optional<uint32_t>>(),
            nb::kw_only(),
            nb::arg("num_slices") = nb::none(),
            nb::arg("units_per_core") = nb::none())
        .def_rw("num_slices", &FusedRecurrentGdnProgramConfig::num_slices)
        .def_rw("units_per_core", &FusedRecurrentGdnProgramConfig::units_per_core)
        .def("__repr__", [py_opt](const FusedRecurrentGdnProgramConfig& c) {
            return fmt::format(
                "FusedRecurrentGdnProgramConfig(num_slices={}, units_per_core={})",
                py_opt(c.num_slices),
                py_opt(c.units_per_core));
        });

    const auto* doc =
        R"doc(
        Fused recurrent Gated Delta Rule (flash-linear-attention `fused_recurrent_gated_delta_rule`): the per-token
        recurrence of gated-DeltaNet decode (T = 1) and speculative verify (T = K + 1), one program per call, the
        state carried on-core across the T tokens.

        Per token t and value head hv, which reads key head hv // (HV // H), with S the fp32 state [K, V]:

            q̂ = q / sqrt(sum(q²) + ε), k̂ = k / sqrt(sum(k²) + ε)   use_qk_l2norm_in_kernel; ε = 1e-6 INSIDE the root
            q̃ = q̂ · scale                                          scale AFTER the norm; default K**-0.5
            S ← exp(g_t) · S                                        decay before the read
            u = β_t · (v_t − k̂ S);  S ← S + k̂ᵀ u;  o_t = q̃ S       o reads the POST-update state

        Argument order: positional (q, k, v, g, beta), as FLA's fused kernels. FLA's naive reference
        `naive_recurrent_gated_delta_rule` takes (q, k, v, beta, g): the last two are swapped. The registered golden
        (ttnn.get_golden_function) takes this op's order.

        Args:
            q (ttnn.Tensor): [B, T, H, K] bf16 or fp32, TILE, interleaved. Normalised over K by the caller unless
                use_qk_l2norm_in_kernel.
            k (ttnn.Tensor): [B, T, H, K], as q.
            v (ttnn.Tensor): [B, T, HV, V]. HV is a multiple of H.
            g (ttnn.Tensor): [B, T, HV] log-space decay, g ≤ 0 (exp(g) is applied in the op); [B, T, HV, K] is the
                KDA per-key decay (stage 5).
            beta (ttnn.Tensor): [B, T, HV] post-sigmoid write strength (raw logits with use_beta_sigmoid_in_kernel,
                stage 5).

        Keyword Args:
            scale (float, optional): multiplies q after the optional norm. Default K**-0.5.
            initial_state (ttnn.Tensor, optional): [B, HV, K, V] fp32, the state before token 0; zeros when absent.
                With state_indices it is the pool [slots, HV, K, V] (stage 3a); in ring mode the [T*B*HV, K, V] ring.
            state_indices (ttnn.Tensor, optional): int32 [B] or [B, T]: the pool slot of each row, or of each
                (row, token); a negative index skips the row. Stage 3a.
            num_accepted_tokens (ttnn.Tensor, optional): int32 [B]; with 2-D state_indices, row b starts from slot
                state_indices[b, n_b - 1]. Stage 3a.
            inplace_final_state (bool): write the states back into initial_state at state_indices and return the
                pool itself. Stage 3a.
            output_final_state (bool): return the state after the last token, [B, HV, K, V].
            output_per_token_state (bool): return the state after every token, [B, T, HV, K, V]; overrides
                output_final_state.
            use_qk_l2norm_in_kernel (bool): normalise q and k over K inside the op (ε inside the root), before the
                scale. Formerly `use_qk_l2norm`, which no in-tree caller passed.
            use_gate_in_kernel (bool), A_log (ttnn.Tensor [HV]), dt_bias (ttnn.Tensor [HV], KDA [HV*K]),
                lower_bound (float): compute g inside the op from raw gate values: GDN
                g = -exp(A_log) * softplus(g_raw + dt_bias); KDA bounded form
                g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias)). Stage 5.
            use_beta_sigmoid_in_kernel (bool), allow_neg_eigval (bool): β = sigmoid(beta), times 2 with
                allow_neg_eigval. Stage 5.
            cu_seqlens (ttnn.Tensor, optional): int32 [N+1] sequence boundaries over T, B == 1. Not scheduled.
            state_v_first (bool): accept vLLM's [.., HV, V, K] state layout. Stage 8 decision.
            output_tensor (ttnn.Tensor, optional): preallocated o [B, T, HV, V]. Stage 4.
            initial_state_block_idx (ttnn.Tensor, optional): deprecated ring mode, kept for the Qwen3.6 verify path
                until stage 7 moves it to state_indices. Head h starts from ring block idx[h] of the [T*BH, K, V]
                initial_state instead of block h and writes its per-token states back in place; the returned state
                is that same tensor. [BH] or [1, BH] uint32/int32, ROW_MAJOR, interleaved on device; BH = B*HV,
                h = b*HV + hv. Requires output_per_token_state=True and an fp32 TILE initial_state of exact shape
                [T*BH, K, V]. Caller contract: idx[h] % BH == h for every h.
            memory_config (ttnn.MemoryConfig, optional): placement of the outputs; DRAM interleaved by default.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): HiFi4, fp32 accumulation and exact
                mode by default; fp32 accumulation is required.
            program_config (ttnn.transformer.FusedRecurrentGdnProgramConfig, optional): the geometry; absent, the
                op chooses. This build accepts only the defaults (one whole head per core).

        Returns:
            tuple[ttnn.Tensor, Optional[ttnn.Tensor]]: o [B, T, HV, V] fp32 TILE, row t the read of the state after
            token t's own write; state None, [B, HV, K, V] (final), [B, T, HV, K, V] (per token), or the ring
            tensor itself in ring mode.

        What this build serves and what it rejects (the message names the stage of the v2 plan that brings it):

            served     fp32 or bf16 TILE inputs (bf16 is cast on the host; o is fp32 until stage 4), initial_state
                       [B, HV, K, V], the final and the per-token state, ring mode, use_qk_l2norm_in_kernel (as
                       host ops until stage 4), scale, program_config with default fields, T up to 32
            stage 2    B*HV above the compute grid; num_slices 2 or 4; units_per_core above 1
            stage 3a   state_indices, num_accepted_tokens, inplace_final_state
            stage 4    output_tensor; ROW_MAJOR inputs
            stage 5    rank-4 g (KDA), use_gate_in_kernel, A_log, dt_bias, lower_bound,
                       use_beta_sigmoid_in_kernel, allow_neg_eigval
            stage 8    state_v_first
            never      T above 32; cu_seqlens is not scheduled

        Every combination rule of the contract is checked before the stage check, so a wrong combination fails on
        its own message whether or not the feature is served: HV must be a multiple of H; K and V multiples of 32;
        cu_seqlens needs B == 1; allow_neg_eigval needs use_beta_sigmoid_in_kernel; lower_bound needs rank-4 g;
        use_gate_in_kernel needs A_log and dt_bias; inplace_final_state needs initial_state and state_indices;
        num_accepted_tokens needs 2-D state_indices; 2-D state_indices has T columns; a slot index must lie inside
        the pool (a watcher-only kernel check from stage 3a).
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
        nb::arg("state_indices") = nb::none(),
        nb::arg("num_accepted_tokens") = nb::none(),
        nb::arg("inplace_final_state") = false,
        nb::arg("output_final_state") = false,
        nb::arg("output_per_token_state") = false,
        nb::arg("use_qk_l2norm_in_kernel") = false,
        nb::arg("use_gate_in_kernel") = false,
        nb::arg("A_log") = nb::none(),
        nb::arg("dt_bias") = nb::none(),
        nb::arg("lower_bound") = nb::none(),
        nb::arg("use_beta_sigmoid_in_kernel") = false,
        nb::arg("allow_neg_eigval") = false,
        nb::arg("cu_seqlens") = nb::none(),
        nb::arg("state_v_first") = false,
        nb::arg("output_tensor") = nb::none(),
        nb::arg("initial_state_block_idx") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("program_config") = nb::none());

    const auto* kda_doc =
        R"doc(
        Kimi Delta Attention recurrence (flash-linear-attention `fused_recurrent_kda`): fused_recurrent_gated_delta_rule
        with a per-key decay vector. g is [B, T, HV, K] and scales row k of the state by exp(g[k]) before the read;
        H == HV (no GQA); the gate is the softplus form g = -exp(A_log) * softplus(g_raw + dt_bias) with dt_bias
        [HV*K], or the bounded form g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias)) (Kimi K3,
        lower_bound = -5), both with use_gate_in_kernel. Every other argument, and the returned (o, state), are
        those of fused_recurrent_gated_delta_rule; the registered golden is the GDN golden with the rank-4 g.

        This build validates the call and rejects it: the KDA path arrives in stage 5 of the v2 plan.
        )doc";

    ttnn::bind_function<"fused_recurrent_kda", "ttnn.transformer.">(
        mod,
        kda_doc,
        &ttnn::transformer::fused_recurrent_kda,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("beta").noconvert(),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("state_indices") = nb::none(),
        nb::arg("num_accepted_tokens") = nb::none(),
        nb::arg("inplace_final_state") = false,
        nb::arg("output_final_state") = false,
        nb::arg("output_per_token_state") = false,
        nb::arg("use_qk_l2norm_in_kernel") = false,
        nb::arg("use_gate_in_kernel") = false,
        nb::arg("A_log") = nb::none(),
        nb::arg("dt_bias") = nb::none(),
        nb::arg("lower_bound") = nb::none(),
        nb::arg("use_beta_sigmoid_in_kernel") = false,
        nb::arg("allow_neg_eigval") = false,
        nb::arg("cu_seqlens") = nb::none(),
        nb::arg("state_v_first") = false,
        nb::arg("output_tensor") = nb::none(),
        nb::arg("initial_state_block_idx") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("program_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
