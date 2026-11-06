// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fused_recurrent_gated_delta_rule_nanobind.hpp"

#include "fused_recurrent_gated_delta_rule.hpp"
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
        per core and accepts only the defaults.

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
        state carried on-core across the T tokens. One Tensix core per (batch row, value head) in this build.

        Per token t and value head hv, which reads key head hv // (HV // H), with S the fp32 state [K, V]:

            q̂ = q / sqrt(sum(q²) + ε), k̂ = k / sqrt(sum(k²) + ε)   use_qk_l2norm_in_kernel; ε = 1e-6 INSIDE the root;
                                                                   otherwise q̂ = q and k̂ = k (the caller normalised)
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
            v (ttnn.Tensor): [B, T, HV, V], the dtype of q. HV is a multiple of H.
            g (ttnn.Tensor): [B, T, HV] log-space decay, g ≤ 0 (exp(g) is applied in the op), bf16 or fp32.
            beta (ttnn.Tensor): [B, T, HV] post-sigmoid write strength, bf16 or fp32.

        Keyword Args:
            scale (float, optional): multiplies q after the optional norm. Default K**-0.5.
            initial_state (ttnn.Tensor, optional): [Bs, HV, K, V] fp32 TILE, Bs >= B: rows [0:B] are the state
                before token 0 of each batch row, rows [B:Bs] are not read (the leading-row form of a wider decode
                bucket); zeros when absent. In ring mode the [T*B*HV, K, V] ring (see initial_state_block_idx).
            inplace_final_state (bool): write the state after the last token back into rows [0:B] of initial_state
                and return that tensor (the same buffer, not a copy). Requires initial_state; exclusive with
                output_per_token_state and with ring mode.
            output_final_state (bool): return the state after the last token, [B, HV, K, V].
            output_per_token_state (bool): return the state after every token, [B, T, HV, K, V]; overrides
                output_final_state.
            use_qk_l2norm_in_kernel (bool): normalise q and k over K inside the op (ε inside the root), before the
                scale.
            output_tensor (ttnn.Tensor, optional): preallocated o, [B, T, HV, V] in the input dtype, TILE; o is
                written into it and it is returned.
            initial_state_block_idx (ttnn.Tensor, optional): ring mode, the deferred per-head initial-state select of
                the Qwen3.6 verify path. Head h starts from ring block idx[h] of the [T*BH, K, V] initial_state
                instead of block h and writes its per-token states back in place; the returned state is that same
                tensor. [BH] or [1, BH] uint32/int32, ROW_MAJOR, interleaved on device; BH = B*HV, h = b*HV + hv.
                Requires output_per_token_state=True and an fp32 TILE initial_state of exact shape [T*BH, K, V].
                Caller contract: idx[h] % BH == h for every h.
            memory_config (ttnn.MemoryConfig, optional): placement of the outputs; DRAM interleaved by default.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): HiFi4, fp32 accumulation and exact
                mode by default; fp32 accumulation is required.
            program_config (ttnn.transformer.FusedRecurrentGdnProgramConfig, optional): the geometry; absent, the
                op chooses. This build accepts only the defaults (one whole head per core).

        Returns:
            tuple[ttnn.Tensor, Optional[ttnn.Tensor]]: o [B, T, HV, V] in the input dtype, row t the read of the
            state after token t's own write; state None, [B, HV, K, V] (final), [B, T, HV, K, V] (per token), the
            initial_state tensor (in place), or the ring tensor itself (ring mode).

        What this build serves: the gated-DeltaNet decode (T = 1) and speculative verify (T = K + 1) of the
        Qwen3.5 / 3.6 / 3.8 layers on one chip or a tensor-parallel slice, straight from the layer's own tensors:
        the GQA key-head mapping, the norm, the scale, the bf16 widening and rounding, the zero state and the state
        and output layouts (in place, per token, the ring) are the kernels' work, so one call is one program with
        common runtime arguments and nothing else is launched. The per-head state pool and the in-kernel gates
        are later keyword additions.

        Limits of this build: B*HV must fit the compute grid (one core per head); T is at most 32 per call;
        the inputs are TILE. Every combination rule is checked in a fixed order, so a wrong call fails on its own
        message: HV a multiple of H; K and V multiples of 32; q, k and v of one dtype; initial_state [Bs >= B, HV, K,
        V] fp32; inplace_final_state needs initial_state; output_tensor of o's shape in the input dtype; ring mode
        needs output_per_token_state. The indexed state pool and the in-kernel gate options are not in this
        signature; they arrive as keyword additions, so a call written against this signature keeps working.
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
        nb::arg("inplace_final_state") = false,
        nb::arg("output_final_state") = false,
        nb::arg("output_per_token_state") = false,
        nb::arg("use_qk_l2norm_in_kernel") = false,
        nb::arg("output_tensor") = nb::none(),
        nb::arg("initial_state_block_idx") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("program_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
