// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "gdn_spec_step_nanobind.hpp"

#include "gdn_spec_step.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::gdn_spec_step::detail {

void bind_gdn_spec_step(nb::module_& mod) {
    ttnn::bind_function<"gdn_spec_step", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Fused GDN spec-verify step, one core per (user u, value head h), state resident in L1 across the T tokens.
        qkvzab [1, B*T <= R <= round_up(B*T, 32), W] BFLOAT16 TILE: RAW projection rows u*T + t of
        [q | k | v | z | a | b] (pre-conv).
        win_a / win_b [B, Lw <= 32, C] BFLOAT16 TILE: conv window ping-pong pair. The op reads rows 0..Lw-1 of user u
        from win[par] (E_prev), forms W = [E_prev[u, mi+1 : mi+K] ; qkvzab rows u*T .. u*T+T-1 (first C columns)],
        runs the depthwise causal conv (taps, kernel K) + SiLU over W, and writes W (rows 0..Lw-1, padded to an even
        row count with an exact zero) into win[1-par]. q/k chunks are written by heads with h % (Nv/Nk) == 0 only.
        ring FLOAT32 TILE (>= T*B*Nv blocks [Dk, Dv]): block ctrl[1+B+u*Nv+h] is the initial state of (u,h); block
        (t*B*Nv + u*Nv + h) receives the state after token t (in place). At T = 1, ring = rec_state [B, Nv, Dk, Dv] with
        an identity ctrl page is the seed step.
        ctrl UINT32 ROW_MAJOR one page [1, N] (N*4 % 64 == 0, N >= 1 + B + B*Nv): word 0 = par, words 1..B = mi[u],
        then the ring block per (u,h); 0xFFFFFFFF = HOLD (no ring writes for the head, window rows copied through).
        HOLD must be applied to ALL Nv heads of a user (the q/k window chunks are owned per key-head group). ctrl
        contents are data and are not validated: a wrong block index reads/writes the wrong ring block silently and
        mi >= T is clamped to T-1.
        taps [1, K, C] BFLOAT16 TILE (tap j in row j). dt_bias / neg_exp_A [1,1,Nv]; weight [Dv] BFLOAT16.
        Returns [1, R, Nv*Dv]: rows u*T + t valid, every other row (incl. the tile padding rows [B*T, R)) exactly 0.
        Keyword Args:
            max_users (int): default 0 (= B). Ring/ctrl layout width M >= B: ctrl blocks start at word 1+M, ring token
                stride is M*Nv blocks, ring/ctrl/win may be M wide while only users 0..B-1 are computed.
            fast_mode (bool): default False. gdn_decode_step's fast_mode arithmetic: the T beta/decay gate pairs are
                computed in software fp32 on the reader (sigmoid / softplus(threshold 20) / exp) instead of the
                SFPU, and the stored state is the full fp32 sum of the state add (not a SrcA-truncated copy).
            fp32_state (bool): default False, needs fast_mode. gdn_decode_step's fp32_state arithmetic: the
                recurrent-state decay and add run on the SFPU in exact fp32 (no per-token tf32 truncation).
            num_slices (int): default 1. Splits every (user, value head) over NV in {1, 2} cores by value tile columns;
                each core keeps Kt x Vt/NV state tiles. Output, ring and window are bit-identical to num_slices=1.
                Needs B*Nv*NV <= grid cores and Vt % NV == 0.
            With fast_mode=True, fp32_state=True a T = 1 call is bit-identical (out, state, window) to
            gdn_decode_step(..., fast_mode=True, fp32_state=True), and a T-step call equals T chained T = 1 calls.
        )doc",
        &ttnn::experimental::kda::gdn_spec_step,
        nb::arg("qkvzab").noconvert(),
        nb::arg("win_a").noconvert(),
        nb::arg("win_b").noconvert(),
        nb::arg("ring").noconvert(),
        nb::arg("ctrl").noconvert(),
        nb::arg("taps").noconvert(),
        nb::arg("dt_bias").noconvert(),
        nb::arg("neg_exp_A").noconvert(),
        nb::arg("weight").noconvert(),
        nb::arg("num_value_heads"),
        nb::arg("num_key_heads"),
        nb::arg("key_dim"),
        nb::arg("value_dim"),
        nb::arg("T"),
        nb::arg("B"),
        nb::arg("qkvz_dim"),
        nb::kw_only(),
        nb::arg("conv_kernel") = 4,
        nb::arg("scale") = nb::none(),
        nb::arg("l2_epsilon") = 1e-6f,
        nb::arg("norm_epsilon") = 1e-6f,
        nb::arg("hnew_depth") = 2,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_dtype") = ttnn::DataType::BFLOAT16,
        nb::arg("fast_mode") = false,
        nb::arg("fp32_state") = false,
        nb::arg("max_users") = 0,
        nb::arg("num_slices") = 1);
}

}  // namespace ttnn::operations::experimental::kda::gdn_spec_step::detail
