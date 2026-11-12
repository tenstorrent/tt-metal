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
        Fused GDN speculative-verify step, one core per (user u, value head h), state resident in L1 across the T
        tokens. For each user it runs T candidate tokens and writes the recurrent state after every token, so the
        caller can keep the state of whichever continuation it accepts.
        qkvzab [1, B*T <= R <= round_up(B*T, 32), W] BFLOAT16 TILE: raw projection rows u*T + t of
        [q | k | v | z | a | b] (before the convolution). 2*Nv <= 32 (a | b in one tile).
        win_a / win_b [B, Lw <= L <= 32, C] BFLOAT16 TILE or None (with conv_hist), Lw = K - 1 + T: conv window
        ping-pong pair. The op reads
        rows 0..Lw-1 of user u from win[par] (E_prev), forms W = [E_prev[u, mi+1 : mi+K] ; qkvzab rows u*T .. u*T+T-1
        (first C columns)], runs the depthwise causal conv (taps, kernel K) + SiLU over W, and writes W (rows 0..Lw-1,
        padded to an even row count with an exact zero) into win[1-par]. Rows t+1 .. t+K-1 of W are the convolution
        history after token t.
        ring FLOAT32 TILE (>= T*B*Nv blocks [Dk, Dv]): block ctrl[1+B+u*Nv+h] is the start state of (u,h); block
        (t*B*Nv + u*Nv + h) receives the state after token t (in place).
        ctrl UINT32 ROW_MAJOR one page [1, N] (N*4 % 64 == 0, N >= 1 + B + B*Nv): word 0 = par, words 1..B = mi[u],
        then the start ring block per (u,h); 0xFFFFFFFF = HOLD (no state writes for the head, window rows copied
        through). HOLD must be applied to ALL Nv heads of a user (the q/k window chunks are owned per key-head group).
        ctrl contents are data and are not validated: a wrong block index reads or writes the wrong block silently and
        mi >= T is clamped to T-1.
        taps [1, K, C] BFLOAT16 TILE (tap j in row j). dt_bias / neg_exp_A [1,1,Nv]; weight [Dv] BFLOAT16.
        Returns [1, R, Nv*Dv]: rows u*T + t valid, every other row (incl. the tile padding rows [B*T, R)) exactly 0.
        Keyword Args:
            fast_mode (bool): default False. gdn_decode_step's fast_mode arithmetic: the T beta/decay gate pairs are
                computed in software fp32 on the reader (sigmoid / softplus(threshold 20) / exp) instead of the
                SFPU, and the stored state is the full fp32 sum of the state add (not a SrcA-truncated copy).
            fp32_state (bool): default False, needs fast_mode. gdn_decode_step's fp32_state arithmetic: the
                recurrent-state decay and add run on the SFPU in exact fp32 (no per-token tf32 truncation).
            ring_page_idx (ttnn.Tensor, optional): ``[>= B, T*Nv]`` ROW_MAJOR INT32 or UINT32 device tensor. Entry
                ``(u, t*Nv + h)`` is the first FLOAT32 page of ``ring`` that receives value head ``h``'s state after
                token ``t`` of user ``u`` (KV = Dk*Dv/1024 pages, tile order as in ``[Dk, Dv]``), in place of ring
                block ``t*B*Nv + u*Nv + h``. ``ring`` may then be any FLOAT32 tiled page pool (for example a view of
                vLLM-owned cache blocks) of at least B*Nv blocks (a held head reads block u*Nv + h as a placeholder).
                In this mode the ctrl word ``1 + B + u*Nv + h`` is the first FLOAT32 page of that head's start state
                (any page, no block alignment), not a ring block index. Pages written for different
                (user, token, head) triples must not overlap; a (user, head)'s writes may overlap its own start
                pages. Read on device on every call, so a captured trace follows new pages when only the contents
                change. Default None.
            ring_pool2 (ttnn.Tensor, optional): with ``ring_page_idx`` only. A second FLOAT32 tiled pool: a page
                ``p >= P`` (P = page count of ``ring``) in ``ring_page_idx`` or in the ctrl start words addresses
                page ``p - P`` of ``ring_pool2``, so a user's heads can be split over two buffers (for example the K
                and V cache of a layer). A head's KV pages must lie in one pool.
            value_blocks (int): default 1. Split each (user, head)'s state columns over up to this many cores
                (<= 4; the largest divisor of value_dim / 32 whose B * Nv * blocks cores fit the grid is taken). Each
                core runs the recurrence on its column tiles; the blocks swap their pre-norm output tiles so the gated
                RMSNorm sees the full row. Bit-identical to value_blocks=1 (outputs, states, window rows or history
                pages).
            kernel_opts (int): default 0. Bitmask of kernel restructurings, fast_mode=True and fp32_state=True only;
                outputs, states and window rows or history pages stay bit-identical to kernel_opts=0. 2 = gate
                scalars (the writer
                computes the beta / decay gates, the compute fills its scalar tiles in DST); 8 = hoisted q scale
                (rsqrt(q stats + eps) * scale once, not per token); 16 = q/k share (value_blocks > 1: each block
                convolves 2*Kt/VB of the q/k chunks and the blocks swap the conv tiles over the NoC); 32 = the conv's
                SFPU tail on the tile half holding the user's rows (conv_kernel 4); 64 = gate split (with 2 and
                value_blocks > 1: the head's blocks split the gate math); 128 = each value block normalizes / gates
                only its own output columns; 1024 = narrow gate reads (with 2: the gate reads fetch only the face rows
                they use). Bits that need value_blocks > 1 have no effect without it. All bits: 1274.
            hnew_depth (int): must be 2. max_users (int): must be 0 or B.
            conv_hist (ttnn.Tensor, optional): with ``ring_page_idx`` only, and ``conv_kernel`` 4. A BFLOAT16 TILE page
                pool (for example the ``conv_hist`` view of a vLLM-owned cache block that gdn_decode_step reads) that
                holds the convolution history in gdn_decode_step's packed format: 4 pages per value head, slot s >= 1 =
                history row s - 1 (slot 3 the newest), channel chunk c of the head's [q | k | v] row in tile rows 2c
                and 2c + 1. It replaces the window pair: ``win_a`` and ``win_b`` must be None, and the ctrl mi words
                are not read. The ctrl page grows by B*Nv words: word
                ``1 + B + B*Nv + u*Nv + h`` is the slot-0 page of (u, h)'s start history; ``ring_page_idx`` grows to
                ``[>= B, 2*T*Nv]``: entry ``(u, T*Nv + t*Nv + h)`` is the slot-0 page that takes (u, h)'s history
                after token t (slots 1..3 are written, slot 0 is not; rows 2c, 2c + 1 hold chunk c, the other rows 0).
                Every block then holds a complete state that gdn_decode_step can continue. Default None.
            With fast_mode=True, fp32_state=True a T-token call is bit-identical (outputs, states, window rows or
            history pages) to T chained gdn_decode_step(..., fast_mode=True, fp32_state=True) calls at B = 1.
        )doc",
        &ttnn::experimental::kda::gdn_spec_step,
        nb::arg("qkvzab").noconvert(),
        nb::arg("win_a").none(),
        nb::arg("win_b").none(),
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
        nb::arg("value_blocks") = 1,
        nb::arg("kernel_opts") = 0,
        nb::arg("ring_page_idx") = nb::none(),
        nb::arg("ring_pool2") = nb::none(),
        nb::arg("conv_hist") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::gdn_spec_step::detail
