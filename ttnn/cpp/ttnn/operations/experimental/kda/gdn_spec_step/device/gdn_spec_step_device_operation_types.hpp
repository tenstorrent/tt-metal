// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Fused GDN speculative-verify step: one dispatch per GDN layer runs the conv window rebuild, the depthwise causal conv
// + SiLU, the l2norms, the gates, T gated delta-rule steps with the state after every token written out, the gated
// RMSNorm and the silu(z) gate. One core per (user u, value head h); the state stays in L1 across the T tokens.
#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct GdnSpecStepParams {
    uint32_t num_value_heads;
    uint32_t num_key_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t T;            // candidate tokens per user (rows u*T + t of qkvzab)
    uint32_t B;            // users
    uint32_t conv_kernel;  // K (4): window length Lw = K - 1 + T
    uint32_t qkvz_dim;     // column offset of the a|b block (= 2*Nk*Dk + 2*Nv*Dv)
    float scale;
    float l2_epsilon;
    float norm_epsilon;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    // fast_mode: gdn_decode_step's fast_mode arithmetic. The T beta/decay gate pairs are computed in software fp32 on
    // the reader (gdn_step_gate_math.hpp) and handed to the compute as ready scalar tiles, instead of the SFPU
    // sigmoid/softplus/exp; the state add packs hnew from the same DST tile as hn (the full fp32 sum).
    bool fast_mode = false;
    // fp32_state (fast_mode only): gdn_decode_step's fp32_state arithmetic. The recurrent-state decay multiply and
    // the state add run on the SFPU in exact fp32 (state_in / dec / hd32 / hn32 are UnpackToDest DFBs) instead of the
    // FPU, whose SrcA path truncates the fp32 state every token.
    bool fp32_state = false;
    // value_blocks: split each (u, h)'s Dv state columns over up to this many cores. A state column's recurrence never
    // reads another column, so each core runs the T tokens on its Vt / value_blocks column tiles with the unchanged
    // per-tile arithmetic; the blocks of a head swap their last-token pre-norm output tiles (exact fp32 bytes over the
    // NoC) so that each one runs the full-row gated RMSNorm. Outputs, states and window (or conv_hist history pages)
    // are bit-identical to value_blocks = 1. The factory takes the largest divisor of Vt <= value_blocks (<= 4) whose
    // B*Nv*blocks cores fit the grid; 0 or 1 = one core per (u, h).
    uint32_t value_blocks = 1;
    // kernel_opts: GdnSpecStepKernelOpt bits (fast_mode + fp32_state only). Each bit restructures the kernels and keeps
    // outputs, states and window (or conv_hist history pages) bit-identical to 0.
    uint32_t kernel_opts = 0;
};

// gdn_spec_step kernel_opts bits. Bits 0, 2, 8 and 9 are not assigned.
enum GdnSpecStepKernelOpt : uint32_t {
    // the writer computes the T beta / decay gate pairs (the reader's software-fp32 math, bit for bit) into one small
    // entry; the compute fills its scalar tiles in DST from those bits (fill_tile_bitcast) instead of the reader
    // filling 2*T broadcast tiles with 1024 RISC-V stores each, and the reader pushes the start state as soon as it
    // lands
    kGdnSpecOptGateScalars = 1u << 1,
    // the token-independent half of the per-token q l2-norm scale, rsqrt(q stats + eps) * scale, once before the token
    // loop; each token keeps only its row-mask multiply
    kGdnSpecOptHoistInv = 1u << 3,
    // value_blocks > 1: each of a head's VB blocks convolves 2*Kt/VB of the q/k chunks (instead of all 2*Kt) and the
    // blocks swap the exact fp32 conv tiles over the NoC; each block reads, rebuilds and writes back only its own
    // chunks (conv_hist: its own chunks' rows of the history pages)
    kGdnSpecOptQkShare = 1u << 4,
    // conv_kernel 4: the conv's SFPU tail (tap sum + silu) on the tile half that holds the user's rows only
    kGdnSpecOptA3Half = 1u << 5,
    // gate scalars with value_blocks > 1: the head's blocks split the T gate computations and swap the results
    kGdnSpecOptGscSplit = 1u << 6,
    // value_blocks > 1: each block normalizes / gates only its own output columns (the row statistics stay full-row)
    kGdnSpecOptVsTail = 1u << 7,
    // gate scalars: the writer's gate reads fetch only the face rows they use (the user's T rows of the a|b tile, tile
    // 0's face row of dt_bias / -exp(A) holding the head) instead of whole pages that every core reads from one bank
    kGdnSpecOptNarrowGateReads = 1u << 10,
    kGdnSpecOptAll = kGdnSpecOptGateScalars | kGdnSpecOptHoistInv | kGdnSpecOptQkShare | kGdnSpecOptA3Half |
                     kGdnSpecOptGscSplit | kGdnSpecOptVsTail | kGdnSpecOptNarrowGateReads,
};

struct GdnSpecStepInputs {
    // [1, B*T <= R <= round_up(B*T,32), W >= qkvz_dim + 2*Nv] bf16 TILE, raw rows u*T + t of [q|k|v|z|a|b]
    Tensor qkvzab;
    // [B, Lw <= 32, C] bf16 TILE: conv window ping-pong pair; the op reads win[par] (rows 0..Lw-1 of user u = the
    // previous window) and writes win[1 - par] (rows 0..Lw-1 = [E_prev[mi+1:mi+K]; new]). Absent with conv_hist.
    std::optional<Tensor> win_a;
    std::optional<Tensor> win_b;
    Tensor ring;     // fp32 TILE, >= T*B*Nv blocks of [Dk, Dv]: block (t*B*Nv + u*Nv + h) = state after token t; the
                     // start block per (u,h) comes from the ctrl page
    Tensor ctrl;     // uint32 ROW_MAJOR, one page [1, N >= 1 + B + B*Nv] (N*4 % 64 == 0): word 0 = window parity,
                     // words [1, 1+B) = mi[u], words [1+B, 1+B+B*Nv) = start ring block per (u,h) or HOLD sentinel
                     // (HOLD must cover all Nv heads of a user; contents are data, not validated)
    Tensor taps;     // [1, K, C] bf16 TILE: conv taps, tap j in row j of tile column c (row-broadcast operand)
    Tensor dt_bias;  // [1, 1, Nv] fp32/bf16
    Tensor neg_exp_A;  // [1, 1, Nv] fp32/bf16
    Tensor weight;     // [Dv] bf16 gated-norm weight
    // optional: [>= B, T*Nv] ROW_MAJOR (u)int32. Entry (u, t*Nv + h) is the first ring page (KV = Dk*Dv/1024 fp32
    // tiles) of value head h's state after token t of user u, so the per-token states can go to arbitrary pages of a
    // pool. Absent: ring block (t*B*Nv + u*Nv + h). With it, the ctrl page's start word of (u, h) is the first fp32
    // ring page of that head's start state (any page) instead of a block index.
    std::optional<Tensor> ring_page_idx;
    // optional second fp32 pool (ring_page_idx only): ring pages >= ring's page count (in ring_page_idx and in the ctrl
    // start words) address ring_pool2 at (page - that count), so one user's heads can span two buffers.
    std::optional<Tensor> ring_pool2;
    // optional bf16 TILE page pool (ring_page_idx only) holding the convolution history in gdn_decode_step's packed
    // format: 4 pages per value head, slot s >= 1 = history row s - 1 (slot 3 the newest), channel chunk c of the
    // head's [q | k | v] row in tile rows 2c and 2c + 1. It replaces the window pair (win_a / win_b absent): the
    // start history of (u, h) is slots 1..3 at ctrl word 1 + B + B*Nv + u*Nv + h (its slot-0 page), the history after
    // token t goes to slots 1..3 at ring_page_idx entry (u, T*Nv + t*Nv + h), and the ctrl mi words are not read.
    std::optional<Tensor> conv_hist;
};

}  // namespace ttnn::experimental::prim
