// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Fused GDN spec-verify step (M2 stage 1): one dispatch per GDN layer replaces the composite verify chain
// (conv window rebuild, depthwise causal conv + SiLU, l2norms, gates, T-step delta rule on a per-token state ring,
// gated RMSNorm, silu(z) gate). One core per (user u, value head h); the state stays in L1 across the T tokens.
// T = 1 with ring := rec_state and an identity ctrl page is the seed step (same kernel, same arithmetic).
#pragma once

#include <cstdint>

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
    uint32_t max_users = 0;  // ring/ctrl layout width M (0 = B); M >= B. Part of the program hash.
    uint32_t conv_kernel;  // K (4): window length Lw = K - 1 + T
    uint32_t qkvz_dim;     // column offset of the a|b block (= 2*Nk*Dk + 2*Nv*Dv)
    float scale;
    float l2_epsilon;
    float norm_epsilon;
    uint32_t hnew_depth;  // states buffered between compute and the ring writer (2 or 4)
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    // fast_mode: gdn_decode_step's fast_mode arithmetic. The T beta/decay gate pairs are computed in software fp32
    // on the reader RISC (reader_gdn_decode_step_conv_fast.cpp's exp/log1p/sigmoid, shared through
    // gdn_step_gate_math.hpp) and handed to the compute as ready scalar tiles, instead of the SFPU
    // sigmoid/softplus/exp of the plain kernels; the state add packs hnew from the same DST tile as hn (the full
    // fp32 sum, not a SrcA-truncated copy). Part of the program hash (default attribute hashing).
    bool fast_mode = false;
    // fp32_state (fast_mode only): gdn_decode_step's fp32_state arithmetic. The recurrent-state decay multiply and
    // the state add run on the SFPU in exact fp32 (state_in / dec / hd32 / hn32 / outer are UnpackToDest DFBs)
    // instead of the FPU, whose SrcA path truncates the fp32 state to ~tf32 every token. Costs two extra KV-tile
    // DFBs (hd32, hn32). Part of the program hash.
    bool fp32_state = false;
    // value_blocks (QWEN36_SPEC_STEP_VALUE_BLOCKS): split each (u, h)'s 128 state columns over up to this many cores
    // (tt-lab's 4-core GDN head split). Every column's recurrence is independent, so each core runs the T tokens on
    // its Vt / value_blocks column tiles with the unchanged per-tile arithmetic; the blocks of a head swap their final
    // pre-norm output tiles (exact fp32 bytes over the NoC) so each one runs the full-row gated RMSNorm as before and
    // writes only its own output columns. Outputs, ring and window are bit-identical to value_blocks = 1. The factory
    // takes the largest divisor of Vt <= value_blocks (<= 4) whose B*Nv*blocks cores fit the grid. 0 or 1 = one core
    // per (u, h) (the original program). Part of the program hash.
    uint32_t value_blocks = 1;
    // kernel_opts (QWEN36_SPEC_STEP_OPT): bitmask of bit-identical kernel restructurings (0 = the original program,
    // byte for byte). See GdnSpecStepKernelOpt; every bit keeps out, ring and window bit-identical to 0, except
    // kGdnSpecOptLazy (K4), which changes the arithmetic and the ring / ctrl contract. Part of the program hash.
    uint32_t kernel_opts = 0;
};

// gdn_spec_step kernel_opts bits (fast_mode + fp32_state only; ignored -- masked to 0 -- otherwise)
enum GdnSpecStepKernelOpt : uint32_t {
    // the next token's state decay (S * dec, exact fp32 SFPU) runs in the same DST acquire as this token's state add:
    // no hn32 DFB, no hn32 pack / UnpackToDest reload per state tile and token
    kGdnSpecOptFusedDecay = 1u << 0,
    // the writer RISC computes the T beta / decay gate pairs (the reader's software-fp32 math, bit for bit) into one
    // small entry; the compute fills its scalar tiles in DST from those bits (fill_tile_bitcast) instead of the
    // reader filling 2*T broadcast tiles with 1024 RISC-V stores each, and the reader pushes the initial state as soon
    // as it lands (the reader's gate block was the op's critical path: ~12.5 us per token)
    kGdnSpecOptGateScalars = 1u << 1,
    // the reader pushes src_in and the conv selectors before it builds the taps / norm weight row / row masks (needed
    // from phase A3 on), so phase A starts earlier
    kGdnSpecOptEarlySrc = 1u << 2,
    // the token-independent half of the per-token q l2-norm scale, rsqrt(q stats + eps) * scale, once before the token
    // loop; each token keeps only its row-mask multiply
    kGdnSpecOptHoistInv = 1u << 3,
    // value_blocks > 1 only: each of a head's VB blocks convolves 2*Kt/VB of the q/k chunks (instead of all 2*Kt) and
    // the blocks swap the exact fp32 conv tiles over the NoC (the acc_x exchange's handshake); each block reads,
    // rebuilds and writes back only its own chunks
    kGdnSpecOptQkShare = 1u << 4,
    // the conv's SFPU tail (tap sum + silu) on the tile half that holds the user's rows only (rows outside them are
    // zero in every conv operand and the one-hot placement maps whatever finite values they hold to the same +0)
    kGdnSpecOptA3Half = 1u << 5,
    // gate scalars with value_blocks > 1: the head's blocks split the T gate computations and swap the results
    kGdnSpecOptGscSplit = 1u << 6,
    // value_blocks > 1: each block normalizes / gates only its own output columns (the row stats stay full-row)
    kGdnSpecOptVsTail = 1u << 7,
    // K4 (QWEN36_SPEC_STEP_LAZY): chunked (WY) token phase + lazy rollback. NOT bit-identical (every other bit is) and
    // it CHANGES THE RING CONTRACT, so the caller must opt in on both sides:
    //   ring slot 0 block (u, h) = D, the state BEFORE the previous verify's T tokens (seeds keep writing slot 0);
    //   ring slot 1 + par (u, h) = the previous verify's stash: page j < Kt = K^T tile j (normalized keys, token
    //     columns r0 + t), page Kt + v = the WY deltas U tile v (token rows r0 + t), page Kt + Vt = the per-token decay
    //     exp(g_t) at element (r0 + t, 0) (column 0 only); this verify writes its own stash into slot 1 + (1 - par)
    //     (the parity double-buffer: no page is read and written in one launch);
    //   ctrl word 1 + u = mi of the previous verify (replay mi + 1 stashed tokens onto D), or any value >= T = FRESH
    //     (slot 0 was just seeded / restored: no replay, conv-window mi 0); the ring-block words only carry HOLD.
    // Per (u, h): S' = D replayed (exact-fp32 SFPU decay + one rank-(mi+1) FPU update) -> written back to slot 0; the T
    // new tokens' outputs from S' in WY form (U = (I + N)^-1 diag(beta)(V - diag(E) K S'), O = diag(E) Q S' +
    // (Q K^T o G) U), elementwise ops on the SFPU in exact fp32: one state read + one state write per step instead of
    // one read + T writes, and no serial per-token state passes. Needs fast_mode + fp32_state and 3 <= T <= 8;
    // implies kGdnSpecOptGateScalars (the writer computes the gate scalars and the WY coefficient tiles; GscSplit keeps
    // splitting the gate math over the value blocks) and ignores the token-loop bits (FusedDecay, HoistInv) and A3Half
    // (code size: the program fills the 70656 B kernel config buffer).
    kGdnSpecOptLazy = 1u << 8,
    // K4 only (x7 gdn-sfpu, QWEN36_SPEC_STEP_LAZY_SFPU): the gate transcendentals run on the compute SFPU instead of the
    // writer's software fp32. The writer hands the compute the raw a|b tile(s) and the head's dt_bias / -exp(A) bits;
    // the compute evaluates beta = sigmoid(b), dec = exp(-exp(A) softplus(a + dt_bias)) over the whole tile in the plain
    // (non-fast) kernel's SFPU op order and returns the two fp32 tiles; the writer picks each token's pair and builds the
    // WY coefficient tiles from them as before, except C (the replay decay), which the compute fills in DST from one
    // scalar. Every block evaluates all T tokens itself (no gate exchange: GscSplit is ignored). NOT bit-identical to K4
    // (SFPU instead of software exp / log1p); ignored without kGdnSpecOptLazy.
    kGdnSpecOptSfpuGate = 1u << 9,
    // gate scalars (bit 1 or K4): the writer's gate reads fetch only the face rows they use -- the user's T rows of the
    // a|b tile(s), tile 0's face row of dt_bias / -exp(A) holding the head, the replay rows of the decay page's column 0
    // -- instead of whole 2-4 KB pages that every core reads from the same DRAM bank. Bit-identical.
    kGdnSpecOptNarrowGateReads = 1u << 10,
    // x8 (with kGdnSpecOptSfpuGate): the compute's SFPU gate ops run on the ONE face that holds every value the writer
    // reads ((r0 + t, h) and (r0 + t, Nv + h)), instead of all four. Applied only when 2*Nv <= 16 (one a|b tile, both
    // columns in face column 0); ignored otherwise. Bit-identical to kGdnSpecOptSfpuGate.
    kGdnSpecOptSfpuGateFace = 1u << 11,
    // x8 (with kGdnSpecOptSfpuGate + kGdnSpecOptNarrowGateReads): the writer issues the ctrl-independent gate reads (a|b
    // rows, dt_bias / -exp(A) face rows) together with its ctrl read, ahead of the reader's bulk reads. Bit-identical.
    kGdnSpecOptEarlyGateReads = 1u << 12,
    kGdnSpecOptAll = kGdnSpecOptFusedDecay | kGdnSpecOptGateScalars | kGdnSpecOptEarlySrc | kGdnSpecOptHoistInv |
                     kGdnSpecOptQkShare | kGdnSpecOptA3Half | kGdnSpecOptGscSplit | kGdnSpecOptVsTail |
                     kGdnSpecOptLazy | kGdnSpecOptSfpuGate | kGdnSpecOptNarrowGateReads | kGdnSpecOptSfpuGateFace |
                     kGdnSpecOptEarlyGateReads,
};

struct GdnSpecStepInputs {
    Tensor
        qkvzab;  // [1, B*T <= R <= round_up(B*T,32), W >= qkvz_dim + 2*Nv] bf16 TILE, RAW rows u*T + t of [q|k|v|z|a|b]
    Tensor win_a;    // [B, Lw <= 32, C] bf16 TILE: conv window ping-pong pair; the op reads win[par] (rows 0..Lw-1 of
    Tensor win_b;    // user u = the previous window) and writes win[1 - par] (rows 0..Lw-1 = [E_prev[mi+1:mi+K]; new])
    Tensor ring;     // fp32 TILE, >= T*B*Nv blocks of [Dk, Dv]: block (t*B*Nv + u*Nv + h) = state after token t; the
                     // initial block per (u,h) comes from the ctrl page. rec_state [B, Nv, Dk, Dv] at T = 1 (seed)
    Tensor ctrl;     // uint32 ROW_MAJOR, one page [1, N >= 1 + B + B*Nv] (N*4 % 64 == 0): word 0 = window parity,
                     // words [1, 1+B) = mi[u], words [1+B, 1+B+B*Nv) = initial ring block per (u,h) or HOLD sentinel
                     // (HOLD must cover all Nv heads of a user; contents are data, not validated)
    Tensor taps;     // [1, K, C] bf16 TILE: conv taps, tap j in row j of tile column c (row-broadcast operand)
    Tensor dt_bias;  // [1, 1, Nv] fp32/bf16
    Tensor neg_exp_A;  // [1, 1, Nv] fp32/bf16
    Tensor weight;     // [Dv] bf16 gated-norm weight
};

}  // namespace ttnn::experimental::prim
