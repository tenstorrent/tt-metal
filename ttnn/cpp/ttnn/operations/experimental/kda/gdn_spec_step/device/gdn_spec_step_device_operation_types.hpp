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
    // byte for byte). See GdnSpecStepKernelOpt; every bit keeps out, ring and window bit-identical to 0. Part of the
    // program hash.
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
    // gate scalars (bit 1): the writer's gate reads fetch only the face rows they use -- the user's T rows of the a|b
    // tile(s), tile 0's face row of dt_bias / -exp(A) holding the head -- instead of whole 2-4 KB pages that every core
    // reads from the same DRAM bank. Bit-identical. (Bits 8 / 9 are K4 / SFPU gates on the x7 branch; not here.)
    kGdnSpecOptNarrowGateReads = 1u << 10,
    // gov gdn-v2: the reader's src / z reads fetch only the rows the op uses -- window rows [0, Lw) and the user's T
    // rows [r0, r0 + T) of the raw qkv / z tiles (every user's core reads the same qkv / z pages) -- into slots that
    // are zero-filled first, so the rows the one-hot gathers / row masks multiply by 0 hold finite zeros. Bit-identical.
    // (Bits 8 / 9 / 11 / 12 are taken on other branches.)
    kGdnSpecOptNarrowSrc = 1u << 13,
    // gov gdn-v2: the SFPU tails whose rows outside the user's T rows never reach an output -- the conv's tap sum +
    // silu (with A3_HALF: same tile half) and the silu(z) gate operand -- run only on the SFPU row pairs holding the
    // user's rows (T even). Rows outside keep finite values that the one-hot placement / per-row tail ignore. Bit-identical.
    kGdnSpecOptRowsSfpu = 1u << 14,
    // gov gdn-v2 (with NARROW_GR): the writer's a|b gate read is one contiguous read of the user's T rows per face
    // holding a[h] / b[Nv + h] (1 request per core at Nv <= 16 instead of 2 * T): every core reads the same page.
    kGdnSpecOptNarrowGr2 = 1u << 15,
    // gov gdn-v2 (with ROWS_SFPU): the norms' SFPU tails (k / q l2 rsqrt + scale, the output rmsnorm's scale + rsqrt) on
    // the user's row pairs; every consumer masks / multiplies the other rows to the same zeros. Bit-identical.
    kGdnSpecOptRowsNorm = 1u << 16,
    // gov gdn-v2 (gate scalars): the writer's software-fp32 gate math with fast bit-exact IEEE multiply / add (normal
    // operand fast path, the compiler's soft-float otherwise), op for op. Bit-identical.
    kGdnSpecOptFastSoftfloat = 1u << 17,
    kGdnSpecOptAll = kGdnSpecOptFusedDecay | kGdnSpecOptGateScalars | kGdnSpecOptEarlySrc | kGdnSpecOptHoistInv |
                     kGdnSpecOptQkShare | kGdnSpecOptA3Half | kGdnSpecOptGscSplit | kGdnSpecOptVsTail |
                     kGdnSpecOptNarrowGateReads | kGdnSpecOptNarrowSrc | kGdnSpecOptRowsSfpu |
                     kGdnSpecOptNarrowGr2 | kGdnSpecOptRowsNorm | kGdnSpecOptFastSoftfloat,
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
