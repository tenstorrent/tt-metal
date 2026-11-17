// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct GdnDecodeStepParams {
    uint32_t num_value_heads;
    uint32_t num_key_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    float scale;
    float l2_epsilon;
    float norm_epsilon;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    // fused-conv mode: qkv is the full projection row [q|k|v|z|a|b]; conv + gates computed in-kernel
    bool fuse_conv = false;
    uint32_t qkvz_dim = 0;  // column offset of the a|b block (= 2*Nk*Dk + 2*Nv*Dv)
    // fast_mode (fused-conv mode only): the *_conv_fast reader/compute kernels: reads ordered by first use, unrolled
    // scalar-tile fills, beta/decay gates computed on the reader in fp32, late DFB waits, and a dual-output state add
    // (the state is stored as the full fp32 sum instead of a TF32-truncated copy). o is unchanged; the state differs
    // in the last bits. Part of the program hash (default attribute hashing).
    bool fast_mode = false;
    // fp32_state (fast_mode only): keep the recurrent-state decay multiply and state add in exact fp32 on the SFPU
    // (UnpackToDest) instead of the FPU (SrcA, which truncates the fp32 state every token). Costs one extra KV-tile
    // DFB. Part of the program hash (default attribute hashing).
    bool fp32_state = false;
    // row_pos given: rows at a position < 0 keep their state and conv history (an idle row may hold a paused request)
    bool skip_idle_rows = false;
};

struct GdnDecodeStepInputs {
    Tensor qkv;     // bf16; plain: [1, 1, 2*Nk*Dk + Nv*Dv] post conv+silu; fused-conv: [1, B <= 32, W] [q|k|v|z|a|b]
    Tensor beta;    // fp32/bf16; plain: [1, 1, Nv]; fused-conv: dt_bias, one row of Nv values
    Tensor g;       // fp32/bf16; plain: [1, 1, Nv] log decay; fused-conv: -exp(A_log), one row of Nv values
    Tensor state;   // [Bmax, Nv, Dk, Dv] fp32 (Bmax = 1 plain, >= B fused-conv), rows 0..B-1 updated in place
    Tensor weight;  // bf16 gated-norm weight, one row of Dv values
    // fused-conv: [Bmax, Nv, 4, 32, 32] bf16, user u's chunk c in row 2c + (u & 1), slot 3 newest; shifted in place
    std::optional<Tensor> conv_hist;
    // fused-conv mode: packed taps [Nv, 4, 32, 32] bf16, chunk c in rows 2c and 2c + 1, tap 0 oldest
    std::optional<Tensor> conv_taps;
    std::optional<Tensor> row_pos;  // fused-conv: [B] int32 RM decode positions, -1 = idle row
};

}  // namespace ttnn::experimental::prim
