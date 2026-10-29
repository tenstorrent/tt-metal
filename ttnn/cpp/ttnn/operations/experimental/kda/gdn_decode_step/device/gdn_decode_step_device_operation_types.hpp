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
    // value_blocks (QWEN36_GDN_STEP_VALUE_BLOCKS; fused + fast_mode only): split each (head, user group)'s 128 state
    // columns over up to this many cores (tt-lab's 4-core GDN head split, as gdn_spec_step's value_blocks). Each core
    // runs the unchanged per-tile arithmetic on its Vt / value_blocks column tiles; the blocks of a head swap their
    // pre-norm output tiles (exact fp32 bytes, unicast NoC writes + ready/valid semaphores) so each one runs the
    // full-row gated RMSNorm as before and writes only its own output columns. Out, state and history are
    // bit-identical to value_blocks = 1. The factory takes the largest divisor of Vt <= value_blocks (<= 4) whose cores
    // fit the grid. 0 or 1 = the original program. Part of the program hash (default attribute hashing).
    uint32_t value_blocks = 1;
};

struct GdnDecodeStepInputs {
    Tensor qkv;     // [1, 1, 2*Nk*Dk + Nv*Dv] bf16, post conv+silu
    Tensor beta;    // [1, 1, Nv] fp32/bf16
    Tensor g;       // [1, 1, Nv] fp32/bf16 (log decay)
    Tensor state;   // [1, Nv, Dk, Dv] fp32, updated in place
    Tensor weight;  // [Dv] bf16 gated-norm weight
    std::optional<Tensor>
        conv_hist;  // fused-conv mode: packed history [Nv, 4, 32, 32] bf16 (slot 3 newest), shifted in place
    std::optional<Tensor> conv_taps;  // fused-conv mode: packed taps [Nv, 4, 32, 32] bf16 (row c = channel chunk c)
};

}  // namespace ttnn::experimental::prim
