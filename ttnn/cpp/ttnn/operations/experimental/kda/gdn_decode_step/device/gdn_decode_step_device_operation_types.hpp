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
    // fused-conv mode, optional: [2*Nv, >= B] ROW_MAJOR (u)int32. Column b, row h holds the first state page of value
    // head h of batch row b (KV fp32 pages), row Nv + h its first conv_hist page (4 bf16 pages), so `state` and
    // `conv_hist` can be page pools. Absent: row b owns state pages [b*Nv*KV, (b+1)*Nv*KV) and conv_hist pages
    // [b*Nv*4, (b+1)*Nv*4). With it, the packed history holds channel chunk c in both tile rows 2c and 2c + 1 (without
    // it: row 2c + (b & 1) only), so a request's pages stay valid when it moves to a batch row of the other parity.
    std::optional<Tensor> state_page_idx;
    // optional second pools (state_page_idx only): state index values >= state's page count address state_pool2 at
    // (value - that count); conv_hist likewise with conv_hist_pool2. Lets one request's heads span two buffers.
    std::optional<Tensor> state_pool2;      // fp32 TILE
    std::optional<Tensor> conv_hist_pool2;  // bf16 TILE
};

}  // namespace ttnn::experimental::prim
