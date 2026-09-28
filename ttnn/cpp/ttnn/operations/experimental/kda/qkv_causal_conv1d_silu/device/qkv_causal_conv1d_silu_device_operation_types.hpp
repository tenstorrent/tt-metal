// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct QkvCausalConv1dSiluParams {
    uint32_t sequence;
    uint32_t q_width;
    uint32_t k_width;
    uint32_t v_width;
    // ROW_MAJOR input: channels per work item. TILE input: 32 * B, where B is the tiled-path block size.
    uint32_t channel_chunk_size;
    // TILE input only: also return new_state = TILE [1,3,Q+K+V] that holds x[T-3..T-1].
    bool return_conv_state = false;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct QkvCausalConv1dSiluInputs {
    Tensor input;
    // Required for ROW_MAJOR input. Optional for TILE input: std::nullopt means three zero rows.
    std::optional<Tensor> history;
    Tensor tap0;
    Tensor tap1;
    Tensor tap2;
    Tensor tap3;
};

}  // namespace ttnn::experimental::prim
