// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
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
    uint32_t channel_chunk_size;
    uint32_t sequence_parallel_axis;
    // First channel of the convolution within the input's last dimension.
    uint32_t input_column_offset;
    // Layout of history, predecessor_carry and the replacement carry: token-paged [1, history_rows, Q+K+V] when
    // empty, else channel-paged [1, (Q+K+V) / P, history_rows * P] with P = page_channels channels per page.
    std::optional<uint32_t> page_channels;
    // Fabric history exchange only: L1 addresses of its global semaphores, entry, predecessor arrival, carry arrival
    // and (SP-sharded history slab) old-tail arrival.
    std::array<uint32_t, 4> exchange_semaphore_addresses{};
    uint32_t slab_index_scale = 1;
    uint32_t slab_index_offset = 0;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct QkvCausalConv1dSiluInputs {
    Tensor input;
    Tensor history;
    Tensor tap0;
    Tensor tap1;
    Tensor tap2;
    Tensor tap3;

    Tensor actual_start;
    // Exactly one of predecessor_carry and exchange_buffer supplies the predecessor's rows. exchange_buffer is a
    // persistent [1, qkv_conv::exchange_rows, Q+K+V] ROW_MAJOR DRAM tensor the operation fills itself over the fabric;
    // with it the operation also returns the replacement carry.
    std::optional<Tensor> predecessor_carry;
    std::optional<Tensor> exchange_buffer;
    std::optional<Tensor> actual_end;
    // Exchange only: history is a slab whose entry is read (zeros at actual_start == 0) and replaced in place. An entry
    // holds every channel (replicated over SP), or SP-sharded: per Q/K/V branch, SP rank r holds the branch's channel
    // slice [r * W_b / SP, (r + 1) * W_b / SP), branches in order (channel-paged only).
    std::optional<Tensor> slab_index;
};

}  // namespace ttnn::experimental::prim
