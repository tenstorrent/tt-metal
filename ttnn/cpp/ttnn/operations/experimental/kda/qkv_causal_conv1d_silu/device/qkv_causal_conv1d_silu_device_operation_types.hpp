// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::kda {

// How `history`, the predecessor carry and the returned carry store the three rows preceding the input.
enum class QkvCausalConv1dSiluHistoryLayout : uint8_t {
    // [1, 3, Q+K+V]: page r is history row r.
    ROWS,
    // [1, (Q+K+V)/64, 3*64]: page c holds the three rows of channels [64c, 64c+64), row r at element 64r.
    SEGMENTS,
};

}  // namespace ttnn::experimental::kda

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
    // Selects the history-row and segment-page read and write paths in the reader and exchange kernels.
    ttnn::experimental::kda::QkvCausalConv1dSiluHistoryLayout history_layout;
    // Fabric history exchange only: L1 addresses of its global semaphores, entry, predecessor arrival, carry arrival.
    std::array<uint32_t, 3> exchange_semaphore_addresses{};
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
};

}  // namespace ttnn::experimental::prim
