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
