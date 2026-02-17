// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "tt-metalium/work_split.hpp"

namespace ttnn::experimental::prim {

struct RingMatmulConfig {
    uint32_t in0_block_w{};
    uint32_t out_subblock_h{};
    uint32_t out_subblock_w{};
    uint32_t per_core_M{};
    uint32_t per_core_N{};
    uint32_t num_global_cb_receivers{1};

    tt::tt_metal::CoreCoord compute_with_storage_grid_size = {0, 0};
    tt::tt_metal::CoreRangeSet hop_cores;
};

struct RingMatmulParams {
    std::optional<RingMatmulConfig> config;
    std::optional<operations::unary::UnaryWithParam> fused_activation;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;

    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt;
    bool untilize_out = false;
    std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer> global_cb = std::nullopt;
};

struct RingMatmulInputs {
    Tensor input_tensor_a;
    std::vector<Tensor> input_tensors_b;
    std::optional<Tensor> bias_tensor;
};

}  // namespace ttnn::experimental::prim
