// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/ring_matmul_device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::ring_matmul {

struct RingMatmulOperation {
    static Tensor invoke(
        const Tensor& input_tensor,
        const Tensor& weight_tensor,
        std::optional<unary::UnaryWithParam> fused_activation = std::nullopt,
        const std::optional<RingMatmulConfig>& config = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> dtype = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const CoreRangeSet& hop_cores = CoreRangeSet{},
        const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb = std::nullopt,
        uint32_t num_global_cb_receivers = 1,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt,
        std::optional<CoreRangeSet> restricted_cores = std::nullopt,
        bool untilize_out = false) {
        return ttnn::prim::ring_matmul(
            input_tensor,
            weight_tensor,
            fused_activation,
            config,
            memory_config,
            dtype,
            compute_kernel_config,
            hop_cores,
            global_cb,
            num_global_cb_receivers,
            sub_device_id,
            restricted_cores,
            untilize_out);
    }
};

}  // namespace ttnn::operations::experimental::ring_matmul

namespace ttnn::experimental {
constexpr auto ring_matmul = ttnn::
    register_operation<"ttnn::experimental::ring_matmul", operations::experimental::ring_matmul::RingMatmulOperation>();
}  // namespace ttnn::experimental
