#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
// #include <tt-metalium/operation.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "device/conv3d_device_operation.hpp"

namespace ttnn::operations::conv {
namespace conv3d {

struct ExecuteConv3d {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const Conv3dConfig& config,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const Conv3dConfig& config,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace conv3d

}  // namespace ttnn::operations::conv
namespace ttnn {
constexpr auto conv3d =
    ttnn::register_operation_with_auto_launch_op<"ttnn::conv3d", ttnn::operations::conv::conv3d::ExecuteConv3d>();

}
