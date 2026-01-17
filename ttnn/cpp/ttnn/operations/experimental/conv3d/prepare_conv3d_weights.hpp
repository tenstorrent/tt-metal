#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <optional>

namespace ttnn::operations::experimental::conv3d {

struct Conv3dWeightsBiasPrepConfig {
    uint32_t groups = 1;
    uint32_t iC_per_group_block = 32;
};

ttnn::Tensor prepare_conv_weights(
    const ttnn::Tensor& weight_tensor, const Conv3dWeightsBiasPrepConfig& config, MeshDevice* device);

}  // namespace ttnn::operations::experimental::conv3d
