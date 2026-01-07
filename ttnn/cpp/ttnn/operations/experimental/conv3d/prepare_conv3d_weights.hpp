#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <optional>

namespace ttnn::operations::experimental::conv3d {

struct Conv3dWeightsBiasPrepConfig {
    uint32_t groups = 1;
};

ttnn::Tensor prepare_conv_weights(
    const ttnn::Tensor& weight_tensor, const Conv3dWeightsBiasPrepConfig& config, MeshDevice* device);

}  // namespace ttnn::operations::experimental::conv3d
