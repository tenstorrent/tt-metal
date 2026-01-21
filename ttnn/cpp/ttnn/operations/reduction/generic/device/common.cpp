// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common.hpp"

namespace ttnn::prim {
tt::tt_metal::ReduceOpParallelizationStrategy get_parallelization_strategy(
    const tt::tt_metal::Tensor& input_tensor, tt::tt_metal::ReduceOpDim reduce_dim) {
    uint32_t num_tiles = input_tensor.physical_volume() / tt::constants::TILE_HW;
    if (reduce_dim == tt::tt_metal::ReduceOpDim::H) {
        return tt::tt_metal::ReduceOpParallelizationStrategy::MULTI_CORE_H;
    }
    if (reduce_dim == tt::tt_metal::ReduceOpDim::W) {
        return tt::tt_metal::ReduceOpParallelizationStrategy::MULTI_CORE_W;
    }
    if (reduce_dim == tt::tt_metal::ReduceOpDim::HW) {
        if (num_tiles > 1) {
            return tt::tt_metal::ReduceOpParallelizationStrategy::MULTI_CORE_HW;
        }
        return tt::tt_metal::ReduceOpParallelizationStrategy::SINGLE_CORE_HW;
    }
    TT_THROW("Unsupported reduce dim");
}
}  // namespace ttnn::prim
