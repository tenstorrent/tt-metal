// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/pad/pad.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {
    ttnn::Tensor pad_to_tile_vol(uint8_t queue_id,
                                 const ttnn::Tensor& tensor,
                                 const float value,
                                 const bool use_multicore,
                                 const std::optional<MemoryConfig>& memory_config) {
        auto logical_shape = tensor.get_logical_shape();
        auto padded_shape = tensor.get_padded_shape();
        auto rank = tensor.get_shape().rank();
        if (padded_shape.volume() % tt::constants::TILE_HW != 0) {
            TT_ASSERT(rank >= 2, "rank of tensor to pad to tile must be at least 2.");

            auto padded_height = tt::round_up(padded_shape[-2], tt::constants::TILE_HEIGHT);
            auto padded_width = tt::round_up(padded_shape[-1], tt::constants::TILE_WIDTH);
            uint32_t num_non_hw_dims = rank - 2u;
            auto padding_vec = std::vector<std::pair<uint32_t, uint32_t>>(num_non_hw_dims, {0,0});
            padding_vec.reserve(rank);
            padding_vec.emplace_back(0, padded_height - padded_shape[-2]);
            padding_vec.emplace_back(0, padded_width - padded_shape[-1]);

            constexpr bool pad_use_multicore = true;
            auto padded_output = ttnn::pad(queue_id,
                                            tensor,
                                            padding_vec,
                                            value,
                                            use_multicore,
                                            memory_config);
            return padded_output;
        }
        return tensor;
    }
    uint32_t wrap_index(int index, int size) {
        return index < 0 ? size + index : index;
    }
}
}
}
