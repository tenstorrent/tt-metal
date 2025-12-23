// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::tilize_with_val_padding {

struct operation_attributes_t {
    ttnn::Shape output_padded_shape{};
    tt::tt_metal::PadValue pad_value;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{tt::tt_metal::DataType::INVALID};
    bool use_multicore{};
    bool enough_space_width{};
    bool enough_space_height{};
    std::optional<CoreRangeSet> sub_core_grids;
};

struct tensor_args_t {
    Tensor input_tensor;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::data_movement::tilize_with_val_padding
