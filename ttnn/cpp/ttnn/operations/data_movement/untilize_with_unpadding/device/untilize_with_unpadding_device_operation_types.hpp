// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::untilize_with_unpadding {

struct operation_attributes_t {
    ttnn::Shape output_tensor_end{};
    tt::tt_metal::MemoryConfig output_mem_config;
    bool use_multicore = false;
    bool use_pack_untilize = false;
    bool fp32_dest_acc_en = false;
    bool enough_space_width = false;
    bool enough_space_height = false;
    std::optional<CoreRangeSet> sub_core_grids = std::nullopt;
};

struct tensor_args_t {
    Tensor input_tensor;
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttnn::operations::data_movement::untilize_with_unpadding
