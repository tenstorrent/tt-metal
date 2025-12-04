// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::untilize_with_unpadding {

struct operation_attributes_t {
    ttnn::Shape output_tensor_end;
    tt::tt_metal::MemoryConfig output_mem_config;
    bool use_multicore;
    bool use_pack_untilize;
    bool fp32_dest_acc_en;
    bool enough_space_width;
    bool enough_space_height;
};

struct tensor_args_t {
    const Tensor& input_tensor;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::data_movement::untilize_with_unpadding
