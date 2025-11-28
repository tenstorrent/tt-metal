

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::repeat {

struct operation_attributes_t {
    uint32_t m_num_repeats;
    bool m_is_last_dim;
    tt::tt_metal::MemoryConfig m_output_mem_config;
    ttnn::SmallVector<uint32_t> repetition_vector;
};

struct tensor_args_t {
    Tensor input;
};

using tensor_return_value_t = Tensor;

// maybe this can be simplified to a single TensorSpec
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::data_movement::repeat
