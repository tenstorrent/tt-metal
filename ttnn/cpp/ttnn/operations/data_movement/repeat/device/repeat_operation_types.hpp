

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::repeat {

struct operation_attributes_t {
    const uint32_t m_num_repeats;
    const bool m_is_last_dim;
    tt::tt_metal::MemoryConfig m_output_mem_config;
};

struct tensor_args_t {
    const Tensor& input;
    const ttnn::SmallVector<uint32_t>& repetition_vector;
};

using tensor_return_value_t = Tensor;

// maybe this can be simplified to a single TensorSpec
using spec_return_value_t = std::vector<TensorSpec>;

}  // namespace ttnn::operations::data_movement::repeat
