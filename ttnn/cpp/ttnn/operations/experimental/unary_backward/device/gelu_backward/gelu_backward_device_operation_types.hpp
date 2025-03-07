// gelu_backward_device_operation_types.hpp
#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::gelu_backward {

struct operation_attributes_t {
    const tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    const tt::tt_metal::MemoryConfig output_memory_config;
    const std::string approximate = "none";

    // TODO: verify those are needed
    const bool fp32_dest_acc_en = false;
    const bool preserve_fp32_precision = false;
    const bool bfp8_pack_precise = false;
};

struct tensor_args_t {
    const Tensor& grad_output;
    const Tensor& input;
    std::optional<Tensor> preallocated_input_grad;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::gelu_backward
