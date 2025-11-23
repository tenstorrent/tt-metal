#pragma once

#include "conv2d_device_operation_types.hpp"

namespace ttnn::operations::conv::conv2d {

void post_conv2d_op_checks(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor);

}  // namespace ttnn::operations::conv::conv2d
