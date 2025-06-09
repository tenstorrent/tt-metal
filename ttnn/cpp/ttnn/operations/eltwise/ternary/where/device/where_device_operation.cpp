#include "where_device_operation.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::ternary {

// implement the device operation here

// Implement the invoke methods for different combinations of inputs
std::tuple<WhereDeviceOperation::operation_attributes_t, WhereDeviceOperation::tensor_args_t>
WhereDeviceOperation::invoke(
    const Tensor& predicate,
    const Tensor& value_true,
    const Tensor& value_false,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    operation_attributes_t attributes{
        .memory_config = memory_config.value_or(predicate.memory_config()),
        .input_dtype = predicate.dtype(),
        .dtype = output_dtype,
        .worker_grid = predicate.memory_config().core_range_set(),
        .compute_kernel_config = std::nullopt,

    };

    tensor_args_t args{
        .predicate = predicate,
        .value_true = value_true,
        .value_false = value_false,
        .output_tensor = optional_output_tensor};

    return {attributes, args};
}

// Implement other invoke overloads similarly

}  // namespace ttnn::operations::ternary
