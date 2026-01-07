#include "ttnn/operations/experimental/conv3d/device/conv3d_device_operation.hpp"
#include "ttnn/operations/experimental/conv3d/device/conv3d_program_factory.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

namespace ttnn::operations::experimental::conv3d {

void Conv3dDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, 
    const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to conv3d need to be on device!");
    bool valid_layout = (input_tensor.get_layout() == tt::tt_metal::Layout::TILE || 
                         input_tensor.get_layout() == tt::tt_metal::Layout::ROW_MAJOR);
    TT_FATAL(valid_layout, "Input to conv3d must be TILE or ROW_MAJOR");
    if (input_tensor.is_sharded()) {
        auto mem_config = input_tensor.memory_config();
        bool valid_sharding = (mem_config.memory_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
                               mem_config.memory_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED ||
                               mem_config.memory_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED);
        TT_FATAL(valid_sharding, "Unsupported sharding layout for conv3d");
    }
    TT_FATAL(weight_tensor.get_layout() == tt::tt_metal::Layout::TILE, "Weights must be Tiled");
}

void Conv3dDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

Conv3dDeviceOperation::program_factory_t Conv3dDeviceOperation::select_program_factory(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return program::Conv3dProgramFactory{};
}

spec_return_value_t Conv3dDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return TensorSpec(input_tensor.get_logical_shape(),
        TensorLayout(attributes.dtype, tt::tt_metal::PageConfig(input_tensor.get_layout()), attributes.output_mem_config));
}

tensor_return_value_t Conv3dDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t Conv3dDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return operation::hash_operation<Conv3dDeviceOperation>(attributes, tensor_args);
}
} 

namespace ttnn::prim {
ttnn::operations::experimental::conv3d::Conv3dDeviceOperation::tensor_return_value_t conv3d(
    const Tensor& input_tensor, const Tensor& weight_tensor, const std::optional<Tensor>& bias_tensor,
    const ttnn::operations::experimental::conv3d::Conv3dConfig& config, tt::tt_metal::DataType dtype,
    uint32_t output_channels, const std::array<uint32_t, 3>& kernel_size, const std::array<uint32_t, 3>& stride,
    const std::array<uint32_t, 3>& padding, const std::array<uint32_t, 3>& dilation, const std::string& padding_mode,
    uint32_t groups, const std::optional<MemoryConfig>& memory_config, std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = ttnn::operations::experimental::conv3d::Conv3dDeviceOperation;
    auto kernel_config_val = init_device_compute_kernel_config(input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);
    auto operation_attributes = OperationType::operation_attributes_t{
        .config = config, .output_mem_config = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        .compute_kernel_config = kernel_config_val, .dtype = dtype, .output_channels = output_channels,
        .kernel_size = kernel_size, .stride = stride, .padding = padding, .dilation = dilation, .padding_mode = padding_mode, .groups = groups
    };
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor, .weight_tensor = weight_tensor, .bias_tensor = bias_tensor};
    return ttnn::device_operation::detail::launch_on_device<OperationType>(operation_attributes, tensor_args);
}
}