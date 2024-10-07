#include "full_like_device_operation.hpp"
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::full_like {

FullLikeOperation::program_factory_t FullLikeOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void FullLikeOperation::validate(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto shape = tensor_args.input.get_shape();
    if (operation_attributes.layout == Layout::TILE) {
        if (shape.rank() < 2) {
            TT_THROW("TILE layout requires rank >= 2");
        }
        TT_ASSERT(
            shape[-1] % tt::constants::TILE_WIDTH == 0,
            "TILE layout requires width dimension to be multiple of {}",
            tt::constants::TILE_WIDTH);
        TT_ASSERT(
            shape[-2] % tt::constants::TILE_HEIGHT == 0,
            "TILE layout requires height dimension to be multiple of {}",
            tt::constants::TILE_HEIGHT);
    }
}

void FullLikeOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}

void FullLikeOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}

FullLikeOperation::shape_return_value_t FullLikeOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.input.get_shape();
}

FullLikeOperation::tensor_return_value_t FullLikeOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_shape = compute_output_shapes(operation_attributes, tensor_args);
    const auto& input = tensor_args.input;
    return create_device_tensor(
        output_shape.value,
        operation_attributes.dtype,
        operation_attributes.layout,
        input.device(),
        operation_attributes.memory_config
    );
}

std::tuple<FullLikeOperation::operation_attributes_t, FullLikeOperation::tensor_args_t>
FullLikeOperation::invoke(
        const Tensor &input,
        const std::variant<float, int> fill_value,
        const std::optional<DataType> &dtype,
        const std::optional<Layout> &layout,
        const std::optional<MemoryConfig> &memory_config) {
    return {
        operation_attributes_t {
            fill_value,
            dtype.value_or(input.tensor_attributes->dtype),
            layout.value_or(input.tensor_attributes->layout),
            memory_config.value_or(input.memory_config())},
        tensor_args_t {input}};
    }



}
