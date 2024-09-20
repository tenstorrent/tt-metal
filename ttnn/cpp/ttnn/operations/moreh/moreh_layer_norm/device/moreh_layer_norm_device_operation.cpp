// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_device_operation.hpp"

#include <cstdint>

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm {

namespace {
inline void check_tensor(const Tensor& tensor, const std::string& op_name) {
    TT_FATAL(
        tensor.get_layout() == Layout::TILE, "{} only supports tiled layout. Got: {}", op_name, tensor.get_layout());
    TT_FATAL(
        tensor.get_dtype() == DataType::BFLOAT16, "{} only supports bfloat16. Got: {}", op_name, tensor.get_dtype());
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE,
        "Operands to {} need to be on device! Got: {}",
        op_name,
        tensor.storage_type());
    TT_FATAL(tensor.buffer() != nullptr, "Operands to {} need to be allocated in buffers on device!", op_name);
}
}  // namespace

void MorehLayerNormOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    check_tensor(input, "moreh_layer_norm");

    const auto normalized_dims = operation_attributes.normalized_dims;

    TT_FATAL(normalized_dims > 0, "normalized_dims should > 0. Got {}", normalized_dims);
    TT_FATAL(
        normalized_dims <= input.get_legacy_shape().rank(),
        "normalized_dims should <= input rank ({}). Got: {}",
        input.get_legacy_shape().rank(),
        normalized_dims);

    if (gamma.has_value()) {
        check_tensor(gamma.value(), "moreh_layer_norm");
        TT_FATAL(input.device() == gamma.value().device(), "input and gamma should be on the same device.");
    }

    if (beta.has_value()) {
        check_tensor(beta.value(), "moreh_layer_norm");
        TT_FATAL(input.device() == beta.value().device(), "input and beta should be on the same device");
    }

    auto& mean = tensor_args.mean;
    auto& rstd = tensor_args.rstd;

    if (mean.has_value()) {
        check_tensor(mean.value(), "moreh_layer_norm");
    }

    if (rstd.has_value()) {
        check_tensor(rstd.value(), "moreh_layer_norm");
    }
}

MorehLayerNormOperation::program_factory_t MorehLayerNormOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void MorehLayerNormOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehLayerNormOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehLayerNormOperation::shape_return_value_t MorehLayerNormOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input = tensor_args.input;
    auto normalized_dims = operation_attributes.normalized_dims;
    auto input_shape = input.get_shape();
    auto input_shape_without_padding = input_shape.value.without_padding();
    auto input_rank = input_shape.rank();
    auto output_rank = input_rank - normalized_dims;

    std::vector<uint32_t> output_size_vec;
    auto dimensions_pads = std::vector<Padding::PadDimension>();

    if (output_rank == 1) {
        output_size_vec.push_back(32);
        dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = 31});
    }

    for (uint32_t dim = 0; dim < output_rank; dim++) {
        auto input_shape_without_padding_size = input_shape_without_padding[dim];
        if (tt::operations::primary::is_hw_dim(dim, output_rank)) {
            output_size_vec.push_back(round_up_to_mul32(input_shape_without_padding_size));
            auto padding_back = output_size_vec[dim] - input_shape_without_padding_size;
            dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = padding_back});
        } else {
            output_size_vec.push_back(input_shape_without_padding_size);
            dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = 0});
        }
    }

    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    auto mean_rstd_output_shape = Shape{tt::tt_metal::LegacyShape(output_size_vec, padding)};
    return {input_shape, mean_rstd_output_shape, mean_rstd_output_shape};
};

MorehLayerNormOperation::tensor_return_value_t MorehLayerNormOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_shapes = compute_output_shapes(operation_attributes, tensor_args);
    auto input = tensor_args.input;
    auto dtype = input.get_dtype();
    Layout layout{Layout::TILE};
    auto device = input.device();

    std::vector<std::optional<Tensor>> result;
    result.reserve(3);

    if (tensor_args.output.has_value())
        result.push_back(tensor_args.output.value());
    else
        result.push_back(
            create_device_tensor(output_shapes.at(0).value, dtype, layout, device, operation_attributes.memory_config));

    if (tensor_args.mean.has_value())
        result.push_back(tensor_args.mean.value());
    else
        result.push_back(std::nullopt);

    if (tensor_args.rstd.has_value())
        result.push_back(tensor_args.rstd.value());
    else
        result.push_back(std::nullopt);

    return std::move(result);
}

std::tuple<MorehLayerNormOperation::operation_attributes_t, MorehLayerNormOperation::tensor_args_t>
MorehLayerNormOperation::invoke(
    const Tensor& input,
    uint32_t normalized_dims,
    float eps,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& output,
    const std::optional<const Tensor>& mean,
    const std::optional<const Tensor>& rstd,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        operation_attributes_t{
            normalized_dims,
            eps,
            memory_config.value_or(input.memory_config()),
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4),
        },
        tensor_args_t{input, gamma, beta, output, mean, rstd}};
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm
