// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_matmul_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_matmul {

void MorehMatmulOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const bool transpose_input = operation_attributes.transpose_input;
    const bool transpose_other = operation_attributes.transpose_other;

    log_debug(tt::LogOp, "{}:{}", __func__, __LINE__);

    const auto& input = tensor_args.input;
    const auto& other = tensor_args.other;
    const auto& bias = tensor_args.bias;
    const auto& output = tensor_args.output;

    // validate tensor
    check_tensor(input, "moreh_matmul", "input", {DataType::BFLOAT16});
    check_tensor(other, "moreh_matmul", "other", {DataType::BFLOAT16});
    check_tensor(output, "moreh_matmul", "output", {DataType::BFLOAT16});
    check_tensor(bias, "moreh_matmul", "bias", {DataType::BFLOAT16});

    // check matrix dims
    const auto& input_shape = input.logical_shape();
    const auto& other_shape = other.logical_shape();
    uint32_t input_m = (transpose_input) ? (input_shape[-1]) : (input_shape[-2]);
    uint32_t input_k = (transpose_input) ? (input_shape[-2]) : (input_shape[-1]);
    uint32_t other_k = (transpose_other) ? (other_shape[-1]) : (other_shape[-2]);
    uint32_t other_n = (transpose_other) ? (other_shape[-2]) : (other_shape[-1]);

    TT_FATAL(input_k == other_k, "k must be the same. input_k {}, other_k {}", input_k, other_k);

    // check batch dims
    ttnn::SmallVector<uint32_t> input_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    ttnn::SmallVector<uint32_t> other_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(input_dim, input_shape);
    get_tensor_dim(other_dim, other_shape);
    for (auto i = 2; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        if (input_dim[i] != other_dim[i]) {
            TT_FATAL(
                input_dim[i] == 1 || other_dim[i] == 1,
                "one of dim must be one. {}th dim input_dim {}, other_dim {}",
                i,
                input_dim[i],
                other_dim[i]);
        }
    }

    // check output dims
    if (output.has_value()) {
        const auto& output_shape = output.value().logical_shape();
        uint32_t output_m = output_shape[-2];
        uint32_t output_n = output_shape[-1];
        TT_FATAL(input_m == output_m, "m must be the same. input_m {}, output_m {}", input_m, output_m);
        TT_FATAL(other_n == output_n, "n must be the same. other_n {}, output_n {}", other_n, output_n);

        ttnn::SmallVector<uint32_t> output_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        get_tensor_dim(output_dim, output_shape);

        for (auto i = 2; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
            TT_FATAL(
                std::max(input_dim[i], other_dim[i]) == output_dim[i],
                "{}th max(input_dim[i], other_dim[i]) {} must be the same as output_dim[i] {}",
                i,
                std::max(input_dim[i], other_dim[i]),
                output_dim[i]);
        }
    }

    // check bias size
    if (bias.has_value()) {
        const auto& bias_wo_shape = bias.value().logical_shape();
        uint32_t bias_rank = bias_wo_shape.rank();
        uint32_t bias_w = bias_wo_shape[-1];
        TT_FATAL(bias_rank == 2, "bias rank {} must be 2 (tilized).", bias_rank);
        TT_FATAL(
            bias_w == 1 || bias_w == other_n,
            "bias_w must be one or the same as other_n. bias_w {}, other_n {}",
            bias_w,
            other_n);
    }
}

void MorehMatmulOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

void MorehMatmulOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

MorehMatmulOperation::tensor_return_value_t MorehMatmulOperation::create_output_tensors(
    const MorehMatmulOperation::operation_attributes_t& operation_attributes,
    const MorehMatmulOperation::tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output.value();
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
};

MorehMatmulOperation::program_factory_t MorehMatmulOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MultiCoreProgramFactory{};
}

MorehMatmulOperation::spec_return_value_t MorehMatmulOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_shape = tensor_args.input.padded_shape();
    const auto& other_shape = tensor_args.other.padded_shape();
    bool transpose_input = operation_attributes.transpose_input;
    bool transpose_other = operation_attributes.transpose_other;
    const auto& input_shape_wo_padding = tensor_args.input.logical_shape();
    const auto& other_shape_wo_padding = tensor_args.other.logical_shape();

    auto h = (transpose_input) ? (input_shape[-1]) : (input_shape[-2]);
    auto w = (transpose_other) ? (other_shape[-2]) : (other_shape[-1]);
    auto h_wo_padding = (transpose_input) ? (input_shape_wo_padding[-1]) : (input_shape_wo_padding[-2]);
    auto w_wo_padding = (transpose_other) ? (other_shape_wo_padding[-2]) : (other_shape_wo_padding[-1]);

    ttnn::SmallVector<uint32_t> input_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    ttnn::SmallVector<uint32_t> other_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(input_dim, input_shape);
    get_tensor_dim(other_dim, other_shape);

    int32_t output_rank = std::max(input_shape.rank(), other_shape.rank());
    log_debug(
        tt::LogOp,
        "{}:{} input, other, output rank {}, {}, {}",
        __func__,
        __LINE__,
        input_shape.rank(),
        other_shape.rank(),
        output_rank);

    ttnn::SmallVector<uint32_t> output_dim(output_rank);
    // batch dims
    for (int i = 0; i < output_rank - 2; ++i) {
        int idx = output_rank - 1 - i;
        TT_ASSERT(idx >= 0);
        const uint32_t max_dim = std::max(input_dim[idx], other_dim[idx]);
        output_dim[i] = max_dim;
    }
    // matrix dims
    output_dim[output_rank - 2] = h;
    output_dim[output_rank - 1] = w;

    ttnn::Shape output_shape({output_dim});
    ttnn::Shape output_shape_wo_padding = output_shape;
    output_shape_wo_padding[output_rank - 2] = h_wo_padding;
    output_shape_wo_padding[output_rank - 1] = w_wo_padding;
    return TensorSpec(
        output_shape_wo_padding,
        TensorLayout::fromPaddedShape(
            tensor_args.input.dtype(),
            PageConfig(Layout::TILE),
            operation_attributes.output_memory_config,
            output_shape_wo_padding,
            output_shape));
}
}  // namespace ttnn::operations::moreh::moreh_matmul

namespace ttnn::prim {
ttnn::operations::moreh::moreh_matmul::MorehMatmulOperation::tensor_return_value_t moreh_matmul(
    const Tensor& input,
    const Tensor& other,
    bool transpose_input,
    bool transpose_other,
    const std::optional<Tensor>& output,
    const std::optional<const Tensor>& bias,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_matmul::MorehMatmulOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        transpose_input,
        transpose_other,
        output_memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config)};
    auto tensor_args = OperationType::tensor_args_t{input, other, output, bias};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
