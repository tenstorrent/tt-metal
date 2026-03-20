// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_ng_device_operation.hpp"
#include "ttnn/operations/eltwise/unary_ng/common/unary_ng_utils.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::unary_ng {

tt::stl::hash::hash_t UnaryNgDeviceOperation::operation_attributes_t::to_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        op_chain,
        output_dtype,
        memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        sub_core_grids,
        worker_grid);
}

void UnaryNgDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& output_tensor = tensor_args.output_tensor;

    auto out_memory_config = args.memory_config;
    if (output_tensor.has_value()) {
        out_memory_config = output_tensor->memory_config();
    }

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "UnaryNg operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "UnaryNg: Operands need to be allocated in buffers on the device. Buffer is null.");

    if (!input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "UnaryNg: Non-sharded input must be interleaved. Input memory layout: {}",
            static_cast<int>(input_tensor.memory_config().memory_layout()));
    }

    if (!out_memory_config.is_sharded()) {
        TT_FATAL(
            out_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "UnaryNg: Non-sharded output must be interleaved. Output memory layout: {}",
            static_cast<int>(out_memory_config.memory_layout()));
    }

    if (output_tensor.has_value()) {
        const auto computed_output_shape = compute_output_specs(args, tensor_args).logical_shape();
        const auto preallocated_output_shape = output_tensor->logical_shape();
        TT_FATAL(
            preallocated_output_shape == computed_output_shape,
            "UnaryNg: Preallocated output shape must match computed shape. Computed: {}, Preallocated: {}",
            computed_output_shape,
            preallocated_output_shape);

        TT_FATAL(
            output_tensor->layout() == input_tensor.layout(),
            "UnaryNg: Output format (tile/row-major) must match input when preallocated. Output: {}, Input: {}",
            static_cast<int>(output_tensor->layout()),
            static_cast<int>(input_tensor.layout()));
    }
}

void UnaryNgDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

TensorSpec UnaryNgDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor->tensor_spec();
    }

    const auto output_shape = tensor_args.input.logical_shape();

    if (args.memory_config.is_sharded()) {
        const auto& memory_layout = args.memory_config.memory_layout();
        const auto& buffer_type = args.memory_config.buffer_type();
        auto shard_spec_opt = args.memory_config.shard_spec();

        if (!shard_spec_opt.has_value()) {
            const auto& padded_out_shape = tensor_args.input.padded_shape();
            if (tensor_args.input.is_sharded()) {
                shard_spec_opt = adjust_to_shape(
                    *tensor_args.input.memory_config().shard_spec(),
                    tensor_args.input.padded_shape(),
                    padded_out_shape);
            } else {
                shard_spec_opt = generate_shard_spec_all_cores(tensor_args.input, padded_out_shape, memory_layout);
            }
        }

        return TensorSpec(
            output_shape,
            TensorLayout(
                args.output_dtype, PageConfig(Layout::TILE), MemoryConfig(memory_layout, buffer_type, shard_spec_opt)));
    }

    const auto output_layout = tensor_args.input.layout();
    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            args.output_dtype,
            PageConfig(output_layout),
            args.memory_config,
            output_shape,
            tensor_args.input.padded_shape()));
}

Tensor UnaryNgDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return *tensor_args.output_tensor;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t UnaryNgDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    TT_FATAL(
        tt::tt_metal::is_device_tensor(input_tensor),
        "UnaryNg: Unexpected tensor type {}",
        input_tensor.storage_type());

    const auto output_spec = compute_output_specs(attributes, tensor_args);
    const auto shard_specs = get_shard_specs(input_tensor.tensor_spec(), output_spec);
    std::optional<uint32_t> src_shard_vol = std::nullopt;
    std::optional<uint32_t> dst_shard_vol = std::nullopt;
    if (shard_specs.has_value()) {
        const auto tile_hw = input_tensor.tensor_spec().tile().get_tile_hw();
        if (input_tensor.is_sharded()) {
            src_shard_vol = shard_specs->input_shard_spec.numel() / tile_hw;
        }
        const auto out_tile_hw = output_spec.tile().get_tile_hw();
        dst_shard_vol = shard_specs->output_shard_spec.numel() / out_tile_hw;
    }

    return operation::hash_operation<UnaryNgDeviceOperation>(
        attributes, input_tensor.dtype(), input_tensor.memory_config(), src_shard_vol, dst_shard_vol);
}

bool UnaryNgDeviceOperation::skip_launch(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& /*tensor_args*/,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

}  // namespace ttnn::operations::unary_ng

namespace ttnn::prim {

Tensor unary_ng(
    const Tensor& input,
    const std::vector<ttnn::operations::unary::EltwiseUnaryWithParam>& op_chain,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    bool fp32_dest_acc_en,
    bool preserve_fp32_precision,
    bool bfp8_pack_precise,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::unary_ng::UnaryNgDeviceOperation;

    auto mem_config_actual =
        optional_output_tensor.has_value() ? optional_output_tensor->memory_config() : (output_memory_config);

    auto worker_grid = ttnn::operations::unary_ng::get_worker_grid(
        input,
        optional_output_tensor,
        std::optional<MemoryConfig>(output_memory_config),
        sub_core_grids,
        mem_config_actual);

    auto operation_attributes = OperationType::operation_attributes_t{
        .op_chain = op_chain,
        .output_dtype = output_dtype,
        .memory_config = mem_config_actual,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .preserve_fp32_precision = preserve_fp32_precision,
        .bfp8_pack_precise = bfp8_pack_precise,
        .worker_grid = worker_grid,
        .sub_core_grids = sub_core_grids,
    };

    auto tensor_args = OperationType::tensor_args_t{.input = input, .output_tensor = optional_output_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
