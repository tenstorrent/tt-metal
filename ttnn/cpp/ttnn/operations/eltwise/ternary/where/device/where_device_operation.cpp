// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where_device_operation.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::ternary {

DataType WhereDeviceOperation::operation_attributes_t::get_dtype() const { return dtype.value_or(input_dtype); }

WhereDeviceOperation::program_factory_t WhereDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(!tensor_args.predicate.is_sharded(), "WhereDeviceOperation is not implemented for sharded tensors");
    return WhereProgramFactory{};
}

tt::stl::hash::hash_t WhereDeviceOperation::operation_attributes_t::to_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        where_variant, broadcast_type, memory_config, get_dtype(), compute_kernel_config);
}

void WhereDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void WhereDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& predicate_tensor = tensor_args.predicate;
    const auto& value_true_tensor = tensor_args.value_true;
    const auto& value_false_tensor = tensor_args.value_false;
    const auto& optional_output_tensor = tensor_args.optional_output_tensor;

    auto out_memory_config = args.memory_config;
    if (optional_output_tensor.has_value()) {
        out_memory_config = optional_output_tensor->memory_config();
    }

    TT_FATAL(
        predicate_tensor.storage_type() == StorageType::DEVICE,
        "Where operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(predicate_tensor.storage_type()));

    TT_FATAL(
        predicate_tensor.buffer() != nullptr,
        "Operands to eltwise where need to be allocated in buffers on the device. Buffer is null.");

    TT_FATAL(
        predicate_tensor.memory_config().memory_layout() == out_memory_config.memory_layout(),
        "Where operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
        static_cast<int>(predicate_tensor.memory_config().memory_layout()),
        static_cast<int>(out_memory_config.memory_layout()));

    // Validate tensor shapes based on variant - now supporting broadcast
    auto is_broadcastable = [](const Tensor& a, const Tensor& b) -> bool {
        const auto& a_shape = a.logical_shape();
        const auto& b_shape = b.logical_shape();

        // Same shape is always valid
        if (a_shape == b_shape) {
            return true;
        }

        // Check for column broadcast: tensors have same rank and all dimensions match except width
        if (a_shape.rank() != b_shape.rank() || a_shape.rank() < 2) {
            return false;
        }

        auto rank = a_shape.rank();
        auto a_w = a_shape[-1];
        auto a_h = a_shape[-2];
        auto b_w = b_shape[-1];
        auto b_h = b_shape[-2];

        // Check if heights match
        if (a_h != b_h) {
            return false;
        }

        // Check for column broadcast pattern: one has width 1, other has width > 1
        if (!((a_w == 1 && b_w > 1) || (a_w > 1 && b_w == 1) || (a_w == b_w))) {
            return false;
        }

        // Check that all other dimensions match
        for (int i = 0; i < rank - 2; ++i) {
            if (a_shape[i] != b_shape[i]) {
                return false;
            }
        }

        return true;
    };

    if (args.where_variant == WhereVariant::TTT) {
        TT_FATAL(
            is_broadcastable(predicate_tensor, value_true_tensor.value()),
            "Where TTT operation requires predicate and value_true to be broadcastable. Predicate: {}, Value true: {}",
            predicate_tensor.logical_shape(),
            value_true_tensor.value().logical_shape());
        TT_FATAL(
            is_broadcastable(predicate_tensor, value_false_tensor.value()),
            "Where TTT operation requires predicate and value_false to be broadcastable. Predicate: {}, Value false: "
            "{}",
            predicate_tensor.logical_shape(),
            value_false_tensor.value().logical_shape());
        TT_FATAL(
            is_broadcastable(value_true_tensor.value(), value_false_tensor.value()),
            "Where TTT operation requires value_true and value_false to be broadcastable. Value true: {}, Value false: "
            "{}",
            value_true_tensor.value().logical_shape(),
            value_false_tensor.value().logical_shape());
    } else if (args.where_variant == WhereVariant::TTS) {
        TT_FATAL(
            is_broadcastable(predicate_tensor, value_true_tensor.value()),
            "Where TTS operation requires predicate and value_true to be broadcastable. Predicate: {}, Value true: {}",
            predicate_tensor.logical_shape(),
            value_true_tensor.value().logical_shape());
        TT_FATAL(
            args.value_false_scalar.has_value(),
            "Where TTS operation requires value_false_scalar to be set in operation attributes");
    } else if (args.where_variant == WhereVariant::TST) {
        TT_FATAL(
            is_broadcastable(predicate_tensor, value_false_tensor.value()),
            "Where TST operation requires predicate and value_false to be broadcastable. Predicate: {}, Value false: "
            "{}",
            predicate_tensor.logical_shape(),
            value_false_tensor.value().logical_shape());
        TT_FATAL(
            args.value_true_scalar.has_value(),
            "Where TST operation requires value_true_scalar to be set in operation attributes");
    }

    if (!predicate_tensor.is_sharded()) {
        TT_FATAL(
            predicate_tensor.layout() == Layout::TILE,
            "Where operation requires tensor to be in Tile layout when working with non-sharded input tensor. Input "
            "tensor layout: {}",
            static_cast<int>(predicate_tensor.layout()));

        TT_FATAL(
            predicate_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Where operation requires Interleaved memory layout when working with non-sharded input tensor. Input "
            "memory layout: `{}`",
            static_cast<int>(predicate_tensor.memory_config().memory_layout()));
    }

    if (optional_output_tensor.has_value()) {
        const auto computed_output_shape = compute_output_specs(args, tensor_args).logical_shape();
        const auto optional_output_tensor_shape = optional_output_tensor.value().logical_shape();
        TT_FATAL(
            optional_output_tensor_shape == computed_output_shape,
            "When preallocted output tensor is used, Where operation requires its shape to match the computed "
            "shape. Computed shape: {}, Shape in preallocated output tensor: {}",
            computed_output_shape,
            optional_output_tensor_shape);

        if (!predicate_tensor.is_sharded()) {
            TT_FATAL(
                (optional_output_tensor.value().layout() == Layout::TILE),
                "Where operation requires output tensor to be in Tile layout when working with non-sharded tensor.");
        }
    }
}

TensorSpec WhereDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }

    auto output_layout = Layout::TILE;
    if (args.memory_config.is_sharded()) {
        output_layout = tensor_args.predicate.layout();
    }

    // Compute the correct output shape, considering broadcast cases
    auto output_shape = tensor_args.predicate.logical_shape();

    // For column broadcast cases, determine the correct output shape
    if (args.broadcast_type == WhereBroadcastType::COL_BCAST && args.where_variant == WhereVariant::TTT) {
        const auto& predicate = tensor_args.predicate;
        const auto& value_true = tensor_args.value_true.value();
        const auto& value_false = tensor_args.value_false.value();

        const auto& pred_shape = predicate.logical_shape();
        const auto& true_shape = value_true.logical_shape();
        const auto& false_shape = value_false.logical_shape();

        // For column broadcast, determine which tensor has the target shape
        if (pred_shape[-1] == 1 && true_shape[-1] == false_shape[-1] && true_shape[-1] > 1) {
            // Predicate broadcast: output shape matches value tensors
            output_shape = true_shape;
        }
        // For value tensor broadcast, output shape already matches predicate (no change needed)
    }

    return TensorSpec(output_shape, tt::tt_metal::TensorLayout(args.dtype.value(), output_layout, args.memory_config));
}

Tensor WhereDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return *tensor_args.optional_output_tensor;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.predicate.device());
}

tt::stl::hash::hash_t WhereDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& predicate_tensor = tensor_args.predicate;
    const auto& value_true = tensor_args.value_true;
    const auto& value_false = tensor_args.value_false;
    const auto& predicate_shape = predicate_tensor.padded_shape();
    WhereVariant variant = args.where_variant;

    auto program_factory = select_program_factory(args, tensor_args);

    tt::stl::hash::hash_t hash = tt::tt_metal::operation::hash_operation<WhereDeviceOperation>(
        args,
        program_factory.index(),
        predicate_tensor.dtype(),
        predicate_tensor.memory_config(),
        predicate_shape.volume());

    if (variant == WhereVariant::TTT) {
        hash = tt::tt_metal::operation::hash_operation<WhereDeviceOperation>(
            args,
            program_factory.index(),
            predicate_tensor.dtype(),
            predicate_tensor.memory_config(),
            value_true.value().dtype(),
            value_true.value().memory_config(),
            value_false.value().dtype(),
            value_false.value().memory_config(),
            predicate_shape.volume());
    } else if (variant == WhereVariant::TTS) {
        // For TTS, include the scalar value in hash
        hash = tt::tt_metal::operation::hash_operation<WhereDeviceOperation>(
            args,
            program_factory.index(),
            predicate_tensor.dtype(),
            predicate_tensor.memory_config(),
            value_true.value().dtype(),
            value_true.value().memory_config(),
            predicate_shape.volume());
    } else if (variant == WhereVariant::TST) {
        // For TST, include the scalar value in hash
        hash = tt::tt_metal::operation::hash_operation<WhereDeviceOperation>(
            args,
            program_factory.index(),
            predicate_tensor.dtype(),
            predicate_tensor.memory_config(),
            value_false.value().dtype(),
            value_false.value().memory_config(),
            predicate_shape.volume());
    }

    return hash;
}

bool WhereDeviceOperation::skip_launch(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

std::tuple<WhereDeviceOperation::operation_attributes_t, WhereDeviceOperation::tensor_args_t>
WhereDeviceOperation::invoke(
    const Tensor& predicate,
    const Tensor& value_true,
    const Tensor& value_false,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    // Determine broadcast type based on tensor shapes
    WhereBroadcastType broadcast_type = WhereBroadcastType::NONE;
    const auto& pred_shape = predicate.logical_shape();
    const auto& true_shape = value_true.logical_shape();
    const auto& false_shape = value_false.logical_shape();

    // Check if we have column broadcast pattern (rank >= 2 and last dimension difference)
    if (pred_shape.rank() >= 2) {
        bool has_col_bcast = false;

        // Case 1: value tensors need broadcasting (original cases)
        if ((true_shape[-1] == 1 && pred_shape[-1] > 1) || (false_shape[-1] == 1 && pred_shape[-1] > 1)) {
            // Check if all other dimensions match (excluding the last one)
            bool all_other_dims_match = true;
            for (int i = 0; i < pred_shape.rank() - 1; ++i) {
                if (true_shape[i] != pred_shape[i] || false_shape[i] != pred_shape[i]) {
                    all_other_dims_match = false;
                    break;
                }
            }
            has_col_bcast = all_other_dims_match;
        }

        // Case 2: predicate needs broadcasting (new case)
        if (!has_col_bcast && pred_shape[-1] == 1 && true_shape[-1] == false_shape[-1] && true_shape[-1] > 1) {
            // Check if all other dimensions match (excluding the last one)
            bool all_other_dims_match = true;
            for (int i = 0; i < pred_shape.rank() - 1; ++i) {
                if (true_shape[i] != pred_shape[i] || false_shape[i] != pred_shape[i]) {
                    all_other_dims_match = false;
                    break;
                }
            }
            has_col_bcast = all_other_dims_match;
        }

        if (has_col_bcast) {
            broadcast_type = WhereBroadcastType::COL_BCAST;
        }
    }

    operation_attributes_t attributes{
        .where_variant = WhereVariant::TTT,
        .broadcast_type = broadcast_type,
        .memory_config = memory_config.value_or(predicate.memory_config()),
        .input_dtype = predicate.dtype(),
        .dtype = output_dtype.value_or(value_true.dtype()),
        .compute_kernel_config = std::nullopt,
    };

    tensor_args_t args{
        .predicate = predicate,
        .value_true = value_true,
        .value_false = value_false,
        .optional_output_tensor = optional_output_tensor};

    return {attributes, args};
}

std::tuple<WhereDeviceOperation::operation_attributes_t, WhereDeviceOperation::tensor_args_t>
WhereDeviceOperation::invoke(
    const Tensor& predicate,
    const Tensor& value_true,
    float value_false_scalar,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    operation_attributes_t attributes{
        .where_variant = WhereVariant::TTS,
        .broadcast_type = WhereBroadcastType::NONE,
        .memory_config = memory_config.value_or(predicate.memory_config()),
        .input_dtype = predicate.dtype(),
        .dtype = output_dtype.value_or(value_true.dtype()),
        .compute_kernel_config = std::nullopt,
        .value_false_scalar = value_false_scalar,
    };

    tensor_args_t args{
        .predicate = predicate,
        .value_true = value_true,
        .value_false = std::nullopt,
        .optional_output_tensor = optional_output_tensor};

    return {attributes, args};
}

std::tuple<WhereDeviceOperation::operation_attributes_t, WhereDeviceOperation::tensor_args_t>
WhereDeviceOperation::invoke(
    const Tensor& predicate,
    float value_true_scalar,
    const Tensor& value_false,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    operation_attributes_t attributes{
        .where_variant = WhereVariant::TST,
        .broadcast_type = WhereBroadcastType::NONE,
        .memory_config = memory_config.value_or(predicate.memory_config()),
        .input_dtype = predicate.dtype(),
        .dtype = output_dtype.value_or(value_false.dtype()),
        .compute_kernel_config = std::nullopt,
        .value_true_scalar = value_true_scalar,
    };

    tensor_args_t args{
        .predicate = predicate,
        .value_true = std::nullopt,
        .value_false = value_false,
        .optional_output_tensor = optional_output_tensor};

    return {attributes, args};
}

}  // namespace ttnn::operations::ternary
