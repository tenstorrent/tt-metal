// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary_device_operation.hpp"
#include "ternary_op_utils.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::ternary {

static ttnn::Shape compute_broadcasted_output_ternary(
    const ttnn::Shape& a_shape, const ttnn::Shape& b_shape, const ttnn::Shape& c_shape) {
    const int rank_a = a_shape.rank();
    const int rank_b = b_shape.rank();
    const int rank_c = c_shape.rank();
    const int largest_rank = std::max({rank_a, rank_b, rank_c});

    SmallVector<uint32_t> output_shape(largest_rank, 1);

    for (int i = -1; i >= -largest_rank; --i) {
        auto dim_a = (i >= -rank_a) ? a_shape[i] : 1;
        auto dim_b = (i >= -rank_b) ? b_shape[i] : 1;
        auto dim_c = (i >= -rank_c) ? c_shape[i] : 1;

        uint32_t max_dim = 1;
        if (dim_a != 1) {
            max_dim = std::max(max_dim, dim_a);
        }
        if (dim_b != 1) {
            max_dim = std::max(max_dim, dim_b);
        }
        if (dim_c != 1) {
            max_dim = std::max(max_dim, dim_c);
        }

        bool compatible = true;
        if (dim_a != 1 && dim_a != max_dim) {
            compatible = false;
        }
        if (dim_b != 1 && dim_b != max_dim) {
            compatible = false;
        }
        if (dim_c != 1 && dim_c != max_dim) {
            compatible = false;
        }

        TT_FATAL(
            compatible,
            "Broadcasting rule violation for rank {}, dim a: {}, dim b: {}, dim c: {}",
            i,
            dim_a,
            dim_b,
            dim_c);

        if (i <= -6) {
            TT_FATAL(
                dim_a == dim_b && dim_b == dim_c,
                "Broadcasting rule violation for rank >= 6 : dim {}, Broadcast is supported up to rank 5, "
                "dim a: {}, dim b: {}, dim c: {}",
                i,
                dim_a,
                dim_b,
                dim_c);
        }

        output_shape[i + largest_rank] = max_dim;
    }
    return ttnn::Shape(output_shape);
}

static ttnn::Shape compute_broadcasted_output_binary(const ttnn::Shape& a_shape, const ttnn::Shape& b_shape) {
    const int rank_a = a_shape.rank();
    const int rank_b = b_shape.rank();
    const int largest_rank = std::max(rank_a, rank_b);
    SmallVector<uint32_t> output_shape(largest_rank, 1);

    for (int i = -1; i >= -largest_rank; --i) {
        auto a_dim = (i >= -rank_a) ? a_shape[i] : 1;
        auto b_dim = (i >= -rank_b) ? b_shape[i] : 1;

        TT_FATAL(
            a_dim == b_dim || a_dim == 1 || b_dim == 1,
            "Broadcasting rule violation for rank {}, dim a: {}, dim b: {}",
            i,
            a_dim,
            b_dim);

        if (i <= -6) {
            TT_FATAL(
                a_dim == b_dim,
                "Broadcasting rule violation for rank >= 6 : dim {}, Broadcast is supported up to rank 5, dim a: "
                "{}, "
                "dim b: {}",
                i,
                a_dim,
                b_dim);
        }

        uint32_t out_dim = std::max<uint32_t>(a_dim, b_dim);
        output_shape[i + largest_rank] = out_dim;
    }
    return ttnn::Shape(output_shape);
}

DataType TernaryDeviceOperation::operation_attributes_t::get_dtype() const { return dtype.value_or(input_dtype); }

TernaryDeviceOperation::program_factory_t TernaryDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(!tensor_args.input_tensor_a.is_sharded(), "TernaryDeviceOperation is not implemented for sharded tensors");
    return TernaryProgramFactory{};
}

tt::stl::hash::hash_t TernaryDeviceOperation::operation_attributes_t::to_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        ternary_op_type, ternary_variant, broadcast_type, memory_config, get_dtype(), compute_kernel_config, scalar);
}

void TernaryDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void TernaryDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_a = tensor_args.input_tensor_a;
    const auto& input_b = tensor_args.input_tensor_b;
    const auto& input_c = tensor_args.input_tensor_c;
    const auto& optional_output_tensor = tensor_args.optional_output_tensor;

    auto out_memory_config = args.memory_config;
    auto broadcast_type = args.broadcast_type;

    if (optional_output_tensor.has_value()) {
        out_memory_config = optional_output_tensor->memory_config();
    }

    TT_FATAL(
        input_a.storage_type() == StorageType::DEVICE,
        "Ternary operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(input_a.storage_type()));

    TT_FATAL(
        input_a.buffer() != nullptr,
        "Operands to eltwise ternary operation need to be allocated in buffers on the device. Buffer is null.");

    TT_FATAL(
        input_a.memory_config().memory_layout() == out_memory_config.memory_layout(),
        "Ternary operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
        static_cast<int>(input_a.memory_config().memory_layout()),
        static_cast<int>(out_memory_config.memory_layout()));

    // Validate tensor shapes based on variant
    if (args.ternary_variant == TernaryVariant::TTT) {
        TT_FATAL(input_b.has_value() && input_c.has_value(), "TTT variant requires both input_b and input_c tensors");

        TT_FATAL(
            ((broadcast_type != TernaryBroadcastType::SCALAR_A_BCAST) &&
             (broadcast_type != TernaryBroadcastType::SCALAR_B_BCAST)),
            "Unsupported broadcast type for TTT operation. scalar broadcast for TTT requires SCALAR_BCAST");

    } else if (args.ternary_variant == TernaryVariant::TTS) {
        TT_FATAL(input_b.has_value() && !input_c.has_value(), "TTS variant requires input_b tensor and input_c scalar");
        TT_FATAL(
            args.scalar_input_b.has_value(),
            "Ternary TTS operation requires scalar_input_b to be set in operation attributes");

        TT_FATAL(
            (broadcast_type != TernaryBroadcastType::SCALAR_BCAST),
            "Unsupported broadcast type for TTS operation. scalar broadcast for TTS requires SCALAR_A_BCAST or "
            "SCALAR_B_BCAST");

    } else if (args.ternary_variant == TernaryVariant::TST) {
        TT_FATAL(!input_b.has_value() && input_c.has_value(), "TST variant requires input_b scalar and input_c tensor");
        TT_FATAL(
            args.scalar_input_a.has_value(),
            "Ternary TST operation requires scalar_input_a to be set in operation attributes");

        TT_FATAL(
            (broadcast_type != TernaryBroadcastType::SCALAR_BCAST),
            "Unsupported broadcast type for TST operation. scalar broadcast for TST requires SCALAR_A_BCAST or "
            "SCALAR_B_BCAST");
    }

    if (!input_a.is_sharded()) {
        TT_FATAL(
            input_a.layout() == Layout::TILE,
            "Ternary operation requires tensor to be in Tile layout when working with non-sharded input tensor. Input "
            "tensor layout: {}",
            static_cast<int>(input_a.layout()));

        TT_FATAL(
            input_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Ternary operation requires Interleaved memory layout when working with non-sharded input tensor. Input "
            "memory layout: `{}`",
            static_cast<int>(input_a.memory_config().memory_layout()));
    }

    if (optional_output_tensor.has_value()) {
        const auto computed_output_shape = compute_output_specs(args, tensor_args).logical_shape();
        const auto optional_output_tensor_shape = optional_output_tensor.value().logical_shape();
        TT_FATAL(
            optional_output_tensor_shape == computed_output_shape,
            "When preallocted output tensor is used, Ternary operation requires its shape to match the computed "
            "shape. Computed shape: {}, Shape in preallocated output tensor: {}",
            computed_output_shape,
            optional_output_tensor_shape);

        if (!input_a.is_sharded()) {
            TT_FATAL(
                (optional_output_tensor.value().layout() == Layout::TILE),
                "Ternary operation requires output tensor to be in Tile layout when working with non-sharded tensor.");
        }
    }
}

TensorSpec TernaryDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }

    auto output_layout = Layout::TILE;
    if (args.memory_config.is_sharded()) {
        output_layout = tensor_args.input_tensor_a.layout();
    }

    auto broadcast_type = args.broadcast_type;
    auto output_shape = tensor_args.input_tensor_a.logical_shape();

    if (broadcast_type == TernaryBroadcastType::NONE) {
        return TensorSpec(
            output_shape, tt::tt_metal::TensorLayout(args.dtype.value(), output_layout, args.memory_config));
    }

    if (args.ternary_variant == TernaryVariant::TTT) {
        auto a_shape = tensor_args.input_tensor_a.logical_shape();
        auto b_shape = tensor_args.input_tensor_b.value().logical_shape();
        auto c_shape = tensor_args.input_tensor_c.value().logical_shape();

        output_shape = compute_broadcasted_output_ternary(a_shape, b_shape, c_shape);
    } else if (args.ternary_variant == TernaryVariant::TTS) {
        output_shape = compute_broadcasted_output_binary(
            tensor_args.input_tensor_a.logical_shape(), tensor_args.input_tensor_b.value().logical_shape());
    } else if (args.ternary_variant == TernaryVariant::TST) {
        output_shape = compute_broadcasted_output_binary(
            tensor_args.input_tensor_a.logical_shape(), tensor_args.input_tensor_c.value().logical_shape());
    }

    return TensorSpec(output_shape, tt::tt_metal::TensorLayout(args.dtype.value(), output_layout, args.memory_config));
}

Tensor TernaryDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return *tensor_args.optional_output_tensor;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor_a.device());
}

tt::stl::hash::hash_t TernaryDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_a = tensor_args.input_tensor_a;
    const auto& input_b = tensor_args.input_tensor_b;
    const auto& input_c = tensor_args.input_tensor_c;
    const auto& a_shape = input_a.padded_shape();
    TernaryVariant variant = args.ternary_variant;

    auto program_factory = select_program_factory(args, tensor_args);

    tt::stl::hash::hash_t hash = tt::tt_metal::operation::hash_operation<TernaryDeviceOperation>(
        args, program_factory.index(), input_a.dtype(), input_a.memory_config(), a_shape.volume());

    if (variant == TernaryVariant::TTT) {
        hash = tt::tt_metal::operation::hash_operation<TernaryDeviceOperation>(
            args,
            program_factory.index(),
            input_a.dtype(),
            input_a.memory_config(),
            input_b.value().dtype(),
            input_b.value().memory_config(),
            input_c.value().dtype(),
            input_c.value().memory_config(),
            a_shape.volume());
    } else if (variant == TernaryVariant::TTS) {
        hash = tt::tt_metal::operation::hash_operation<TernaryDeviceOperation>(
            args,
            program_factory.index(),
            input_a.dtype(),
            input_a.memory_config(),
            input_b.value().dtype(),
            input_b.value().memory_config(),
            a_shape.volume());
    } else if (variant == TernaryVariant::TST) {
        hash = tt::tt_metal::operation::hash_operation<TernaryDeviceOperation>(
            args,
            program_factory.index(),
            input_a.dtype(),
            input_a.memory_config(),
            input_c.value().dtype(),
            input_c.value().memory_config(),
            a_shape.volume());
    }

    return hash;
}

bool TernaryDeviceOperation::skip_launch(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

std::tuple<TernaryDeviceOperation::operation_attributes_t, TernaryDeviceOperation::tensor_args_t>
TernaryDeviceOperation::invoke(
    TernaryOpType op_type,
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    // Detect broadcast type for TTT variant
    TernaryBroadcastType broadcast_type =
        get_broadcast_type(input_a.logical_shape(), input_b.logical_shape(), input_c.logical_shape());

    operation_attributes_t attributes{
        .ternary_op_type = op_type,
        .ternary_variant = TernaryVariant::TTT,
        .broadcast_type = broadcast_type,
        .memory_config = memory_config.value_or(input_b.memory_config()),
        .input_dtype = input_a.dtype(),
        .dtype = output_dtype.value_or(input_b.dtype()),
        .compute_kernel_config = std::nullopt,
        .scalar_input_a = std::nullopt,
        .scalar_input_b = std::nullopt,
        .scalar = std::nullopt,
    };

    tensor_args_t args{
        .input_tensor_a = input_a,
        .input_tensor_b = input_b,
        .input_tensor_c = input_c,
        .optional_output_tensor = optional_output_tensor};

    return {attributes, args};
}

std::tuple<TernaryDeviceOperation::operation_attributes_t, TernaryDeviceOperation::tensor_args_t>
TernaryDeviceOperation::invoke(
    TernaryOpType op_type,
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float scalar,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    // Detect broadcast type for TTT variant
    TernaryBroadcastType broadcast_type =
        get_broadcast_type(input_a.logical_shape(), input_b.logical_shape(), input_c.logical_shape());

    operation_attributes_t attributes{
        .ternary_op_type = op_type,
        .ternary_variant = TernaryVariant::TTT,
        .broadcast_type = broadcast_type,
        .memory_config = memory_config.value_or(input_b.memory_config()),
        .input_dtype = input_a.dtype(),
        .dtype = output_dtype.value_or(input_b.dtype()),
        .compute_kernel_config = std::nullopt,
        .scalar_input_a = std::nullopt,
        .scalar_input_b = std::nullopt,
        .scalar = scalar,
    };

    tensor_args_t args{
        .input_tensor_a = input_a,
        .input_tensor_b = input_b,
        .input_tensor_c = input_c,
        .optional_output_tensor = optional_output_tensor};

    return {attributes, args};
}

std::tuple<TernaryDeviceOperation::operation_attributes_t, TernaryDeviceOperation::tensor_args_t>
TernaryDeviceOperation::invoke(
    TernaryOpType op_type,
    const Tensor& input_a,
    const Tensor& input_b,
    float scalar_c,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    // Detect broadcast type for TTS variant
    TernaryBroadcastType broadcast_type = get_broadcast_type(input_a.logical_shape(), input_b.logical_shape());

    operation_attributes_t attributes{
        .ternary_op_type = op_type,
        .ternary_variant = TernaryVariant::TTS,
        .broadcast_type = broadcast_type,
        .memory_config = memory_config.value_or(input_b.memory_config()),
        .input_dtype = input_a.dtype(),
        .dtype = output_dtype.value_or(input_b.dtype()),
        .compute_kernel_config = std::nullopt,
        .scalar_input_b = scalar_c,
    };

    tensor_args_t args{
        .input_tensor_a = input_a,
        .input_tensor_b = input_b,
        .input_tensor_c = std::nullopt,
        .optional_output_tensor = optional_output_tensor};

    return {attributes, args};
}

std::tuple<TernaryDeviceOperation::operation_attributes_t, TernaryDeviceOperation::tensor_args_t>
TernaryDeviceOperation::invoke(
    TernaryOpType op_type,
    const Tensor& input_a,
    float scalar_b,
    const Tensor& input_c,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TernaryBroadcastType broadcast_type = get_broadcast_type(input_a.logical_shape(), input_c.logical_shape());

    operation_attributes_t attributes{
        .ternary_op_type = op_type,
        .ternary_variant = TernaryVariant::TST,
        .broadcast_type = broadcast_type,
        .memory_config = memory_config.value_or(input_c.memory_config()),
        .input_dtype = input_a.dtype(),
        .dtype = output_dtype.value_or(input_c.dtype()),
        .compute_kernel_config = std::nullopt,
        .scalar_input_a = scalar_b,
    };

    tensor_args_t args{
        .input_tensor_a = input_a,
        .input_tensor_b = std::nullopt,
        .input_tensor_c = input_c,
        .optional_output_tensor = optional_output_tensor};

    return {attributes, args};
}

}  // namespace ttnn::operations::ternary
