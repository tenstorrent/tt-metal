// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim::qsr {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Float formats the Quasar SFPU unary path is wired for. Quasar has no block-float formats, and the integer
// unary variants (ABS_INT32, bitwise, ...) are not ported here.
bool is_supported_float_dtype(DataType dtype) { return dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32; }

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

UnaryDeviceOperation::program_factory_t UnaryDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return UnaryProgramFactory{};
}

void UnaryDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& preallocated_output = tensor_args.preallocated_output;

    TT_FATAL(!args.op_chain.empty(), "Quasar unary: op_chain must not be empty");
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE,
        "Quasar unary: input must be on device. Input storage type: {}",
        static_cast<int>(input.storage_type()));
    TT_FATAL(input.buffer() != nullptr, "Quasar unary: input must be allocated in a device buffer");

    // The upstream ROW_MAJOR path (rows chunked into tile-sized DFB pages) is not ported yet.
    TT_FATAL(input.layout() == Layout::TILE, "Quasar unary: only TILE layout is supported, got {}", input.layout());
    const auto& tile = input.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
        "Quasar unary: only 32x32 tiles are supported, got {}x{}",
        tile.get_height(),
        tile.get_width());

    TT_FATAL(
        CMAKE_UNIQUE_NAMESPACE::is_supported_float_dtype(input.dtype()),
        "Quasar unary: input dtype must be BFLOAT16 or FLOAT32, got {}",
        input.dtype());
    const DataType output_dtype = preallocated_output.has_value() ? preallocated_output->dtype() : args.output_dtype;
    TT_FATAL(
        CMAKE_UNIQUE_NAMESPACE::is_supported_float_dtype(output_dtype),
        "Quasar unary: output dtype must be BFLOAT16 or FLOAT32, got {}",
        output_dtype);

    const auto& out_memory_config =
        preallocated_output.has_value() ? preallocated_output->memory_config() : args.output_memory_config;
    if (!input.is_sharded()) {
        TT_FATAL(
            input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Quasar unary: a non-sharded input must be interleaved, got {}",
            input.memory_config().memory_layout());
    }
    if (!out_memory_config.is_sharded()) {
        TT_FATAL(
            out_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Quasar unary: a non-sharded output must be interleaved, got {}",
            out_memory_config.memory_layout());
    }

    if (preallocated_output.has_value()) {
        // compute_output_specs checks the preallocated shape.
        compute_output_specs(args, tensor_args);
        TT_FATAL(
            preallocated_output->layout() == input.layout(),
            "Quasar unary: preallocated output layout {} must match the input layout {}",
            preallocated_output->layout(),
            input.layout());
        TT_FATAL(
            preallocated_output->buffer() != nullptr,
            "Quasar unary: preallocated output must be allocated in a device buffer");
    }
}

tt::tt_metal::TensorSpec UnaryDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Elementwise: the output has the input's logical shape.
    const auto& input = tensor_args.input;
    const auto output_shape = input.logical_shape();
    if (tensor_args.preallocated_output.has_value()) {
        const auto preallocated_shape = tensor_args.preallocated_output->logical_shape();
        TT_FATAL(
            preallocated_shape == output_shape,
            "Quasar unary: preallocated output shape {} must match the computed shape {}",
            preallocated_shape,
            output_shape);
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto output_layout = input.layout();
    if (args.output_memory_config.is_sharded()) {
        // Same shard-spec resolution as the upstream unary op.
        if (!args.output_memory_config.shard_spec().has_value() &&
            args.output_memory_config.nd_shard_spec().has_value()) {
            return tt::tt_metal::TensorSpec(
                output_shape,
                tt::tt_metal::TensorLayout(
                    args.output_dtype, tt::tt_metal::PageConfig(output_layout), args.output_memory_config));
        }
        auto shard_spec = args.output_memory_config.shard_spec();
        if (!shard_spec.has_value()) {
            const auto& padded_out_shape = input.padded_shape();
            if (input.memory_config().shard_spec().has_value()) {
                shard_spec = ttnn::operations::unary::adjust_to_shape(
                    *input.memory_config().shard_spec(), input.padded_shape(), padded_out_shape);
            } else {
                const uint32_t output_element_size_bytes = args.output_dtype == DataType::FLOAT32 ? 4u : 2u;
                shard_spec = ttnn::operations::unary::generate_output_shard_spec(
                    input, padded_out_shape, args.output_memory_config.memory_layout(), output_element_size_bytes);
            }
        }
        return tt::tt_metal::TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                args.output_dtype,
                tt::tt_metal::PageConfig(output_layout),
                MemoryConfig(
                    args.output_memory_config.memory_layout(), args.output_memory_config.buffer_type(), shard_spec)));
    }

    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            args.output_dtype,
            tt::tt_metal::PageConfig(output_layout),
            args.output_memory_config,
            output_shape,
            input.padded_shape()));
}

Tensor UnaryDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

bool UnaryDeviceOperation::skip_launch(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& /*tensor_args*/,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

Tensor unary(
    const Tensor& input,
    const std::vector<ttnn::operations::unary::EltwiseUnaryWithParam>& op_chain,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    bool fp32_dest_acc_en,
    bool preserve_fp32_precision,
    const std::optional<Tensor>& optional_output_tensor) {
    const MemoryConfig& memory_config =
        optional_output_tensor.has_value() ? optional_output_tensor->memory_config() : output_memory_config;
    return ttnn::device_operation::launch<UnaryDeviceOperation>(
        UnaryParams{
            .op_chain = op_chain,
            .output_dtype = output_dtype,
            .output_memory_config = memory_config,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .preserve_fp32_precision = preserve_fp32_precision,
            .worker_grid = ttnn::operations::unary::get_worker_grid(
                input, optional_output_tensor, std::optional<MemoryConfig>(output_memory_config), std::nullopt),
        },
        UnaryInputs{.input = input, .preallocated_output = optional_output_tensor});
}

}  // namespace ttnn::prim::qsr
