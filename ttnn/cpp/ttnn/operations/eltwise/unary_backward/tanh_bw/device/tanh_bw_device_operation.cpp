// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanh_bw_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_ops.hpp"
#include "tanh_bw_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::unary_backward::tanh_bw {

void TanhBwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& preallocated_input_grad = tensor_args.preallocated_input_grad;
    const auto& input_tensor = tensor_args.input;
    auto out_memory_config = args.output_memory_config;
    auto output_datatype = args.output_dtype;

    if (output_datatype == DataType::INVALID) {
        output_datatype = input_tensor.dtype();
    }

    if (preallocated_input_grad.has_value()) {
        out_memory_config = preallocated_input_grad->memory_config();
        output_datatype = preallocated_input_grad->dtype();
    }

    TT_FATAL(
        output_datatype == input_tensor.dtype(),
        "TANH_BW operation requires input and output data types to match. Input data type: {}, Output data type: {}",
        input_tensor.dtype(),
        output_datatype);

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "TANH_BW operation requires input to be on Device. Input storage type: {}",
        input_tensor.storage_type());

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to TANH_BW need to be allocated in buffers on the device. Buffer is null.");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == out_memory_config.memory_layout(),
        "TANH_BW operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
        input_tensor.memory_config().memory_layout(),
        out_memory_config.memory_layout());

    TT_FATAL(!input_tensor.is_sharded(), "TANH_BW operation does not support sharded input tensor.");

    TT_FATAL(
        input_tensor.layout() == Layout::TILE,
        "TANH_BW operation requires tensor to be in Tile layout when working with non-sharded input tensor. Input "
        "tensor layout: {}",
        input_tensor.layout());

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "TANH_BW operation requires Interleaved memory layout when working with non-sharded input tensor. Input "
        "memory layout: `{}`",
        static_cast<int>(input_tensor.memory_config().memory_layout()));

    // The factory sizes its circular buffers with tt::tile_size and splits work by
    // physical_volume() / TILE_HW, and neither the layout nor the tile is in compute_program_hash. The
    // layout also reaches the kernels as the aligned page size inside TensorAccessorArgs, a compile-time
    // arg no cache-hit path can refresh, so a ROW_MAJOR operand would be read as tile pages under a
    // cached key. Only input has a layout TT_FATAL of its own, so require both properties here.
    const auto require_standard_tile = [](const Tensor& tensor, const char* name) {
        TT_FATAL(
            tensor.layout() == Layout::TILE,
            "TANH_BW operation does not currently support non-TILE layouts, but {} has {} layout.",
            name,
            tensor.layout());
        const auto tile = tensor.tensor_spec().tile();
        TT_FATAL(
            tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
            "TANH_BW operation does not currently support tiles other than 32x32, but {} has a {}x{} tile.",
            name,
            tile.get_height(),
            tile.get_width());
    };
    require_standard_tile(input_tensor, "the input tensor");
    require_standard_tile(tensor_args.grad_output, "the grad_output tensor");

    // The reader walks the same tile_id range in both operands with a count derived from input alone
    // (physical_volume() / TILE_HW), so an undersized grad_output is read past the end of its
    // allocation. Only input's storage, buffer and shape are covered above.
    TT_FATAL(
        tensor_args.grad_output.storage_type() == StorageType::DEVICE,
        "TANH_BW operation requires grad_output to be on Device. grad_output storage type: {}",
        tensor_args.grad_output.storage_type());
    TT_FATAL(
        tensor_args.grad_output.buffer() != nullptr,
        "TANH_BW operation requires grad_output to be allocated in a buffer on the device. Buffer is null.");
    TT_FATAL(
        tensor_args.grad_output.padded_shape() == input_tensor.padded_shape(),
        "TANH_BW operation requires grad_output and input to have the same padded shape, but got {} and {}.",
        tensor_args.grad_output.padded_shape(),
        input_tensor.padded_shape());

    if (preallocated_input_grad.has_value()) {
        const auto& preallocated = preallocated_input_grad.value();
        require_standard_tile(preallocated, "the preallocated output tensor");
        // Pin the preallocated output to the input rather than to compute_output_specs: that function
        // returns this very tensor's spec when one is supplied, so comparing against it is a tautology
        // that can never fire. The writer emits one page per input tile at an offset derived from
        // input.physical_volume(), so an undersized buffer is written past its end.
        TT_FATAL(
            preallocated.logical_shape() == input_tensor.logical_shape(),
            "When a preallocated output tensor is used, TANH_BW operation requires its logical shape to match the "
            "input's. Input shape: {}, preallocated output shape: {}",
            input_tensor.logical_shape(),
            preallocated.logical_shape());
        TT_FATAL(
            preallocated.padded_shape() == input_tensor.padded_shape(),
            "When a preallocated output tensor is used, TANH_BW operation requires its padded shape to match the "
            "input's, because the writer emits one page per input tile. Input padded shape: {}, preallocated output "
            "padded shape: {}",
            input_tensor.padded_shape(),
            preallocated.padded_shape());
    }
}

tt::tt_metal::TensorSpec TanhBwDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_input_grad.has_value()) {
        return tensor_args.preallocated_input_grad->tensor_spec();
    }

    auto output_layout = Layout::TILE;
    if (args.output_memory_config.is_sharded()) {
        output_layout = tensor_args.input.layout();
    }

    DataType output_dtype = args.output_dtype;
    if (output_dtype == DataType::INVALID) {
        output_dtype = tensor_args.input.dtype();
    }

    const auto output_shape = tensor_args.input.logical_shape();
    return tt::tt_metal::TensorSpec(output_shape, TensorLayout(output_dtype, output_layout, args.output_memory_config));
}

Tensor TanhBwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_input_grad.has_value()) {
        return *tensor_args.preallocated_input_grad;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t TanhBwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& grad_output = tensor_args.grad_output;
    const auto& input_shape = input_tensor.padded_shape();
    operation::Hash hash = operation::hash_operation<TanhBwDeviceOperation>(
        args,
        input_tensor.dtype(),
        input_tensor.memory_config(),
        grad_output.dtype(),
        grad_output.memory_config(),
        input_shape.volume());

    // args only carries the requested output_dtype/output_memory_config; when the caller supplies its
    // own output tensor that is what the factory actually binds, sizing the destination CB from its
    // dtype and baking a TensorAccessorArgs for its buffer into the writer's compile-time args. Neither
    // can be refreshed on a cache hit, so key on the tensor that is really used.
    if (tensor_args.preallocated_input_grad.has_value()) {
        const auto& preallocated = tensor_args.preallocated_input_grad.value();
        hash =
            ttsl::hash::hash_objects(hash, preallocated.dtype(), preallocated.layout(), preallocated.memory_config());
    }

    return hash;
}

Tensor launch_tanh_bw(
    const Tensor& grad_output,
    const Tensor& input,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_output) {
    auto operation_attributes = TanhBwDeviceOperation::operation_attributes_t{
        .output_dtype = output_dtype, .output_memory_config = output_memory_config};
    auto tensor_args = TanhBwDeviceOperation::tensor_args_t{
        .grad_output = grad_output, .input = input, .preallocated_input_grad = preallocated_output};

    return ttnn::device_operation::launch<TanhBwDeviceOperation>(operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::unary_backward::tanh_bw
