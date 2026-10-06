// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write_device_operation.hpp"

#include <algorithm>
#include <tt_stl/assert.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/caller_owned_topology.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

SliceWriteDeviceOperation::program_factory_t SliceWriteDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const bool has_step =
        std::any_of(operation_attributes.step.cbegin(), operation_attributes.step.cend(), [](uint32_t step_val) {
            return step_val != 1;
        });

    // Logic from slice_write_multi_core
    if (input.is_sharded()) {
        TT_FATAL(!has_step, "Step is not supported for sharded slice_write operation");
        if (input.layout() == Layout::ROW_MAJOR) {
            return SliceWriteRMShardedInputProgramFactory{};
        }
        if (input.layout() == Layout::TILE) {
            return SliceWriteTiledShardedInputProgramFactory{};
        }
        TT_THROW("Unsupported input memory layout for slice_write operation");

    } else {
        return SliceWriteRMInterleavedProgramFactory{};
    }
}

void SliceWriteDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& output_tensor = tensor_args.output;
    const auto output_padded_shape = output_tensor.padded_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to slice_write need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to slice_write need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.layout() == Layout::TILE || input_tensor.layout() == Layout::ROW_MAJOR,
        "Input tensor layout must be TILE or ROW_MAJOR but got {}",
        input_tensor.layout());
    TT_FATAL(
        input_tensor.padded_shape().rank() == args.slice_start.rank() &&
            output_padded_shape.rank() == args.slice_start.rank() && args.slice_start.rank() == args.slice_end.rank(),
        "Ranks of input tensor, output_tensor, slice start and slice end should be equal. Got {} {} {} {}",
        input_tensor.padded_shape().rank(),
        output_padded_shape.rank(),
        args.slice_start.rank(),
        args.slice_end.rank());
    for (uint32_t i = 0; i < output_padded_shape.rank(); i++) {
        TT_FATAL(
            args.slice_start[i] < output_padded_shape[i],
            "Start is outside the bounds of the output tensor for index {}. Got {}. Size {}",
            i,
            args.slice_start[i],
            output_padded_shape[i]);
        TT_FATAL(
            args.slice_end[i] <= output_padded_shape[i],
            "Ends {} must be less than or equal to the shape of the tensor {}",
            args.slice_end[i],
            output_padded_shape[i]);
        // Check if start shape is <= end shape
        TT_FATAL(
            args.slice_start[i] <= args.slice_end[i],
            "Slice start {} should be less than slice end {}",
            args.slice_start[i],
            args.slice_end[i]);
    }
    // If the input tensor is sharded, then rank should be 4
    TT_FATAL(
        !input_tensor.is_sharded() || input_tensor.padded_shape().rank() == 4,
        "Sharded input tensor should be of rank 4. Got {}",
        input_tensor.padded_shape().rank());
}

ttsl::hash::hash_t SliceWriteDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto factory = select_program_factory(args, tensor_args);
    // The interleaved factory derives every outer-dim start offset in its runtime args, which
    // override_runtime_arguments recomputes on a hit; only the last-dim start is compiled in.
    // The sharded factories bake the whole start into shared state, so they keep the full key.
    if (std::holds_alternative<SliceWriteRMInterleavedProgramFactory>(factory)) {
        return ttsl::hash::hash_objects_with_default_seed(
            factory.index(), args.slice_start[-1], args.step, tensor_args);
    }
    return ttsl::hash::hash_objects_with_default_seed(factory.index(), args, tensor_args);
}

tt::tt_metal::TensorSpec SliceWriteDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.output.tensor_spec();
}

Tensor SliceWriteDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.output;
}

std::vector<tt::tt_metal::TensorTopology> SliceWriteDeviceOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // In-place partial write into the caller's output. Its label stays while it still describes the data: an
    // input that is replicated, or sharded only along mesh axes the output is sharded along too, writes a slice
    // whose per-device differences the output's label already declares, so the output's distribution is as
    // labelled. An input sharded along a mesh axis on which the output is replicated writes a different slice on
    // every device there; a Replicate label kept on the output would then be false, and the flatbuffer serialiser
    // deduplicates the shards of a Replicate axis. The shared rule declines in that case and the hook returns {}
    // so the framework's union (the input's label) is applied -- to the returned handle and, since it aliases the
    // output's storage, to the caller's handle. See caller_owned_topology.hpp.
    if (const auto label = ttnn::operations::core::caller_owned_output_topology(
            tensor_args.output, {&tensor_args.input}, "slice_write")) {
        return {*label};
    }
    return {};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor slice_write(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& slice_start,
    const ttnn::Shape& slice_end,
    const ttnn::Shape& step) {
    using OperationType = ttnn::experimental::prim::SliceWriteDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .slice_start = slice_start,
        .slice_end = slice_end,
        .step = step,
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input_tensor, .output = output_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
