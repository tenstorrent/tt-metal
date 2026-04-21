// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/hal.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

void validate_sharded_tile_buffer_for_gather(const Tensor& tensor, const char* tensor_name) {
    if (!tensor.memory_config().is_sharded()) {
        return;
    }
    const uint32_t page_size_bytes = tensor.buffer()->page_size();
    const uint32_t alignment_requirement = hal::get_l1_alignment();
    TT_FATAL(
        page_size_bytes == tensor.buffer()->aligned_page_size(),
        "gather: {} TILE page size {} bytes (dtype {}) must equal aligned page size {} bytes for L1 NoC alignment ({} "
        "bytes)",
        tensor_name,
        page_size_bytes,
        tensor.dtype(),
        tensor.buffer()->aligned_page_size(),
        alignment_requirement);
}

}  // namespace

constexpr uint32_t GATHER_WT_THRESHOLD = 60;

GatherDeviceOperation::program_factory_t GatherDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Calculate Wt to decide which program factory to use
    const auto input_tensor_shape = tensor_args.input_tensor.padded_shape();
    const auto input_index_tensor_shape = tensor_args.input_index_tensor.padded_shape();
    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const uint32_t Wt_input = input_tensor_shape[3] / tile_width;
    const uint32_t Wt_index = input_index_tensor_shape[3] / tile_width;

    if (Wt_input > GATHER_WT_THRESHOLD || Wt_index > GATHER_WT_THRESHOLD) {
        // Use GatherProgramFactorySingleRowMultiCore for larger Wt
        return GatherProgramFactorySingleRowMultiCore{};
    }
    return GatherProgramFactorySingleRowSingleCore{};
}

void GatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // Validate shapes of input and output tensors
    const auto input_tensor_shape = tensor_args.input_tensor.logical_shape();
    const auto input_tensor_rank = input_tensor_shape.rank();
    const auto input_index_tensor_shape = tensor_args.input_index_tensor.logical_shape();
    const auto input_index_tensor_rank = input_index_tensor_shape.rank();

    TT_FATAL(
        input_tensor_rank == input_index_tensor_rank,
        "Input and index tensor must have the same number of dimensions. Got input dim: {} and index dim: {}",
        input_tensor_rank,
        input_index_tensor_rank);

    if (tensor_args.output_tensor.has_value()) {
        const auto output_tensor_shape = tensor_args.output_tensor.value().logical_shape();
        TT_FATAL(
            output_tensor_shape == input_index_tensor_shape,
            "Output tensor shape must be the same as index tensor shape. Got output tensor shape: {} and index "
            "tensor shape: {}",
            output_tensor_shape,
            input_index_tensor_shape);
    }
    TT_FATAL(
        tensor_args.input_index_tensor.dtype() == DataType::UINT32 ||
            tensor_args.input_index_tensor.dtype() == DataType::UINT16,
        "Index tensor must be of type UINT32 or UINT16. Got: {}",
        tensor_args.input_index_tensor.dtype());

    for (int i = 0; i < input_tensor_rank - 1; ++i) {
        // Validate all dimensions except the last one, as the tensor has been transposed
        // to move the gather dimension to the last position.
        // Improvement idea: Consider removing transposition and handling arbitrary dimensions directly in the kernel.
        TT_FATAL(
            input_index_tensor_shape[i] <= input_tensor_shape[i],
            "Index tensor shape dimension {} must be less than or equal to input tensor shape dimension {}. Got "
            "index tensor shape: {} and input tensor shape: {}",
            i,
            i,
            input_index_tensor_shape[i],
            input_tensor_shape[i]);
    }

    // Interleaved, legacy sharded, and ND-sharded TILE tensors are supported; TensorAccessor resolves logical tile
    // page_id to physical addresses. Require L1-aligned tile pages for NoC reads/writes on sharded buffers.
    validate_sharded_tile_buffer_for_gather(tensor_args.input_tensor, "input_tensor");
    validate_sharded_tile_buffer_for_gather(tensor_args.input_index_tensor, "input_index_tensor");
    if (tensor_args.output_tensor.has_value()) {
        validate_sharded_tile_buffer_for_gather(tensor_args.output_tensor.value(), "output_tensor");
    }

    TT_FATAL(
        tensor_args.input_tensor.layout() == Layout::TILE,
        "The input tensor must be in tiled format. Current layout: {}",
        tensor_args.input_tensor.layout());
    TT_FATAL(
        tensor_args.input_index_tensor.layout() == Layout::TILE,
        "The input index tensor must be in tiled format. Current layout: {}",
        tensor_args.input_index_tensor.layout());

    TT_FATAL(
        (tensor_args.input_tensor.buffer() != nullptr) && (tensor_args.input_index_tensor.buffer() != nullptr),
        "Operands need to be allocated in buffers on the device. Buffer is null.");
    TT_FATAL(
        tensor_args.input_tensor.storage_type() == StorageType::DEVICE,
        "Operation requires input to be on Device. Input storage type: {}",
        tensor_args.input_tensor.storage_type());
    TT_FATAL(
        tensor_args.input_index_tensor.storage_type() == StorageType::DEVICE,
        "Operation requires input to be on Device. Input storage type: {}",
        tensor_args.input_index_tensor.storage_type());

    TT_FATAL(attributes.sparse_grad == false, "Sparse gradient is not supported yet.");
}

GatherDeviceOperation::spec_return_value_t GatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value().tensor_spec();
    }
    const auto input_index_tensor_shape = tensor_args.input_index_tensor.logical_shape();
    auto output_layout =
        TensorLayout(tensor_args.input_tensor.dtype(), PageConfig(Layout::TILE), attributes.output_mem_config);
    auto output_padded_shape = output_layout.compute_padded_shape(input_index_tensor_shape);
    return TensorSpec(
        input_index_tensor_shape,
        TensorLayout::fromPaddedShape(
            tensor_args.input_tensor.dtype(),
            PageConfig(Layout::TILE),
            attributes.output_mem_config,
            input_index_tensor_shape,
            output_padded_shape));
}

GatherDeviceOperation::tensor_return_value_t GatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }
    const auto output_specs = compute_output_specs(attributes, tensor_args);
    return create_device_tensor(output_specs, tensor_args.input_tensor.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<GatherDeviceOperation::tensor_return_value_t>
GatherDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*op_attr*/, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input_tensor;
    int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(input_tensor, output);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

Tensor gather(
    const Tensor& input_tensor,
    const int8_t dim,
    const Tensor& input_index_tensor,
    const bool sparse_grad,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& output_tensors,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::device_operation::launch<GatherDeviceOperation>(
        GatherParams{dim, sparse_grad, output_memory_config, sub_core_grids},
        GatherInputs{input_tensor, input_index_tensor, output_tensors});
}

}  // namespace ttnn::prim
