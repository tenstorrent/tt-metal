// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/transpose/device/transpose_utils.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

constexpr uint32_t GATHER_WT_THRESHOLD = 60;

GatherDeviceOperation::program_factory_t GatherDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto input_tensor_shape = tensor_args.input_tensor.padded_shape();
    const auto input_index_tensor_shape = tensor_args.input_index_tensor.padded_shape();

    if (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR) {
        // Multi-core splits over W_index only (each core owns a slice of every output row);
        // W_input is mirrored in full to every core, so it gives no parallelism gain.
        const uint32_t W_index = input_index_tensor_shape[-1];
        constexpr uint32_t rm_w_threshold = GATHER_WT_THRESHOLD * tt::constants::TILE_WIDTH;
        if (W_index > rm_w_threshold) {
            return RmSingleRowMultiCore{};
        }
        return RmSingleRowSingleCore{};
    }

    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const uint32_t Wt_input = input_tensor_shape[3] / tile_width;
    const uint32_t Wt_index = input_index_tensor_shape[3] / tile_width;

    if (Wt_input > GATHER_WT_THRESHOLD || Wt_index > GATHER_WT_THRESHOLD) {
        // Use multi core for larger Wt
        return SingleRowMultiCore{};
    }
    return SingleRowSingleCore{};
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
    // Both input and index must share the same layout; mixed TILE/RM is unsupported because
    // the value/index/output streams are co-iterated stick-for-stick (or tile-for-tile).
    TT_FATAL(
        tensor_args.input_tensor.layout() == tensor_args.input_index_tensor.layout(),
        "Input tensor and index tensor must have the same layout. Got input layout: {} and index layout: {}",
        tensor_args.input_tensor.layout(),
        tensor_args.input_index_tensor.layout());
    TT_FATAL(
        tensor_args.input_tensor.layout() == Layout::TILE || tensor_args.input_tensor.layout() == Layout::ROW_MAJOR,
        "Gather only supports TILE or ROW_MAJOR layout. Current layout: {}",
        tensor_args.input_tensor.layout());

    // TILE-only: any user-provided output shard_spec must be tile-aligned. RM operates at
    // stick granularity, so sub-tile shards are valid there.
    if (tensor_args.input_tensor.layout() == Layout::TILE && attributes.output_mem_config.is_sharded() &&
        attributes.output_mem_config.shard_spec().has_value()) {
        const auto& spec = attributes.output_mem_config.shard_spec().value();
        TT_FATAL(
            spec.shape[0] % tt::constants::TILE_HEIGHT == 0 && spec.shape[1] % tt::constants::TILE_WIDTH == 0,
            "TILE sharded gather requires tile-aligned output shard shape; got ({}, {})",
            spec.shape[0],
            spec.shape[1]);
    }

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

    const auto output_shape = tensor_args.input_index_tensor.logical_shape();
    auto output_mem_config = attributes.output_mem_config;

    // Synthesize shard_spec when the user requested sharded output without one. Mirrors slice.
    if (output_mem_config.is_sharded() && !output_mem_config.shard_spec().has_value()) {
        const auto& index = tensor_args.input_index_tensor;
        std::optional<ShardSpec> derived;
        if (index.is_sharded() && index.memory_config().shard_spec().has_value() &&
            index.memory_config().memory_layout() == output_mem_config.memory_layout() &&
            index.logical_shape().rank() == output_shape.rank()) {
            auto adjusted = ttnn::operations::data_movement::transpose::adjust_shard_spec_to_shape(
                *index.memory_config().shard_spec(), index.logical_shape(), output_shape);
            if (adjusted.has_value()) {
                // TILE factories require tile-aligned shards; otherwise re-derive below.
                const bool tile_layout = tensor_args.input_tensor.layout() == Layout::TILE;
                const bool tile_aligned = adjusted->shape[0] % tt::constants::TILE_HEIGHT == 0 &&
                                          adjusted->shape[1] % tt::constants::TILE_WIDTH == 0;
                if (!tile_layout || tile_aligned) {
                    derived = std::move(adjusted);
                }
            }
        }
        if (!derived.has_value()) {
            // Preserve index orientation on cross-layout / sub-tile-fallthrough (mirrors #48025).
            std::optional<ShardOrientation> orientation_hint;
            if (index.shard_spec().has_value()) {
                orientation_hint = index.shard_spec()->orientation;
            }
            derived = ttnn::operations::data_movement::transpose::generate_transpose_shard_spec(
                tensor_args.input_tensor, output_shape, output_mem_config.memory_layout(), orientation_hint);
        }
        output_mem_config = MemoryConfig(output_mem_config.memory_layout(), output_mem_config.buffer_type(), derived);
    }

    // Output layout matches the input layout: TILE-in → TILE-out, RM-in → RM-out.
    return TensorSpec(
        output_shape,
        TensorLayout(
            tensor_args.input_tensor.dtype(), PageConfig(tensor_args.input_tensor.layout()), output_mem_config));
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
