// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {

ReduceDeviceOperation::program_factory_t ReduceDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    auto parallelization_strategy = get_parallelization_strategy(tensor_args, operation_attributes.dim);

    switch (parallelization_strategy) {
        case ReduceOpParallelizationStrategy::MULTI_CORE_H:
            return ReduceDeviceOperation::ReduceMultiCoreHProgramFactory{};
        case ReduceOpParallelizationStrategy::MULTI_CORE_W:
            return ReduceDeviceOperation::ReduceMultiCoreWProgramFactory{};
        case ReduceOpParallelizationStrategy::MULTI_CORE_HW:
        case ReduceOpParallelizationStrategy::SINGLE_CORE_HW:
            return ReduceDeviceOperation::ReduceSingleCoreHwProgramFactory{};
        default: TT_THROW("Unsupported parallelization strategy");
    }
}

void ReduceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    TT_FATAL(
        tensor_args.storage_type() == StorageType::DEVICE,
        "Operands to reduce need to be on device! Got storage type: {}",
        tensor_args.storage_type());
    TT_FATAL(tensor_args.buffer() != nullptr, "Operands to reduce need to be allocated in buffers on device!");
    // Dense RM path is only selected on the host for ttnn.mean-style dispatch (AVG over W/H on 4D BF16/FLOAT32,
    // interleaved I/O). It is lowered to PoolType::SUM + scaler before launch; see reduce_op.cpp.
    TT_FATAL(
        !(operation_attributes.row_major_w_dense_path && operation_attributes.row_major_h_dense_path),
        "Only one of row_major_w_dense_path / row_major_h_dense_path may be set");
    if (operation_attributes.row_major_w_dense_path || operation_attributes.row_major_h_dense_path) {
        const auto expected_dim =
            operation_attributes.row_major_w_dense_path ? tt::tt_metal::ReduceOpDim::W : tt::tt_metal::ReduceOpDim::H;
        const char* path_name =
            operation_attributes.row_major_w_dense_path ? "row_major_w_dense_path" : "row_major_h_dense_path";
        TT_FATAL(
            operation_attributes.dim == expected_dim,
            "{} only supports {}-dim reduce, got dim {}",
            path_name,
            expected_dim,
            operation_attributes.dim);
        TT_FATAL(tensor_args.layout() == Layout::ROW_MAJOR, "{} requires ROW_MAJOR input", path_name);
        TT_FATAL(
            tensor_args.logical_shape().rank() == 4,
            "{} requires 4D input, got rank {}",
            path_name,
            tensor_args.logical_shape().rank());
        TT_FATAL(!operation_attributes.negate, "{} does not support negate (min-reduce) yet", path_name);
        TT_FATAL(
            tensor_args.dtype() == DataType::BFLOAT16 || tensor_args.dtype() == DataType::FLOAT32,
            "{} only supports BFLOAT16 and FLOAT32, got {}",
            path_name,
            tensor_args.dtype());
        TT_FATAL(
            operation_attributes.math_op == tt::tt_metal::ReduceOpMath::SUM ||
                operation_attributes.math_op == tt::tt_metal::ReduceOpMath::MAX,
            "{}: math_op must be SUM (mean lowered from AVG) or MAX, got {}",
            path_name,
            operation_attributes.math_op);
    } else {
        TT_FATAL((tensor_args.layout() == Layout::TILE), "Inputs to reduce must be tilized");
        TT_FATAL(
            tensor_args.dtype() == DataType::BFLOAT16 || tensor_args.dtype() == DataType::FLOAT32 ||
                tensor_args.dtype() == DataType::BFLOAT8_B || tensor_args.dtype() == DataType::UINT32,
            "Only FLOAT32, BFLOAT16, BFLOAT8_B, and UINT32 are supported for generic reduction - got {}",
            tensor_args.dtype());
    }
    validate_reduce_sharded_buffer_types(tensor_args.memory_config(), operation_attributes.output_mem_config, "reduce");
    const auto device_grid_size = tensor_args.device()->compute_with_storage_grid_size();
    TT_FATAL(
        device_grid_size.x > 0 && device_grid_size.y > 0,
        "Device compute grid must be non-empty, got ({}, {})",
        device_grid_size.x,
        device_grid_size.y);
    const CoreRangeSet device_grid =
        num_cores_to_corerangeset(device_grid_size.x * device_grid_size.y, device_grid_size, false);

    const CoreRangeSet program_grid = operation_attributes.sub_core_grids.value_or(device_grid);
    TT_FATAL(!program_grid.ranges().empty(), "Program core grid must not be empty");
    TT_FATAL(
        device_grid.contains(program_grid),
        "Program core grid {} must be contained in device grid {}",
        program_grid,
        device_grid);

    if (tensor_args.shard_spec().has_value()) {
        const auto& in_shard = tensor_args.shard_spec().value();
        const auto& input_shard_grid = in_shard.grid;
        TT_FATAL(
            program_grid.contains(input_shard_grid),
            "Input shard grid {} must be contained in program core grid {}",
            input_shard_grid,
            program_grid);
        const uint32_t tile_height = tensor_args.tensor_spec().tile().get_height();
        const uint32_t tile_width = tensor_args.tensor_spec().tile().get_width();
        if (!operation_attributes.row_major_w_dense_path && !operation_attributes.row_major_h_dense_path) {
            // Tilized paths require tile-aligned shard dimensions.
            TT_FATAL(
                in_shard.shape[0] > 0 && in_shard.shape[1] > 0,
                "Sharded reduce input: shard face shape must be positive, got [{}, {}]",
                in_shard.shape[0],
                in_shard.shape[1]);
            TT_FATAL(
                in_shard.shape[0] % tile_height == 0,
                "Sharded reduce input: shard_shape[0]={} must be tile-height-aligned ({} px per tile face row)",
                in_shard.shape[0],
                tile_height);
            TT_FATAL(
                in_shard.shape[1] % tile_width == 0,
                "Sharded reduce input: shard_shape[1]={} must be tile-width-aligned ({} px per tile face col)",
                in_shard.shape[1],
                tile_width);
        } else if (operation_attributes.row_major_h_dense_path) {
            // H reduce RM (ROW_MAJOR) with WIDTH_SHARDED. Two constraints:
            //   1. Shard page (shard_W * elem_bytes) must be 16B aligned so that every row
            //      in the L1 shard starts at a 16B boundary for NOC DMA. Tile-width alignment is
            //      a TILE-layout constraint and is not the right check for ROW_MAJOR buffers.
            //   2. W_logical must be exactly divisible by shard_W so all shards are full (no
            //      partial last shard). Partial shards need per-core W_logical which the factory
            //      does not currently support.
            const uint32_t shard_W = in_shard.shape[1];
            const uint32_t elem_bytes = tensor_args.dtype() == tt::tt_metal::DataType::FLOAT32 ? 4u : 2u;
            const uint32_t shard_page_bytes = shard_W * elem_bytes;
            TT_FATAL(
                shard_W > 0 && shard_page_bytes % 16 == 0,
                "H reduce RM (dense) WIDTH_SHARDED: shard page size ({} cols * {} bytes = {} bytes) "
                "must be 16B aligned",
                shard_W,
                elem_bytes,
                shard_page_bytes);
            const uint32_t W_logical = tensor_args.logical_shape()[3];
            TT_FATAL(
                W_logical % shard_W == 0,
                "H reduce RM (dense) WIDTH_SHARDED: W_logical={} must be divisible by shard_W={} "
                "(partial last shard not supported)",
                W_logical,
                shard_W);
        }
    }

    if (operation_attributes.output_mem_config.nd_shard_spec().has_value()) {
        const auto out_spec = compute_output_specs(operation_attributes, tensor_args);
        const auto& output_nd_shard_spec = *out_spec.memory_config().nd_shard_spec();
        const auto& output_shard_grid = output_nd_shard_spec.grid;
        const uint32_t output_tile_height = out_spec.tile().get_height();
        const uint32_t output_tile_width = out_spec.tile().get_width();

        TT_FATAL(
            program_grid.contains(output_shard_grid),
            "Output shard grid {} must be contained in program core grid {}",
            output_shard_grid,
            program_grid);
        TT_FATAL(
            device_grid.contains(output_shard_grid),
            "Output shard grid {} must be contained in device grid {}",
            output_shard_grid,
            device_grid);
        // Tile-alignment checks only apply to TILE layout output. RM paths produce
        // ROW_MAJOR output whose shard shape is in element units, not tile units,
        // so shard_shape[-1]=1 (after a W-reduce) is perfectly valid.
        const bool is_rm_path =
            operation_attributes.row_major_w_dense_path || operation_attributes.row_major_h_dense_path;
        if (!is_rm_path && output_nd_shard_spec.shard_shape.rank() >= 2) {
            TT_FATAL(
                output_nd_shard_spec.shard_shape[-2] > 0 && output_nd_shard_spec.shard_shape[-1] > 0,
                "ND sharded output: last-2 shard dims must be positive, got [..., {}, {}] (height/width in "
                "shard_shape)",
                output_nd_shard_spec.shard_shape[-2],
                output_nd_shard_spec.shard_shape[-1]);
            TT_FATAL(
                output_nd_shard_spec.shard_shape[-2] % output_tile_height == 0,
                "ND sharded output: shard_shape[-2]={} must be tile-height-aligned ({}) for tilized output",
                output_nd_shard_spec.shard_shape[-2],
                output_tile_height);
            TT_FATAL(
                output_nd_shard_spec.shard_shape[-1] % output_tile_width == 0,
                "ND sharded output: shard_shape[-1]={} must be tile-width-aligned ({}) for tilized output",
                output_nd_shard_spec.shard_shape[-1],
                output_tile_width);
        }
    }
}

ReduceDeviceOperation::spec_return_value_t ReduceDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_shape = tensor_args.logical_shape();
    switch (operation_attributes.dim) {
        case tt::tt_metal::ReduceOpDim::H: output_shape[2] = 1; break;
        case tt::tt_metal::ReduceOpDim::W: output_shape[3] = 1; break;
        case tt::tt_metal::ReduceOpDim::HW:
            output_shape[2] = 1;
            output_shape[3] = 1;
            break;
    }

    const tt::tt_metal::Layout output_layout =
        (operation_attributes.row_major_w_dense_path || operation_attributes.row_major_h_dense_path)
            ? tt::tt_metal::Layout::ROW_MAJOR
            : tt::tt_metal::Layout::TILE;
    return build_reduce_output_tensor_spec(
        output_shape,
        operation_attributes.output_dtype,
        operation_attributes.output_mem_config,
        tensor_args.memory_config(),
        operation_attributes.dim,
        output_layout);
}

ReduceDeviceOperation::tensor_return_value_t ReduceDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.device());
}

ttsl::hash::hash_t ReduceDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<ReduceDeviceOperation>(
        operation_attributes.math_op,
        operation_attributes.dim,
        operation_attributes.scaler,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        operation_attributes.compute_kernel_config,
        operation_attributes.sub_core_grids,
        operation_attributes.negate,
        operation_attributes.post_mul_scaler,
        operation_attributes.row_major_w_dense_path,
        operation_attributes.row_major_h_dense_path,
        program_factory.index(),
        tensor_args.dtype(),
        tensor_args.memory_config(),
        tensor_args.padded_shape(),
        tensor_args.tensor_spec().tile());
}

ttnn::Tensor reduce(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scaler,
    const MemoryConfig& output_mem_config,
    const std::optional<DataType>& output_dtype,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    bool negate,
    float post_mul_scaler,
    bool row_major_w_dense_path,
    bool row_major_h_dense_path) {
    return ttnn::device_operation::launch<ReduceDeviceOperation>(
        ReduceParams{
            reduce_math,
            reduce_dim,
            scaler,
            output_mem_config,
            output_dtype.value_or(input_tensor.dtype()),
            compute_kernel_config,
            sub_core_grids,
            negate,
            post_mul_scaler,
            row_major_w_dense_path,
            row_major_h_dense_path},
        input_tensor);
}

}  // namespace ttnn::prim
