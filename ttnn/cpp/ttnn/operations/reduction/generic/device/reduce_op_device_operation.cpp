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
    TT_FATAL((tensor_args.layout() == Layout::TILE), "Inputs to reduce must be tilized");
    // INT32 is supported via a dedicated SFPU compute kernel (reduce_sfpu.cpp + the
    // compute_kernel_lib::reduce_sfpu helper) for the combinations enabled by issue
    // #43736 / #26724 / #26726; everything else for INT32 routes through the FPU GMPOOL
    // path which only accepts {Float32, BFLOAT16, BFLOAT8_B, UINT32} and would silently
    // produce zeros for INT32.
    //
    // Currently supported (SFPU INT32) combinations:
    //   - MAX along H or W.  The repro from issue #21071 lives here.
    //   - MIN along H or W, lowered to MAX via the -MAX(-x) negate trick at the
    //     SFPU compute kernel level (reduce_sfpu.cpp's REDUCE_NEGATE define), so
    //     ttnn::reduction::common.cpp's existing negate=true path for reduce_min
    //     also goes through the SFPU INT32 kernel instead of GMPOOL.
    //   - SUM along H or W (issue #26724).  Cross-tile fold uses add_int_tile<Int32>.
    //   - HW for MAX, MIN, and SUM: decomposed at the prim layer in reduce_op.cpp
    //     into a W reduce followed by an H reduce, both INT32, so they reach this
    //     validator with dim ∈ {H, W} after the split.
    //
    // Out of scope (rejected here so they fail fast with a helpful message rather
    // than silently going through the GMPOOL zero-producing path):
    //   - AVG on INT32 (will be lowered to SUM with a 1/N post-multiply -- follow-up).
    const bool is_int32 = tensor_args.dtype() == DataType::INT32;
    const bool int32_op_supported = operation_attributes.math_op == tt::tt_metal::ReduceOpMath::MAX ||
                                    operation_attributes.math_op == tt::tt_metal::ReduceOpMath::SUM;
    const bool int32_dim_supported = operation_attributes.dim == tt::tt_metal::ReduceOpDim::H ||
                                     operation_attributes.dim == tt::tt_metal::ReduceOpDim::W;
    const bool is_int32_sfpu_supported = is_int32 && int32_op_supported && int32_dim_supported;
    TT_FATAL(
        tensor_args.dtype() == DataType::BFLOAT16 || tensor_args.dtype() == DataType::FLOAT32 ||
            tensor_args.dtype() == DataType::BFLOAT8_B || tensor_args.dtype() == DataType::UINT32 ||
            is_int32_sfpu_supported,
        "Only FLOAT32, BFLOAT16, BFLOAT8_B, and UINT32 are supported for generic reduction "
        "(plus INT32 for MAX/MIN/SUM along H or W via the SFPU path -- issues #43736 / "
        "#26724 / #26726; INT32 MIN reaches this validator as MAX with negate=true, and "
        "INT32 HW is split at the prim layer into a W-then-H sequence). "
        "Got dtype={}, math_op={}, dim={}, negate={}.",
        tensor_args.dtype(),
        operation_attributes.math_op,
        operation_attributes.dim,
        operation_attributes.negate);
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
    }

    if (operation_attributes.output_mem_config.nd_shard_spec().has_value()) {
        const auto& output_nd_shard_spec = *operation_attributes.output_mem_config.nd_shard_spec();
        const auto& output_shard_grid = output_nd_shard_spec.grid;
        const uint32_t output_tile_height = tensor_args.tensor_spec().tile().get_height();
        const uint32_t output_tile_width = tensor_args.tensor_spec().tile().get_width();
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
        if (output_nd_shard_spec.shard_shape.rank() >= 2) {
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

    return build_reduce_output_tensor_spec(
        output_shape,
        operation_attributes.output_dtype,
        operation_attributes.output_mem_config,
        tensor_args.memory_config(),
        operation_attributes.dim);
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
    float post_mul_scaler) {
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
            post_mul_scaler},
        input_tensor);
}

}  // namespace ttnn::prim
