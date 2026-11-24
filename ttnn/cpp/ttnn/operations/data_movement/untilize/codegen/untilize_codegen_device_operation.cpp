// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "untilize_codegen_supported.hpp"
#include "untilize_codegen_cb_plan.hpp"

#include <tt_stl/reflection.hpp>

namespace ttnn::prim {

void UntilizeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // supported_by_codegen() is a scope predicate over layout/dtype/memory config, all of which
    // answer for a host tensor too, so the native op's structural preconditions must be asserted
    // here as well -- otherwise create_output_tensors()/the program factory reach input.device()
    // and input.buffer() first and fail on a null instead of on the established error.
    TT_FATAL(
        tensor_args.input.storage_type() == ttnn::StorageType::DEVICE, "Operands to untilize need to be on device!");
    TT_FATAL(tensor_args.input.is_allocated(), "Operands to untilize need to be allocated in buffers on device!");
    TT_FATAL(
        ttnn::operations::data_movement::untilize_codegen::supported_by_codegen(
            tensor_args.input, operation_attributes.output_mem_config),
        "UntilizeCodegenDeviceOperation invoked for a case not supported by the codegen implementation");
    if (tensor_args.gathered_prefix_tensor.has_value()) {
        const auto& ext = *tensor_args.gathered_prefix_tensor;
        const auto& input = tensor_args.input;
        TT_FATAL(
            ext.storage_type() == ttnn::StorageType::DEVICE && ext.buffer() != nullptr,
            "untilize: gathered_prefix_tensor must be allocated on device");
        TT_FATAL(
            ext.dtype() == tt::tt_metal::DataType::UINT32,
            "untilize: gathered_prefix_tensor must be UINT32, got {}",
            ext.dtype());
        TT_FATAL(ext.layout() == ttnn::Layout::ROW_MAJOR, "untilize: gathered_prefix_tensor must be ROW_MAJOR");
        TT_FATAL(ext.logical_volume() == 1, "untilize: gathered_prefix_tensor must hold one element");
        TT_FATAL(
            ext.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM && !ext.memory_config().is_sharded(),
            "untilize: gathered_prefix_tensor must be DRAM interleaved");
        TT_FATAL(
            !input.memory_config().is_sharded() && !operation_attributes.output_mem_config.is_sharded(),
            "untilize with gathered_prefix_tensor: input and output must be interleaved");
        const auto& ls = input.logical_shape();
        TT_FATAL(ls == input.padded_shape(), "untilize with gathered_prefix_tensor: the input must be tile-aligned");
        TT_FATAL(
            ls.rank() == 4 && ls[0] == 1 && ls[1] == 1,
            "untilize with gathered_prefix_tensor: input must be [1, 1, H, W], got {}",
            ls);
        const uint32_t nseg = operation_attributes.num_segments;
        const uint32_t slab = operation_attributes.gathered_slab_global;
        const uint32_t h = ls[2];
        TT_FATAL(
            nseg >= 1 && h % (nseg * tt::constants::TILE_HEIGHT) == 0,
            "untilize: input height {} must split into {} segments of whole tile rows",
            h,
            nseg);
        TT_FATAL(
            slab > 0 && slab % (nseg * tt::constants::TILE_HEIGHT) == 0 && h % slab == 0,
            "untilize: gathered_slab_global {} must be a multiple of {} x 32 dividing the height {}",
            slab,
            nseg,
            h);
    }
}

UntilizeCodegenDeviceOperation::spec_return_value_t UntilizeCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    const auto& input_tensor = tensor_args.input;
    DataType output_dtype = ttnn::operations::data_movement::untilize_output_dtype(input_tensor.dtype());

    const auto& logical_shape = input_tensor.logical_shape();
    const bool tile_aligned =
        logical_shape[-2] % tt::constants::TILE_HEIGHT == 0 && logical_shape[-1] % tt::constants::TILE_WIDTH == 0;
    if (!tile_aligned) {
        // Mirrors native UntilizeWithUnpaddingDeviceOperation::compute_output_specs: the writer
        // (build_with_unpadding) strips physical tile padding down to the logical shape, so the
        // output tensor is genuinely compact -- padded_shape == logical_shape, not the input's
        // tile-grid-rounded padded_shape.
        return TensorSpec(
            logical_shape,
            TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), operation_attributes.output_mem_config));
    }

    // Tile-aligned: same scheme as native UntilizeDeviceOperation::compute_output_specs. The
    // output keeps the input's PADDED shape (so the program factory can write in full physical
    // tile-rows), with only the logical_shape metadata cropping it back down on read.
    return TensorSpec(
        logical_shape,
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(Layout::ROW_MAJOR),
            operation_attributes.output_mem_config,
            logical_shape,
            input_tensor.padded_shape()));
}

UntilizeCodegenDeviceOperation::tensor_return_value_t UntilizeCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

// The CB tier is a function of the L1 that is free right now (choose_codegen_cb_plan samples it on
// every dispatch), so it is part of the cache key: an occupancy change that crosses a tier is a
// cache miss that rebuilds with the new depths, not a hit that replays a plan sized for L1 that is
// no longer free. Only the codegen tiers are ever keyed here -- a case for which no codegen plan
// fits is routed to the native prim by ttnn::untilize before it reaches this op.
ttsl::hash::hash_t UntilizeCodegenDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto chosen = untilize_codegen_detail::choose_codegen_cb_plan(operation_attributes, tensor_args);
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<UntilizeCodegenDeviceOperation>, operation_attributes, tensor_args, chosen.tier);
}

Tensor untilize_codegen(
    const Tensor& input,
    tt::tt_metal::MemoryConfig output_mem_config,
    const std::optional<Tensor>& gathered_prefix_tensor,
    uint32_t gathered_slab_global,
    uint32_t num_segments) {
    return ttnn::device_operation::launch<UntilizeCodegenDeviceOperation>(
        UntilizeCodegenOperationAttributes{
            .output_mem_config = std::move(output_mem_config),
            .gathered_slab_global = gathered_slab_global,
            .num_segments = num_segments},
        UntilizeCodegenTensorArgs{.input = input, .gathered_prefix_tensor = gathered_prefix_tensor});
}

}  // namespace ttnn::prim
