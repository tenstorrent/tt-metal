// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "gather_codegen_supported.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {
using namespace tt::tt_metal;

// Mirrors ops/gather/gather.py's `_gather_impl` builder-selection branch: row-buffered
// ("interleaved") whenever its exact three-CB footprint fits the device's real per-core L1 budget
// (gather_interleaved_fits_l1, see gather_codegen_program_factory.cpp), further split by output tile
// ("tiled") when tile-rows underfill the grid and Wt_index has column parallelism; otherwise the
// width-independent streaming fallback.
GatherCodegenDeviceOperation::program_factory_t GatherCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_index_tensor = tensor_args.input_index_tensor;

    if (gather_interleaved_fits_l1(input_tensor, input_index_tensor, attributes.Wt_input, attributes.Wt_index)) {
        auto* device = input_tensor.device();
        const auto grid_size = device->compute_with_storage_grid_size();
        const uint32_t max_cores = grid_size.x * grid_size.y;
        if (attributes.Wt_index >= 2 && attributes.Ht < max_cores) {
            return GatherCodegenProgramFactoryTiled{};
        }
        return GatherCodegenProgramFactoryInterleaved{};
    }
    return GatherCodegenProgramFactoryStreaming{};
}

void GatherCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // Belt-and-suspenders behind the free function's early gate (call_parity.routing): the scope
    // predicate, never is_demoted (perf-only, auto-branch-only). The gathered axis is -1 here
    // whatever the caller's dim was: pre_gather_transform_tensor has already transposed it last, so
    // this prim -- like ttnn::prim::gather -- is dim-agnostic. attributes.dim is the caller's
    // normalized dim against the caller's rank, which no longer indexes these 4D tensors.
    TT_FATAL(
        ttnn::operations::data_movement::gather::supported_by_codegen(
            tensor_args.input_tensor, /*dim=*/-1, tensor_args.input_index_tensor),
        "gather_codegen: input/index tensors are not supported by the codegen prim (see "
        "supported_by_codegen() -- TILE layout, bfloat16, non-sharded input/index only)");

    // Structural preconditions copied from GatherDeviceOperation::validate_on_program_cache_miss
    // (device/gather_device_operation.cpp): supported_by_codegen() only answers layout/dtype/
    // memory-config questions, all of which a host or deallocated tensor answers too.
    const auto& input_shape = tensor_args.input_tensor.logical_shape();
    const auto& index_shape = tensor_args.input_index_tensor.logical_shape();
    const auto input_rank = input_shape.rank();
    const auto index_rank = index_shape.rank();
    TT_FATAL(
        input_rank == index_rank,
        "gather_codegen: input and index tensor must have the same number of dimensions. Got input dim: {} and "
        "index dim: {}",
        input_rank,
        index_rank);
    for (int i = 0; i < input_rank - 1; ++i) {
        TT_FATAL(
            index_shape[i] <= input_shape[i],
            "gather_codegen: index tensor shape dimension {} must be less than or equal to input tensor shape "
            "dimension {}. Got index tensor shape: {} and input tensor shape: {}",
            i,
            i,
            index_shape[i],
            input_shape[i]);
    }
    TT_FATAL(
        tensor_args.input_index_tensor.dtype() == DataType::UINT32 ||
            tensor_args.input_index_tensor.dtype() == DataType::UINT16,
        "gather_codegen: index tensor must be of type UINT32 or UINT16. Got: {}",
        tensor_args.input_index_tensor.dtype());
    if (tensor_args.output_tensor.has_value()) {
        const auto output_shape = tensor_args.output_tensor.value().logical_shape();
        TT_FATAL(
            output_shape == tensor_args.input_index_tensor.logical_shape(),
            "gather_codegen: output tensor shape must be the same as index tensor shape. Got output tensor shape: "
            "{} and index tensor shape: {}",
            output_shape,
            tensor_args.input_index_tensor.logical_shape());
    }
    TT_FATAL(
        (tensor_args.input_tensor.buffer() != nullptr) && (tensor_args.input_index_tensor.buffer() != nullptr),
        "gather_codegen: operands need to be allocated in buffers on the device. Buffer is null.");
    TT_FATAL(
        tensor_args.input_tensor.storage_type() == StorageType::DEVICE,
        "gather_codegen: operation requires input to be on Device. Input storage type: {}",
        tensor_args.input_tensor.storage_type());
    TT_FATAL(
        tensor_args.input_index_tensor.storage_type() == StorageType::DEVICE,
        "gather_codegen: operation requires index to be on Device. Input storage type: {}",
        tensor_args.input_index_tensor.storage_type());
    // The descriptor launches on input_tensor.device() while embedding the other operands' raw
    // buffer addresses, so every operand must live on that same device.
    TT_FATAL(
        tensor_args.input_tensor.device() == tensor_args.input_index_tensor.device(),
        "gather_codegen: input and index tensors must be on the same device.");
    if (tensor_args.output_tensor.has_value()) {
        TT_FATAL(
            tensor_args.output_tensor.value().device() == tensor_args.input_tensor.device(),
            "gather_codegen: preallocated output tensor must be on the same device as the input.");
        TT_FATAL(
            tensor_args.output_tensor.value().buffer() != nullptr,
            "gather_codegen: preallocated output tensor has no device buffer allocated.");
    }
    TT_FATAL(attributes.sparse_grad == false, "gather_codegen: sparse gradient is not supported.");

    // Second belt-and-suspenders gate, this one over the output placement the free function checks
    // with supported_execution_controls(). Unlike most ported ops the codegen prim DOES carry the
    // requested memory config as an attribute, so its validation step can answer the same question
    // independently of the routing decision -- and must, because none of the three factories has
    // native's sharded path: output-spec computation below never synthesizes the shard_spec that
    // GatherDeviceOperation::compute_output_specs derives, and every CB and per-tile transfer is
    // sized from the output buffer's aligned TILE page (see make_tile_cb). Without this a direct
    // ttnn::prim::gather_codegen() call with a sharded config fails inside output-tensor creation
    // instead of here.
    TT_FATAL(
        !attributes.output_mem_config.is_sharded(),
        "gather_codegen: sharded output memory config is not supported (got memory layout {}); the codegen "
        "factories place output tiles through an interleaved TensorAccessor only.",
        attributes.output_mem_config.memory_layout());
    if (tensor_args.output_tensor.has_value()) {
        TT_FATAL(
            !tensor_args.output_tensor.value().memory_config().is_sharded(),
            "gather_codegen: sharded preallocated output tensor is not supported (got memory layout {}).",
            tensor_args.output_tensor.value().memory_config().memory_layout());
    }
}

GatherCodegenDeviceOperation::spec_return_value_t GatherCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value().tensor_spec();
    }
    const auto output_shape = tensor_args.input_index_tensor.logical_shape();
    return tt::tt_metal::TensorSpec(
        output_shape,
        TensorLayout(
            tensor_args.input_tensor.dtype(),
            PageConfig(tensor_args.input_tensor.layout()),
            attributes.output_mem_config));
}

GatherCodegenDeviceOperation::tensor_return_value_t GatherCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }
    const auto output_specs = compute_output_specs(attributes, tensor_args);
    return create_device_tensor(output_specs, tensor_args.input_tensor.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<GatherCodegenDeviceOperation::tensor_return_value_t>
GatherCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t&, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input_tensor;
    int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(input_tensor, output);
    return tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
}

Tensor gather_codegen(
    const Tensor& input_tensor,
    const int8_t dim,
    const Tensor& input_index_tensor,
    const bool sparse_grad,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    const auto geometry = compute_gather_geometry(input_tensor, input_index_tensor);
    return ttnn::device_operation::launch<GatherCodegenDeviceOperation>(
        GatherCodegenParams{
            dim,
            sparse_grad,
            output_memory_config,
            sub_core_grids,
            geometry.Ht,
            geometry.Wt_input,
            geometry.Wt_index,
            geometry.index_valid_h_last,
            geometry.index_valid_w_last,
            geometry.index_ht_per_batch},
        GatherCodegenInputs{input_tensor, input_index_tensor, output_tensor});
}

}  // namespace ttnn::prim
