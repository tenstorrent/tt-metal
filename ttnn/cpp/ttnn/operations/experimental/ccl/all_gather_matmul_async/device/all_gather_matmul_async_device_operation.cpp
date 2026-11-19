// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_device_operation.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/device_operation_detail.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"

/* All Gather Matmul fusion includes */
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation.hpp"

#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.hpp"
namespace ttnn::experimental::prim {

void AllGatherMatmulAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void AllGatherMatmulAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;

    TT_FATAL(
        input_tensor.logical_shape().rank() == 4 && weight_tensor.logical_shape().rank() == 4,
        "AllGatherMatmulAsync requires input tensors to be of rank 4");
    const auto& ag = operation_attributes.all_gather_async_attributes;
    const auto& mm = operation_attributes.matmul;
    if (ag.topology == ttnn::ccl::Topology::Linear) {
        // LINEAR_FULL_SLICE_V1 is deliberately limited to the interleaved single-sweep
        // contract. Sharded/SINGLE receivers and repeated output sweeps need another ABI.
        const auto& mesh_shape = input_tensor.device()->shape();
        TT_FATAL(
            mesh_shape.dims() == 2 && mesh_shape[0] == 1 && mesh_shape[1] == 4 &&
                input_tensor.device_storage().get_coords().size() == 4 && ag.ring_size == 4 &&
                !ag.cluster_axis.has_value(),
            "Linear AGMM requires the complete 1x4 tensor placement");
        TT_FATAL(
            input_tensor.device() == weight_tensor.device() &&
                std::ranges::equal(input_tensor.device_storage().get_coords(), weight_tensor.device_storage().get_coords()),
            "Linear AGMM operands must occupy the same complete mesh");
        TT_FATAL(
            input_tensor.dtype() == DataType::BFLOAT16 &&
                (weight_tensor.dtype() == DataType::BFLOAT16 || weight_tensor.dtype() == DataType::BFLOAT8_B) &&
                mm.output_dtype.value_or(input_tensor.dtype()) == DataType::BFLOAT16,
            "Linear AGMM initially supports BF16 activations/output and BF16 or BFP8_B weights");
        TT_FATAL(
            ag.dim == 3 && ag.num_links == 1 && ag.num_workers_per_link == 2 && !ag.sub_device_id.has_value() &&
                !mm.sub_device_id.has_value(),
            "Linear AGMM requires width gather, one link, two workers per direction, and the default subdevice");
        TT_FATAL(ag.barrier_semaphore.has_value(), "Linear AGMM requires a global admission barrier");
        TT_FATAL(
            !mm.transpose_a && !mm.transpose_b && !mm.untilize_out && !tensor_args.bias.has_value() &&
                !mm.global_cb.has_value() && mm.prefetcher_pipes.empty() && !mm.in0_column_offset.has_value(),
            "Linear AGMM requires nontransposed A/B, tiled output, no bias, and ordinary tensor operands");
        const auto interleaved_dram = [](const MemoryConfig& config) {
            return config.memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                   config.buffer_type() == BufferType::DRAM;
        };
        TT_FATAL(
            interleaved_dram(input_tensor.memory_config()) && interleaved_dram(weight_tensor.memory_config()) &&
                interleaved_dram(ag.output_mem_config) && interleaved_dram(mm.output_mem_config),
            "Linear AGMM requires interleaved DRAM A, B, gathered scratch and matmul output");
        const auto ordinary_tiles = [](const Tensor& tensor) {
            const auto tile = tensor.tensor_spec().tile();
            return tensor.layout() == Layout::TILE && tile.get_height() == 32 && tile.get_width() == 32 &&
                   !tile.get_transpose_of_faces() && !tile.get_transpose_within_face();
        };
        TT_FATAL(
            ordinary_tiles(input_tensor) && ordinary_tiles(weight_tensor) &&
                (!mm.output_tile.has_value() ||
                 (mm.output_tile->get_height() == 32 && mm.output_tile->get_width() == 32 &&
                  !mm.output_tile->get_transpose_of_faces() && !mm.output_tile->get_transpose_within_face())),
            "Linear AGMM requires ordinary 32x32 tiles");
        if (tensor_args.persistent_output_buffer.has_value()) {
            const auto& scratch = tensor_args.persistent_output_buffer.value();
            TT_FATAL(
                scratch.device() == input_tensor.device() &&
                    std::ranges::equal(
                        scratch.device_storage().get_coords(), input_tensor.device_storage().get_coords()),
                "Linear AGMM persistent scratch must occupy the complete operand mesh");
            auto expected_logical = input_tensor.logical_shape();
            auto expected_padded = input_tensor.padded_shape();
            expected_logical[3] *= ag.ring_size;
            expected_padded[3] *= ag.ring_size;
            TT_FATAL(
                scratch.logical_shape() == expected_logical && scratch.padded_shape() == expected_padded,
                "Linear AGMM persistent scratch must match the exact gathered logical and padded shapes");
            TT_FATAL(
                interleaved_dram(scratch.memory_config()) && ordinary_tiles(scratch),
                "Linear AGMM persistent scratch must be interleaved DRAM with ordinary tiles");
            TT_FATAL(
                scratch.buffer() != input_tensor.buffer() && scratch.buffer() != weight_tensor.buffer(),
                "Linear AGMM scratch must not alias either operand");
        }
        TT_FATAL(
            input_tensor.padded_shape()[0] == 1 && input_tensor.padded_shape()[1] == 1 &&
                input_tensor.padded_shape()[2] == 32 && input_tensor.padded_shape()[3] == 1280 &&
                input_tensor.logical_shape()[3] == 1280 && weight_tensor.padded_shape()[0] == 1 &&
                weight_tensor.padded_shape()[1] == 1 && weight_tensor.padded_shape()[2] == 5120 &&
                weight_tensor.logical_shape()[2] == 5120,
            "Linear AGMM initially supports one physical M tile, local K1280 and full K5120 without outer batches");
        TT_FATAL(mm.program_config.has_value(), "Linear AGMM requires an explicit multicast program config");
        std::visit(
            [&](const auto& config) {
                using Config = std::decay_t<decltype(config)>;
                if constexpr (
                    std::is_same_v<Config, operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig> ||
                    std::is_same_v<Config, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                    TT_FATAL(
                        (config.in0_block_w == 4 || config.in0_block_w == 8) && config.per_core_M == 1 &&
                            config.out_block_h == config.per_core_M && config.per_core_N > 0 &&
                            config.out_block_w == config.per_core_N && config.compute_with_storage_grid_size.x > 0 &&
                            config.compute_with_storage_grid_size.y > 0 && !config.allowed_worker_cores.has_value(),
                        "Linear AGMM requires Kblock4/8 and exactly one output h/w block per core on a rectangular "
                        "grid");
                    const auto n_blocks =
                        (weight_tensor.padded_shape()[3] / 32 + config.per_core_N - 1) / config.per_core_N;
                    if constexpr (std::is_same_v<
                                      Config,
                                      operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                        TT_FATAL(
                            config.mcast_in0 && !config.gather_in0 && !config.stream_in1 && !config.untilize_out &&
                                config.hop_cores.num_cores() == 0 && config.num_global_cb_receivers == 0 &&
                                n_blocks <=
                                    config.compute_with_storage_grid_size.x * config.compute_with_storage_grid_size.y,
                            "Linear AGMM 1D requires ordinary mcast_in0 with enough cores for a single N sweep");
                    } else {
                        TT_FATAL(
                            !config.transpose_mcast && n_blocks <= config.compute_with_storage_grid_size.x,
                            "Linear AGMM 2D requires untransposed multicast with enough columns for a single N sweep");
                    }
                } else {
                    TT_THROW("Linear AGMM requires a 1D mcast_in0 or 2D multicast program config");
                }
            },
            mm.program_config.value());
    }

    // A buffer repeated across input, weight and bias leaves the cached program with no buffer bindings, so later
    // cache hits would keep the first call's addresses (the WorkloadDescriptor adapter has no slow-path rebuild).
    TT_FATAL(
        input_tensor.buffer() != weight_tensor.buffer(),
        "AllGatherMatmulAsync requires input_tensor and weight_tensor to be distinct buffers.");
    if (tensor_args.bias.has_value()) {
        TT_FATAL(
            tensor_args.bias->buffer() != input_tensor.buffer() && tensor_args.bias->buffer() != weight_tensor.buffer(),
            "AllGatherMatmulAsync requires bias to be a buffer distinct from input_tensor and weight_tensor.");
    }
    if (tensor_args.persistent_output_buffer.has_value()) {
        const auto& all_gather_output_tensor = tensor_args.persistent_output_buffer.value();
        // All Gather validate
        AllGatherAsyncDeviceOperation::validate_on_program_cache_miss(
            operation_attributes.all_gather_async_attributes, operation_attributes.all_gather_async_tensor_args);

        // Matmul validate
        ttnn::prim::MatmulDeviceOperation::validate_on_program_cache_miss(
            operation_attributes.matmul,
            {.input_tensors = {all_gather_output_tensor, weight_tensor},
             .optional_input_tensors = {tensor_args.bias},
             .optional_output_tensors = {}});
    }
    // All Gather Matmul validate
    TT_FATAL(
        operation_attributes.all_gather_async_attributes.dim == 3,
        "AllGatherMatmulAsync requires dim=3 for the AllGather operations.");
    TT_FATAL(
        input_tensor.padded_shape()[0] == 1 && input_tensor.padded_shape()[1] == 1,
        "AllGatherMatmulAsync requires input tensor to have batch size of 1.");

    std::visit(
        [&](const auto& config) {
            using ProgramConfigType = std::decay_t<decltype(config)>;
            if (not(std::is_same_v<
                        ProgramConfigType,
                        operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig> ||
                    std::
                        is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>)) {
                TT_THROW(
                    "Unsupported MatmulProgramConfig type for AllGatherMatmulAsync. Needs to be 1D or 2D Multicast.");
            }
        },
        operation_attributes.matmul.program_config.value());  // TODO: migrate this code to use new matmul API.

    if (tensor_args.persistent_output_buffer.has_value()) {
        const auto& all_gather_output_tensor = tensor_args.persistent_output_buffer.value();
        const auto& shard_spec = all_gather_output_tensor.shard_spec();
        if (shard_spec.has_value()) {
            const uint32_t num_all_gather_output_shards =
                shard_builder::get_sharding_core_count(all_gather_output_tensor);
            TT_FATAL(
                operation_attributes.all_gather_async_attributes.ring_size ==
                    num_all_gather_output_shards,  // TODO: migrate this code to use new all_gather_async API.
                "AllGatherMatmulAsync requires number of tensor slices to equal the number of output shards of the "
                "all_gather.");
        }
    }
}

AllGatherMatmulAsyncDeviceOperation::spec_return_value_t AllGatherMatmulAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // All Gather shape
    auto all_gather_output_specs = AllGatherAsyncDeviceOperation::compute_output_specs(
        operation_attributes.all_gather_async_attributes, operation_attributes.all_gather_async_tensor_args);

    // Matmul shape
    auto matmul_output_specs = ttnn::prim::MatmulDeviceOperation::compute_output_specs(
        operation_attributes.matmul,
        {.input_tensors = {tensor_args.input_tensor, tensor_args.weight_tensor},
         .optional_input_tensors = {},
         .optional_output_tensors = {}})[0];

    return {all_gather_output_specs, matmul_output_specs};
}

AllGatherMatmulAsyncDeviceOperation::tensor_return_value_t AllGatherMatmulAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    std::vector<std::optional<Tensor>> optional_output_tensors = {tensor_args.persistent_output_buffer};
    // All Gather output tensor
    auto all_gather_output_tensor = AllGatherAsyncDeviceOperation::create_output_tensors(
        operation_attributes.all_gather_async_attributes, operation_attributes.all_gather_async_tensor_args);

    // Matmul output tensor
    auto matmul_output_tensor = ttnn::prim::MatmulDeviceOperation::create_output_tensors(
        operation_attributes.matmul,
        {.input_tensors = {all_gather_output_tensor, tensor_args.weight_tensor},
         .optional_input_tensors = {},
         .optional_output_tensors = {}})[0];

    return {all_gather_output_tensor, matmul_output_tensor};
}

std::vector<tt::tt_metal::TensorTopology> AllGatherMatmulAsyncDeviceOperation::compute_output_topologies(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using tt::tt_metal::TensorTopology;
    // {all_gather output, matmul output}. The all_gather label is AllGatherAsyncDeviceOperation's over the attributes
    // and inputs it is created from (the input's own label when there is no honest gather label). The matmul of the
    // gathered activation by the weight takes the union label of {gathered activation, weight, bias}, as launch()
    // would give a plain matmul.
    const auto gathered = AllGatherAsyncDeviceOperation::compute_output_topologies(
        operation_attributes.all_gather_async_attributes, operation_attributes.all_gather_async_tensor_args);
    std::vector<std::reference_wrapper<const TensorTopology>> operands{
        std::cref(gathered.front()), std::cref(tensor_args.weight_tensor.tensor_topology())};
    if (tensor_args.bias.has_value()) {
        operands.emplace_back(tensor_args.bias->tensor_topology());
    }
    return {
        gathered.front(),
        ttnn::operations::ccl::common::union_topology(
            operands, tensor_args.input_tensor.tensor_topology().mesh_coords())};
}

ttsl::hash::hash_t AllGatherMatmulAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllGatherMatmulAsyncDeviceOperation::compute_program_hash is called");

    auto subdevice_id = operation_attributes.all_gather_async_attributes.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    return tt::tt_metal::operation::hash_operation<AllGatherMatmulAsyncDeviceOperation>(
        operation_attributes.all_gather_async_attributes.dim,
        operation_attributes.all_gather_async_attributes.num_links,
        operation_attributes.all_gather_async_attributes.ring_size,
        operation_attributes.all_gather_async_attributes.output_mem_config,
        operation_attributes.all_gather_async_attributes.topology,
        operation_attributes.all_gather_async_attributes.cluster_axis,
        operation_attributes.all_gather_async_attributes.barrier_semaphore.has_value(),
        operation_attributes.all_gather_async_attributes.using_persistent_buffers,
        operation_attributes.all_gather_async_attributes.chunks_per_sync,
        operation_attributes.all_gather_async_attributes.num_workers_per_link,
        operation_attributes.all_gather_async_attributes.num_buffers_per_channel,
        operation_attributes.matmul,
        operation_attributes.all_gather_core_grid_offset,
        subdevice_core_range_set,
        tensor_args);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::AllGatherMatmulAsyncDeviceOperation::tensor_return_value_t all_gather_matmul_async(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const tt::tt_metal::CoreCoord all_gather_core_grid_offset,
    const std::optional<const Tensor>& bias,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config_ag,
    const ttnn::ccl::Topology topology,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<MemoryConfig>& memory_config_mm,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::CoreGrid> core_grid,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    using OperationType = ttnn::experimental::prim::AllGatherMatmulAsyncDeviceOperation;

    /* All Gather setup */
    const auto [all_gather_async_operation_attributes, all_gather_async_tensor_args] =
        ttnn::experimental::prim::all_gather_async_build_operation_args(
            input_tensor,
            persistent_output_buffer,
            dim,
            multi_device_global_semaphore,
            num_links,
            memory_config_ag.value_or(input_tensor.memory_config()),
            topology,
            sub_device_id,
            /*cluster_axis=*/std::nullopt,
            /*use_optimal_ccl_for_llama=*/false,
            /*use_all_gather_async_llama_sharded=*/false,
            /*use_all_gather_async_via_broadcast*/ false,
            barrier_semaphore,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel,
            /*reverse_order=*/false,
            /*sub_core_grid=*/std::nullopt,
            /*optional_mesh_device=*/nullptr);

    // Create the all gather output tensor used as input (activation) to the matmul
    ttnn::Tensor all_gather_out_tensor = ttnn::experimental::prim::AllGatherAsyncDeviceOperation::create_output_tensors(
        all_gather_async_operation_attributes, all_gather_async_tensor_args);

    /* Matmul setup */
    bool user_run_batched = ttnn::operations::matmul::detail::is_input_batched(weight_tensor.logical_shape());
    std::optional<tt::tt_metal::CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = tt::tt_metal::CoreCoord(core_grid->x, core_grid->y);
    }

    auto matmul_struct = ttnn::prim::create_matmul_attributes(
        all_gather_out_tensor,
        weight_tensor,
        /*parameters=*/
        ttnn::prim::MatmulParams{
            program_config,
            /*bcast_batch=*/std::nullopt,
            memory_config_mm.value_or(input_tensor.memory_config()),
            dtype.value_or(input_tensor.dtype()),
            compute_kernel_config,
            /*untilize_out=*/false,
            user_core_coord,
            ttnn::operations::matmul::get_fused_activation(activation),
            user_run_batched,
            transpose_a,
            transpose_b,
            /*output_tile=*/std::nullopt,
            /*global_cb=*/std::nullopt},
        {});

    auto operation_attributes = OperationType::operation_attributes_t{
        /* All Gather Params */
        all_gather_async_operation_attributes,
        all_gather_async_tensor_args,
        /* Matmul Params */
        matmul_struct,
        /* Fusion params */
        all_gather_core_grid_offset,
    };

    auto tensor_args = OperationType::tensor_args_t{
        input_tensor,
        weight_tensor,
        bias,
        persistent_output_buffer,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
