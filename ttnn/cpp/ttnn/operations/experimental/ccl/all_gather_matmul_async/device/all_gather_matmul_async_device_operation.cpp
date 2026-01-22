// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

/* All Gather Matmul fusion includes */
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation.hpp"

#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

AllGatherMatmulAsyncDeviceOperation::program_factory_t AllGatherMatmulAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return AllGatherMatmulAsyncMeshWorkloadFactory{};
}

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

tt::stl::hash::hash_t AllGatherMatmulAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllGatherMatmulAsyncDeviceOperation::compute_program_hash is called");

    auto subdevice_id = operation_attributes.all_gather_async_attributes.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

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
        tensor_args,
        program_factory.index());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::AllGatherMatmulAsyncDeviceOperation::tensor_return_value_t all_gather_matmul_async(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const CoreCoord all_gather_core_grid_offset,
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
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);

    /* All Gather setup */
    const auto [all_gather_async_operation_attributes, all_gather_async_tensor_args] =
        ttnn::experimental::prim::AllGatherAsyncDeviceOperation::invoke(
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
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
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
