// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async_device_operation.hpp"
#include "all_gather_async_device_operation_types.hpp"

#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

AllGatherAsyncVersion select_version(const AllGatherAsyncParams& operation_attributes) {
    // Check for minimal sharded case
    if (operation_attributes.use_all_gather_async_llama_sharded) {
        TT_FATAL(
            !operation_attributes.reverse_order,
            "Reversed all-gather (reverse_order=true) is not yet supported with llama-optimized variants "
            "(use_all_gather_async_llama_sharded=true). Please use the regular all_gather_async API instead of "
            "all_gather_async_reversed.");
        return AllGatherAsyncVersion::LLAMA_MINIMAL_SHARDED;
    }
    TT_FATAL(operation_attributes.semaphore.size() == 2, "Default implementation requires 2 semaphores");
    return AllGatherAsyncVersion::MINIMAL_DEFAULT;
}

AllGatherAsyncDeviceOperation::program_factory_t AllGatherAsyncDeviceOperation::select_program_factory(
    const AllGatherAsyncParams& args, const AllGatherAsyncInputs& /*tensor_args*/) {
    AllGatherAsyncVersion version = select_version(args);
    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));
    switch (version) {
        case AllGatherAsyncVersion::LLAMA_MINIMAL_SHARDED: {
            return LlamaShardedMeshWorkloadFactory{};
        }
        case AllGatherAsyncVersion::MINIMAL_DEFAULT:
        default: {
            return DefaultMeshWorkloadFactory{};
        }
    }
}

void AllGatherAsyncDeviceOperation::validate_on_program_cache_hit(
    const AllGatherAsyncParams& args, const AllGatherAsyncInputs& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void AllGatherAsyncDeviceOperation::validate_on_program_cache_miss(
    const AllGatherAsyncParams& args, const AllGatherAsyncInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& layout = input_tensor.layout();
    const auto& dtype = input_tensor.dtype();
    const auto& page_size = input_tensor.buffer()->page_size();
    TT_FATAL(
        (tt::tt_metal::hal::get_arch_name() != "blackhole") ||
            (input_tensor.memory_config().buffer_type() != BufferType::DRAM) ||
            !args.use_all_gather_async_llama_sharded,
        "This kernel does not support blackhole dram as it does not use an accessor to get the noc address as needed "
        "by the fabric api");
    TT_FATAL(page_size % input_tensor.buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(args.num_links > 0, "Error, num_links should be more than 0 but has {}", args.num_links);
    TT_FATAL(
        args.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported input tensor memory layout {}.",
        input_tensor.memory_config().memory_layout());

    AllGatherAsyncVersion version = select_version(args);

    if (tensor_args.persistent_output_buffer.has_value()) {
        const auto& output_tensor = tensor_args.persistent_output_buffer.value();

        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
        TT_FATAL(
            output_tensor.layout() == layout,
            "Error, Output tensor layout should be same as input tensor layout but has {}",
            output_tensor.layout());
        TT_FATAL(
            output_tensor.dtype() == dtype,
            "Error, Output tensor dtype should be same as input tensor dtype but has {}",
            output_tensor.dtype());
        TT_FATAL(
            output_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Error, Output tensor page config should be same as input tensor page config but has {}",
            output_tensor.tensor_spec().page_config());
        TT_FATAL(
            output_tensor.memory_config() == args.output_mem_config,
            "Error, Output tensor memory config should be same as output_mem_config but has {}",
            output_tensor.memory_config());

        TT_FATAL(
            output_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Unsupported output tensor memory layout {}.",
            output_tensor.memory_config().memory_layout());

        // check the output tensor size
        auto output_shape = output_tensor.padded_shape();
        auto input_shape = input_tensor.padded_shape();
        TT_FATAL(
            output_shape.size() == input_shape.size(),
            "Error, Output tensor shape should have same number of dimensions as input tensor but has {}",
            output_shape.size());
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i == args.dim) {
                TT_FATAL(
                    output_shape[i] <= input_shape[i] * args.ring_size,
                    "Error, Output tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i] * args.ring_size,
                    output_shape[i]);
            } else {
                TT_FATAL(
                    output_shape[i] == input_shape[i],
                    "Error, Output tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i],
                    output_shape[i]);
            }
        }

        if (version == AllGatherAsyncVersion::MINIMAL_DEFAULT) {
            // Checks specific to the MINIMAL_DEFAULT case

            // Don't support output DRAM block sharding
            if (output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
                TT_FATAL(
                    output_tensor.memory_config().buffer_type() == BufferType::L1,
                    "We don't support output DRAM block sharding");
            }
        } else {
            // Checks specific to cases that are not MINIMAL_DEFAULT

            TT_FATAL(
                output_tensor.memory_config().memory_layout() == input_tensor.memory_config().memory_layout(),
                "Error, Output tensor memory layout should be same as input tensor memory layout but has {}",
                output_tensor.memory_config().memory_layout());
        }
    }

    // Checks specific to the MINIMAL_DEFAULT case
    if (version == AllGatherAsyncVersion::MINIMAL_DEFAULT) {
        // Don't support input DRAM block sharding
        if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                input_tensor.memory_config().buffer_type() == BufferType::L1,
                "We don't support input DRAM block sharding");
        }
        TT_FATAL(input_tensor.logical_shape().rank() >= 2, "AllGatherAsync requires tensor of rank 2 or greater");
    } else {
        TT_FATAL(input_tensor.logical_shape().rank() == 4, "Llama specific all_gather requires tensor of rank 4");
    }
}

AllGatherAsyncDeviceOperation::spec_return_value_t AllGatherAsyncDeviceOperation::compute_output_specs(
    const AllGatherAsyncParams& args, const AllGatherAsyncInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto shape = input_tensor.logical_shape();
    shape[args.dim] *= args.ring_size;
    return TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), args.output_mem_config));
}

AllGatherAsyncDeviceOperation::tensor_return_value_t AllGatherAsyncDeviceOperation::create_output_tensors(
    const AllGatherAsyncParams& args, const AllGatherAsyncInputs& tensor_args) {
    if (tensor_args.persistent_output_buffer.has_value() && args.using_persistent_buffers) {
        return tensor_args.persistent_output_buffer.value();
    }
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t AllGatherAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllGatherAsyncDeviceOperation::compute_program_hash is called");

    auto subdevice_id = args.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    if (args.sub_core_grid.has_value()) {
        subdevice_core_range_set = subdevice_core_range_set.intersection(args.sub_core_grid.value());
    }

    auto program_factory = select_program_factory(args, tensor_args);

    return tt::tt_metal::operation::hash_operation<AllGatherAsyncDeviceOperation>(
        args.dim,
        args.num_links,
        args.ring_size,
        args.output_mem_config,
        args.topology,
        args.cluster_axis,
        args.barrier_semaphore.has_value(),
        args.using_persistent_buffers,
        args.chunks_per_sync,
        args.num_workers_per_link,
        args.num_buffers_per_channel,
        args.use_all_gather_async_llama_sharded,
        args.use_optimal_ccl_for_llama,
        args.reverse_order,
        subdevice_core_range_set,
        tensor_args,
        program_factory.index());
}

std::tuple<AllGatherAsyncParams, AllGatherAsyncInputs> AllGatherAsyncDeviceOperation::invoke(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<uint32_t>& cluster_axis,
    bool use_optimal_ccl_for_llama,
    bool use_all_gather_async_llama_sharded,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<uint32_t>& chunks_per_sync,
    const std::optional<uint32_t>& num_workers_per_link,
    const std::optional<uint32_t>& num_buffers_per_channel,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid,
    const MeshDevice* optional_mesh_device) {
    // Combine 3 implementations of the old all_gather_async_op.cpp::all_gather_async_impl
    // 1. only input_tensor, no output or optional mesh device
    // 2. has input tensor and output tensor but not optional mesh device
    // 3. has all three

    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    bool using_persistent_buffers = persistent_output_buffer.has_value();

    int32_t rank = input_tensor.logical_shape().rank();

    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    // Prioritize optional mesh device first, then check device of input_tensor
    bool using_optional_mesh_device = optional_mesh_device != nullptr;
    if (using_optional_mesh_device) {
        const auto& mesh_view = optional_mesh_device->get_view();
        TT_FATAL(
            mesh_view.is_mesh_2d(),
            "all-gather invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    } else {
        TT_FATAL(input_tensor.device() != nullptr, "Input tensor has no mesh device assigned.");

        TT_FATAL(num_devices > 1, "all_gather_async op will only work for num_devices > 1, but has {}", num_devices);

        if (using_persistent_buffers) {
            log_debug(tt::LogOp, "creating line_fabric with num devices: {}, num links: {}", num_devices, num_links);
            log_debug(tt::LogOp, "line_fabric is created");
        }
    }

    return {
        AllGatherAsyncParams(
            gather_dim,
            num_links,
            num_devices,
            memory_config.value_or(input_tensor.memory_config()),
            topology,
            multi_device_global_semaphore,
            sub_device_id,
            cluster_axis,
            use_all_gather_async_llama_sharded,
            use_optimal_ccl_for_llama,
            barrier_semaphore,
            using_persistent_buffers,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel,
            reverse_order,
            sub_core_grid),
        AllGatherAsyncInputs{.input_tensor = input_tensor, .persistent_output_buffer = persistent_output_buffer}};
}

}  // namespace ttnn::experimental::prim
