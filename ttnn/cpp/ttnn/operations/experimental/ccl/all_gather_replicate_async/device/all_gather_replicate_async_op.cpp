// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_replicate_async_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {
/*AllGatherReplicateAsync Implementation starts here*/
void AllGatherReplicateAsync::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 3, "Error, Input tensor size should be 3 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].layout();
    const auto& dtype = input_tensors[0].dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(
        page_size % input_tensors[0].buffer()->alignment() == 0,
        "All Gather Replicate currently requires aligned pages");

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather_replicate need to be on device!");
    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to all_gather_replicate need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout());

    if (input_tensors.size() > 1) {
        const auto& intermediate_tensor = input_tensors[1];

        TT_FATAL(
            intermediate_tensor.storage_type() == StorageType::DEVICE,
            "Operands to all_gather_replicate need to be on device!");
        TT_FATAL(
            intermediate_tensor.layout() == layout,
            "Error, intermediate tensor layout should be same as input tensor layout but has {}",
            intermediate_tensor.layout());
        TT_FATAL(
            intermediate_tensor.dtype() == dtype,
            "Error, intermediate tensor dtype should be same as input tensor dtype but has {}",
            intermediate_tensor.dtype());
        TT_FATAL(
            intermediate_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Error, intermediate tensor page config should be same as input tensor page config but has {}",
            intermediate_tensor.tensor_spec().page_config());

        // check the intermediate tensor size
        auto intermediate_shape = intermediate_tensor.padded_shape();
        auto input_shape = input_tensor.padded_shape();
        TT_FATAL(
            intermediate_shape.size() == input_shape.size(),
            "Error, intermediate tensor shape should have same number of dimensions as input tensor but has {}",
            intermediate_shape.size());
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i == this->dim) {
                TT_FATAL(
                    intermediate_shape[i] <= input_shape[i] * this->ring_size,
                    "Error, intermediate tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i] * this->ring_size,
                    intermediate_shape[i]);
            } else {
                TT_FATAL(
                    intermediate_shape[i] == input_shape[i],
                    "Error, intermediate tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i],
                    intermediate_shape[i]);
            }
        }

        // check memory layout
        TT_FATAL(
            intermediate_tensor.memory_config().memory_layout() == input_tensor.memory_config().memory_layout(),
            "Error, intermediate tensor memory layout should be same as input tensor memory layout but has {}",
            intermediate_tensor.memory_config().memory_layout());
    }

    // TODO: Add validation for output_mem_config
}

std::vector<ttnn::TensorSpec> AllGatherReplicateAsync::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& intermediate_tensor = input_tensors[1];
    auto shape = intermediate_tensor.padded_shape();  // TODO: Replace with logical_shape()

    // Replicate output on all the cores in the shard spec
    shape[1] = this->output_mem_config.shard_spec()->grid.num_cores();

    return {TensorSpec(
        shape,
        TensorLayout(intermediate_tensor.dtype(), intermediate_tensor.tensor_spec().page_config(), output_mem_config))};
}

std::vector<Tensor> AllGatherReplicateAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    return tt::tt_metal::operation::default_create_output_tensors(*this, input_tensors, optional_output_tensors);
}

AllGatherReplicateAsyncVersion AllGatherReplicateAsync::select_version(const Tensor& input_tensor) const {
    auto input_tensor_shape = input_tensor.padded_shape();
    auto input_tensor_buffer_layout = input_tensor.buffer()->buffer_layout();
    auto input_tensor_page_layout = input_tensor.layout();
    auto input_tensor_memory_config = input_tensor.memory_config();
    bool input_is_sharded = input_tensor_memory_config.shard_spec().has_value();
    bool output_is_sharded = output_mem_config.shard_spec().has_value();
    uint32_t input_shard_num_cores = 0;
    uint32_t output_shard_num_cores = 0;
    if (input_is_sharded) {
        input_shard_num_cores = input_tensor_memory_config.shard_spec()->grid.num_cores();
        log_trace(
            tt::LogOp,
            "[select_version] input_tensor_memory_config.shard_spec()->shape: {}",
            input_tensor_memory_config.shard_spec()->shape);
    }
    if (output_is_sharded) {
        output_shard_num_cores = output_mem_config.shard_spec()->grid.num_cores();
        log_trace(
            tt::LogOp,
            "[select_version] output_mem_config.shard_spec()->shape: {}",
            output_mem_config.shard_spec()->shape);
    }

    log_trace(tt::LogOp, "[select_version] input_tensor_shape: {}", input_tensor_shape);
    log_trace(tt::LogOp, "[select_version] input_tensor_memory_config: {}", input_tensor_memory_config);
    log_trace(tt::LogOp, "[select_version] output_mem_config: {}", output_mem_config);
    log_trace(tt::LogOp, "[select_version] input_shard_num_cores: {}", input_shard_num_cores);
    log_trace(tt::LogOp, "[select_version] output_shard_num_cores: {}", output_shard_num_cores);

    log_trace(tt::LogOp, "[select_version] input_is_sharded: {}", input_is_sharded);
    log_trace(tt::LogOp, "[select_version] output_is_sharded: {}", output_is_sharded);

    if (input_is_sharded && output_is_sharded) {
        // Check for llama post binary mult+silu case
        if (input_tensor_shape[0] == 1 && input_tensor_shape[1] == 1 && input_tensor_shape[2] == 32 &&
            input_tensor_shape[3] == 960 && input_tensor_memory_config.buffer_type() == BufferType::L1 &&
            output_mem_config.buffer_type() == BufferType::L1 &&
            input_tensor_memory_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
            input_tensor_memory_config.shard_spec()->shape[0] == 32 &&
            input_tensor_memory_config.shard_spec()->shape[1] == 32 && output_mem_config.shard_spec()->shape[0] == 32) {
            log_trace(
                tt::LogOp,
                "Matching conditions for Llama post binary mult+silu, using LLAMA_MINIMAL_SHARDED implementation");
            return AllGatherReplicateAsyncVersion::LLAMA_MINIMAL_SHARDED;
        }

        // Check for llama post SDPA case
        if (input_tensor_shape[0] == 1 && input_tensor_shape[1] == 8 && input_tensor_shape[2] == 32 &&
            input_tensor_shape[3] == 128 && input_tensor_memory_config.buffer_type() == BufferType::L1 &&
            output_mem_config.buffer_type() == BufferType::L1 &&
            input_tensor_memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
            output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
            input_tensor_memory_config.shard_spec()->shape[0] == 32 &&
            input_tensor_memory_config.shard_spec()->shape[1] == 128 &&
            output_mem_config.shard_spec()->shape[0] == 32 && output_mem_config.shard_spec()->shape[1] == 128 &&
            input_shard_num_cores == 8 && output_shard_num_cores == 32) {
            log_trace(tt::LogOp, "Matching conditions for Llama post SDPA, using LLAMA_MINIMAL_SHARDED implementation");
            return AllGatherReplicateAsyncVersion::LLAMA_MINIMAL_SHARDED;
        }

        // Check for llama rms norm case
        if (input_tensor_shape[0] == 1 && input_tensor_shape[1] == 1 && input_tensor_shape[2] == 32 &&
            input_tensor_shape[3] == 32 && input_tensor_memory_config.buffer_type() == BufferType::L1 &&
            output_mem_config.buffer_type() == BufferType::L1 &&
            input_tensor_memory_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            input_tensor_memory_config.shard_spec()->shape[0] == 32 &&
            input_tensor_memory_config.shard_spec()->shape[1] == 32 && output_mem_config.shard_spec()->shape[0] == 32 &&
            output_mem_config.shard_spec()->shape[1] == 128 && input_shard_num_cores == 1 &&
            output_shard_num_cores == 1) {
            log_trace(
                tt::LogOp, "Matching conditions for Llama rms norm case, using LLAMA_MINIMAL_SHARDED implementation");
            return AllGatherReplicateAsyncVersion::LLAMA_MINIMAL_SHARDED;
        }
    }
    log_trace(tt::LogOp, "Using generic implementation");
    return AllGatherReplicateAsyncVersion::GENERIC;
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks AllGatherReplicateAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks AllGatherReplicateAsync::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    log_debug(tt::LogOp, "DEBUG: create_program_at is called");
    auto mesh_device = input_tensors[0].mesh_device();
    AllGatherReplicateAsyncVersion version = select_version(input_tensors[0]);
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();
    std::vector<IDevice*> devices_to_use = {};
    if (this->cluster_axis.has_value()) {
        // User specified the cluster-axis. Derive devices based on the current coordinate
        // and the cluster-axis.
        const auto& mesh_view = input_tensors[0].mesh_device()->get_view();
        devices_to_use = (this->cluster_axis.value() == 0) ? mesh_view.get_devices_on_column(coord[1])
                                                           : mesh_view.get_devices_on_row(coord[0]);
    } else {
        devices_to_use = devices;
    }

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < this->ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(this->ring_size - 1);
            }
            if (i != this->ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }
    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));

    switch (version) {
        case AllGatherReplicateAsyncVersion::LLAMA_MINIMAL_SHARDED:
        default:
            log_trace(
                tt::LogOp,
                "Detected all gather replicate specialized shape. all_gather_replicate_async_sharded is called");
            return all_gather_replicate_async_sharded(
                input_tensors[0],
                input_tensors[1],
                input_tensors[2],
                output_tensors[0],
                target_device,
                forward_device,
                backward_device,
                this->dim,
                this->num_links,
                this->ring_size,
                device_index,
                this->topology,
                this->semaphore,
                this->sub_device_id);
    }
}

tt::tt_metal::operation::Hash AllGatherReplicateAsync::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    AllGatherReplicateAsyncVersion version = select_version(input_tensors[0]);
    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));

    // Input tensor
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();

    // Intermediate tensor
    auto intermediate_shape = input_tensors[1].padded_shape();
    auto intermediate_memory_layout = input_tensors[1].layout();
    auto intermediate_dtype = input_tensors[1].dtype();
    auto intermediate_memory_config = input_tensors[1].memory_config();

    uint32_t semaphore_address = this->semaphore.address();
    return tt::tt_metal::operation::hash_operation<AllGatherReplicateAsync>(
        this->dim,
        this->num_links,
        this->ring_size,
        this->output_mem_config,
        this->topology,
        this->cluster_axis,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config,
        intermediate_shape,
        intermediate_memory_layout,
        intermediate_dtype,
        intermediate_memory_config);
}

/*AllGatherReplicateAsync Implementation ends here*/

/* LlamaAllGatherMatmulAsync Implementation starts here*/
void LlamaAllGatherMatmulAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(input_tensors.size() == 4, "Error, Input tensor size should be 4 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].layout();
    const auto& dtype = input_tensors[0].dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(
        page_size % input_tensors[0].buffer()->alignment() == 0,
        "All Gather Replicate currently requires aligned pages");

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather_replicate need to be on device!");
    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to all_gather_replicate need to be allocated in buffers on device!");
    TT_FATAL(
        this->all_gather_replicate_async_struct.num_links > 0,
        "Error, num_links should be more than 0 but has {}",
        this->all_gather_replicate_async_struct.num_links);
    TT_FATAL(
        this->all_gather_replicate_async_struct.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout());

    if (input_tensors.size() > 1) {
        const auto& intermediate_tensor = input_tensors[2];

        TT_FATAL(
            intermediate_tensor.storage_type() == StorageType::DEVICE,
            "Operands to all_gather_replicate need to be on device!");
        TT_FATAL(
            intermediate_tensor.layout() == layout,
            "Error, intermediate tensor layout should be same as input tensor layout but has {}",
            intermediate_tensor.layout());
        TT_FATAL(
            intermediate_tensor.dtype() == dtype,
            "Error, intermediate tensor dtype should be same as input tensor dtype but has {}",
            intermediate_tensor.dtype());
        TT_FATAL(
            intermediate_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Error, intermediate tensor page config should be same as input tensor page config but has {}",
            intermediate_tensor.tensor_spec().page_config());

        // check the intermediate tensor size
        auto intermediate_shape = intermediate_tensor.padded_shape();
        auto input_shape = input_tensor.padded_shape();
        TT_FATAL(
            intermediate_shape.size() == input_shape.size(),
            "Error, intermediate tensor shape should have same number of dimensions as input tensor but has {}",
            intermediate_shape.size());
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i == this->all_gather_replicate_async_struct.dim) {
                TT_FATAL(
                    intermediate_shape[i] <= input_shape[i] * this->all_gather_replicate_async_struct.ring_size,
                    "Error, intermediate tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i] * this->all_gather_replicate_async_struct.ring_size,
                    intermediate_shape[i]);
            } else {
                TT_FATAL(
                    intermediate_shape[i] == input_shape[i],
                    "Error, intermediate tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i],
                    intermediate_shape[i]);
            }
        }

        // check memory layout
        TT_FATAL(
            intermediate_tensor.memory_config().memory_layout() == input_tensor.memory_config().memory_layout(),
            "Error, intermediate tensor memory layout should be same as input tensor memory layout but has {}",
            intermediate_tensor.memory_config().memory_layout());
    }

    // TODO: Add validation for output_mem_config
}

std::vector<ttnn::TensorSpec> LlamaAllGatherMatmulAsync::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    // All Gather shape
    ttnn::TensorSpec all_gather_output_shape =
        this->all_gather_replicate_async_struct.compute_output_specs({input_tensors[0]})[0];

    // Matmul shape
    ttnn::TensorSpec matmul_output_specs =
        this->matmul_struct.compute_output_specs({input_tensors[0], input_tensors[1]}, {})[0];

    return {all_gather_output_shape, matmul_output_specs};
}

std::vector<Tensor> LlamaAllGatherMatmulAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    // All Gather output tensor
    auto& all_gather_output_tensor = optional_output_tensors.at(0).value();

    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor =
        this->matmul_struct.create_output_tensors({all_gather_output_tensor, input_tensors[1]})[0];

    return {all_gather_output_tensor, matmul_output_tensor};
}

/* LlamaAllGatherMatmulAsync Implementation ends here*/

namespace operations {
namespace experimental {
namespace ccl {

namespace {

LlamaAllGatherMatmulAsync create_llama_all_gather_matmul_async_struct(
    const ttnn::AllGatherReplicateAsync& all_gather_replicate_async_struct,
    const operations::matmul::Matmul& matmul_struct,
    const std::vector<IDevice*>& devices) {
    return LlamaAllGatherMatmulAsync{all_gather_replicate_async_struct, matmul_struct, devices};
}

Tensor llama_all_gather_matmul_async_impl(
    const Tensor& input_tensor,
    const Tensor& input_tensor_b,
    const Tensor& intermediate_tensor,
    const Tensor& aggregated_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const DataType> dtype) {
    const auto mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "all-gather-replicate invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    int32_t rank = input_tensor.logical_shape().rank();

    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);
    // return tt::tt_metal::operation::run(
    //            ttnn::AllGatherReplicateAsync{
    //                {},
    //                gather_dim,
    //                num_preferred_links.has_value() ? num_preferred_links.value() : 1,
    //                num_devices,
    //                memory_config.value_or(input_tensor.memory_config()),
    //                topology,
    //                multi_device_global_semaphore,
    //                sub_device_id,
    //                cluster_axis},
    //            {input_tensor, intermediate_tensor, aggregated_tensor})
    //     .at(0);

    ttnn::AllGatherReplicateAsync all_gather_struct = ttnn::AllGatherReplicateAsync{
        {},
        gather_dim,
        num_preferred_links.has_value() ? num_preferred_links.value() : 1,
        num_devices,
        memory_config.value_or(input_tensor.memory_config()),
        topology,
        multi_device_global_semaphore,
        sub_device_id,
        cluster_axis};

    operations::matmul::Matmul matmul_struct = operations::matmul::create_matmul_struct(
        input_tensor,
        input_tensor_b,
        /*parameters=*/
        operations::matmul::Matmul{
            program_config,
            /*bcast_batch=*/std::nullopt,
            memory_config.value_or(input_tensor.memory_config()),
            dtype.value_or(input_tensor.dtype()),
            compute_kernel_config,
            /*untilize_out=*/false,
            /*user_core_coord=*/std::nullopt,
            /*activation=*/std::nullopt,
            /*user_run_batched=*/false,
            /*transpose_a=*/false,
            /*transpose_b=*/false,
            /*output_tile=*/std::nullopt,
            /*global_cb=*/std::nullopt});

    LlamaAllGatherMatmulAsync llama_all_gather_matmul_async_struct{
        all_gather_struct, matmul_struct, mesh_device.get_devices()};
    // return input_tensor;  // TODO: Implement the actual logic
    return tt::tt_metal::operation::run(all_gather_struct, {input_tensor, intermediate_tensor, aggregated_tensor})
        .at(0);
}
}  // namespace

Tensor llama_all_gather_matmul_async(
    const Tensor& input_tensor,
    const Tensor& input_tensor_b,
    const Tensor& intermediate_tensor,
    const Tensor& aggregated_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const DataType> dtype) {
    return llama_all_gather_matmul_async_impl(
        input_tensor,
        input_tensor_b,
        intermediate_tensor,
        aggregated_tensor,
        dim,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        memory_config,
        num_preferred_links,
        sub_device_id,
        program_config,
        compute_kernel_config,
        dtype);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
