// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_all_gather_matmul_async_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {
/* LlamaAllGatherMatmulAsync Implementation starts here*/
void LlamaAllGatherMatmulAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(input_tensors.size() == 3, "Error, Input tensor size should be 3 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(
        page_size % input_tensors[0].buffer()->alignment() == 0,
        "All Gather Replicate currently requires aligned pages");

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Operands to llama_all_gather_matmul need to be on device!");
    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to llama_all_gather_matmul need to be allocated in buffers on device!");
    TT_FATAL(
        this->all_gather_params.num_links > 0,
        "Error, num_links should be more than 0 but has {}",
        this->all_gather_params.num_links);
    TT_FATAL(
        this->all_gather_params.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout());
}

std::vector<ttnn::TensorSpec> LlamaAllGatherMatmulAsync::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor0 = input_tensors[0];

    auto intermediate_shape = input_tensor0.padded_shape();
    intermediate_shape[-1] = intermediate_shape[-1] * this->all_gather_params.ring_size;
    auto intermediate_shard_shape = this->all_gather_params.output_mem_config.shard_spec()->shape;
    TensorSpec intermediate_tensor_spec = TensorSpec(
        intermediate_shape,
        TensorLayout(
            input_tensor0.dtype(),
            input_tensor0.tensor_spec().page_config(),
            this->all_gather_params.output_mem_config));

    // Calculate aggregated tensor shape and shard specs
    auto aggregated_shape = intermediate_shape;
    aggregated_shape[-1] = intermediate_shape[-1] * 60;

    auto aggregated_shard_shape = intermediate_shard_shape;
    aggregated_shard_shape[1] = intermediate_shard_shape[1] * this->all_gather_params.ring_size;

    // Create aggregated tensor memory config
    MemoryConfig aggregated_mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(
            ttnn::CoreRangeSet({ttnn::CoreRange(ttnn::CoreCoord(1, 0), ttnn::CoreCoord(6, 9))}),  // MCAST_CRS
            aggregated_shard_shape,
            tt::tt_metal::ShardOrientation::ROW_MAJOR));

    TensorSpec aggregated_tensor_spec = TensorSpec(
        aggregated_shape,
        TensorLayout(input_tensor0.dtype(), input_tensor0.tensor_spec().page_config(), aggregated_mem_config));

    // Matmul output spec - using aggregated tensor as input to matmul
    ttnn::TensorSpec matmul_output_specs =
        this->matmul_struct.compute_output_specs({input_tensors[0], input_tensors[1]}, {})[0];

    return {matmul_output_specs, aggregated_tensor_spec};
}

std::vector<Tensor> LlamaAllGatherMatmulAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    // Create aggregated tensor internally with exact same specs as pytest
    // const auto& intermediate_tensor = input_tensors[2];

    auto specs = compute_output_specs(input_tensors);
    const auto& aggregated_tensor_spec = specs[1];

    ttnn::Tensor aggregated_tensor = create_device_tensor(aggregated_tensor_spec, input_tensors[0].device());

    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor =
        this->matmul_struct.create_output_tensors({aggregated_tensor, input_tensors[1]})[0];

    return {matmul_output_tensor, aggregated_tensor};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks LlamaAllGatherMatmulAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, optional_input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks LlamaAllGatherMatmulAsync::create_program_at(
    const ttnn::MeshCoordinate& mesh_coordinate,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto mesh_device = input_tensors[0].device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coordinate) : input_tensors[0].device();
    std::vector<IDevice*> devices_to_use = {};
    if (this->all_gather_params.cluster_axis.has_value()) {
        // User specified the cluster-axis. Derive devices based on the current coordinate
        // and the cluster-axis.
        const auto& mesh_view = input_tensors[0].device()->get_view();
        devices_to_use = (this->all_gather_params.cluster_axis.value() == 0)
                             ? mesh_view.get_devices_on_column(mesh_coordinate[1])
                             : mesh_view.get_devices_on_row(mesh_coordinate[0]);
    } else {
        devices_to_use = this->all_gather_params.devices;
    }

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < this->all_gather_params.ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (this->all_gather_params.topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(this->all_gather_params.ring_size - 1);
            }
            if (i != this->all_gather_params.ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (this->all_gather_params.topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    return llama_all_gather_matmul_async_sharded(
        input_tensors[0],   // in0
        input_tensors[1],   // in1
        output_tensors[0],  // mm output tensor
        input_tensors[2],   // intermediate_tensor
        output_tensors[1],  // aggregated_tensor (now output)
        target_device,
        forward_device,
        backward_device,
        this->all_gather_params.dim,
        this->all_gather_params.num_links,
        this->all_gather_params.ring_size,
        device_index,
        this->all_gather_params.topology,
        this->all_gather_params.semaphore,
        this->all_gather_params.sub_device_id,
        // MM params
        this->matmul_struct.compute_kernel_config.value(),
        this->matmul_struct.program_config.value(),
        this->matmul_struct.global_cb);
}

tt::tt_metal::operation::Hash LlamaAllGatherMatmulAsync::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors) const {
    // Input tensor
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();

    auto input2_shape = input_tensors[1].padded_shape();
    auto input2_memory_layout = input_tensors[1].layout();
    auto input2_dtype = input_tensors[1].dtype();
    auto input2_memory_config = input_tensors[1].memory_config();

    // Intermediate tensor
    auto intermediate_shape = input_tensors[1].padded_shape();
    auto intermediate_memory_layout = input_tensors[1].layout();
    auto intermediate_dtype = input_tensors[1].dtype();
    auto intermediate_memory_config = input_tensors[1].memory_config();

    return tt::tt_metal::operation::hash_operation<LlamaAllGatherMatmulAsync>(
        this->all_gather_params.dim,
        this->all_gather_params.num_links,
        this->all_gather_params.ring_size,
        this->all_gather_params.output_mem_config,
        this->all_gather_params.topology,
        this->all_gather_params.cluster_axis,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config,
        input2_shape,
        input2_memory_layout,
        input2_dtype,
        input2_memory_config,
        intermediate_shape,
        intermediate_memory_layout,
        intermediate_dtype,
        intermediate_memory_config);
}
/* LlamaAllGatherMatmulAsync Implementation ends here*/

namespace operations {
namespace experimental {
namespace ccl {

namespace {

Tensor llama_all_gather_matmul_async_impl(
    const Tensor& input_tensor,
    const Tensor& input1,
    const Tensor& intermediate_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& ag_memory_config,
    const std::optional<MemoryConfig>& mm_memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const DataType> dtype,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb) {
    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "all-gather-replicate invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    int32_t rank = input_tensor.logical_shape().rank();

    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<std::optional<const Tensor>> optional_input_tensors = {};
    optional_input_tensors.push_back(std::nullopt);
    std::vector<std::optional<Tensor>> optional_output_tensors = {};
    optional_output_tensors.push_back(std::nullopt);

    ttnn::AllGatherParams all_gather_params = ttnn::AllGatherParams{
        {},
        gather_dim,
        num_preferred_links.has_value() ? num_preferred_links.value() : 1,
        num_devices,
        ag_memory_config.value_or(input_tensor.memory_config()),
        topology,
        multi_device_global_semaphore,
        sub_device_id,
        cluster_axis};

    operations::matmul::Matmul matmul_struct = operations::matmul::create_matmul_struct(
        input_tensor,
        input1,
        /*parameters=*/
        operations::matmul::Matmul{
            program_config,
            /*bcast_batch=*/std::nullopt,
            mm_memory_config.value_or(input_tensor.memory_config()),
            dtype.value_or(input_tensor.dtype()),
            compute_kernel_config,
            /*untilize_out=*/false,
            /*user_core_coord=*/std::nullopt,
            /*activation=*/std::nullopt,
            /*user_run_batched=*/false,
            /*transpose_a=*/false,
            /*transpose_b=*/false,
            /*output_tile=*/std::nullopt,
            /*global_cb=*/global_cb});

    ttnn::LlamaAllGatherMatmulAsync llama_all_gather_matmul_async_struct =
        ttnn::LlamaAllGatherMatmulAsync{all_gather_params, matmul_struct, devices};
    // return input_tensor;  // TODO: Implement the actual logic
    // return tt::tt_metal::operation::run(all_gather_struct, {input_tensor, intermediate_tensor, aggregated_tensor})
    //     .at(0);
    auto tensors_out = tt::tt_metal::operation::run(
        llama_all_gather_matmul_async_struct,
        {input_tensor, input1, intermediate_tensor},
        optional_input_tensors,
        optional_output_tensors);
    tensors_out.at(1).deallocate(true);
    return tensors_out.at(0);
}
}  // namespace

Tensor llama_all_gather_matmul_async(
    const Tensor& input_tensor0,
    const Tensor& input_tensor1,
    const Tensor& intermediate_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& ag_memory_config,
    const std::optional<MemoryConfig>& mm_memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const DataType> dtype,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb) {
    return llama_all_gather_matmul_async_impl(
        input_tensor0,
        input_tensor1,
        intermediate_tensor,
        dim,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        ag_memory_config,
        mm_memory_config,
        num_preferred_links,
        sub_device_id,
        program_config,
        compute_kernel_config,
        dtype,
        global_cb);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
