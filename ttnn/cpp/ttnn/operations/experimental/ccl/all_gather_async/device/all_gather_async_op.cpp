// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async_op.hpp"
#include "ttnn/operations/math.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {
namespace ccl {
namespace all_gather_detail {

Tensor all_gather_2D_helper(
    const Tensor& input_tensor,
    const int32_t dim,
    const int32_t rank,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    std::vector<GlobalSemaphore> semaphores,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    bool do_horizontal,
    bool is_first) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    // Calculate lower pages
    uint32_t lower_pages = 1;
    uint32_t higher_pages = 1;
    uint32_t page_size = 1;
    const auto& tensor_shape = input_tensor.get_padded_shape();
    if (input_tensor.get_layout() == Layout::TILE) {
        TT_FATAL(
            input_tensor.get_padded_shape() == input_tensor.get_logical_shape(),
            "padding currently not supported in all gather");
        page_size = input_tensor.get_tensor_spec().tile().get_tile_hw();
        if (dim == (rank - 1)) {
            lower_pages = tensor_shape[-1] / input_tensor.get_tensor_spec().tile().get_width();
            higher_pages = tensor_shape[-2] / input_tensor.get_tensor_spec().tile().get_height();
            for (int i = 0; i < rank - 2; i++) {
                higher_pages *= tensor_shape[i];
            }
        } else {
            lower_pages = tensor_shape[-1] / input_tensor.get_tensor_spec().tile().get_width() * tensor_shape[-2] /
                          input_tensor.get_tensor_spec().tile().get_height();
            for (int i = dim; i < rank - 2; i++) {
                lower_pages *= tensor_shape[i];
            }
            for (int i = 0; i < dim; i++) {
                higher_pages *= tensor_shape[i];
            }
        }
    } else {
        TT_FATAL(dim != (rank - 1), "Can't currently support last dim all gather in row major");
        page_size = tensor_shape[-1];
        for (int i = dim; i < rank - 1; i++) {
            lower_pages *= tensor_shape[i];
        }
        for (int i = 0; i < dim; i++) {
            higher_pages *= tensor_shape[i];
        }
    }
    const auto mesh_view = mesh_device.get_view();
    auto grid_size = mesh_device.shape();
    auto num_rows = mesh_device.num_rows();
    auto num_cols = mesh_device.num_cols();
    operation::launch_op(
        [dim,
         memory_config,
         mesh_view,
         grid_size,
         num_rows,
         num_cols,
         topology,
         semaphores,
         lower_pages,
         higher_pages,
         page_size,
         do_horizontal,
         is_first](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_device_tensor = input_tensors.at(0);
            const auto coordinate = mesh_view.find_device(input_device_tensor.device()->id());
            std::vector<IDevice*> devices = (do_horizontal) ? mesh_view.get_devices_on_row(coordinate[0])
                                                            : mesh_view.get_devices_on_column(coordinate[1]);
            std::size_t num_devices = (do_horizontal) ? mesh_view.num_cols() : mesh_view.num_rows();
            std::optional<GlobalSemaphore> semaphore = std::nullopt;
            for (uint32_t i = 0; i < num_devices; ++i) {
                if (devices.at(i) == input_device_tensor.device()) {
                    semaphore = semaphores.at(i);  // Get raw pointer
                }
            }
            return operation::run(
                AllGather2D{
                    coordinate,
                    grid_size,
                    num_rows,
                    num_cols,
                    num_devices,
                    memory_config.value_or(input_device_tensor.memory_config()),
                    topology,
                    semaphore.value(),
                    dim,
                    lower_pages,
                    higher_pages,
                    page_size,
                    do_horizontal,
                    is_first ? num_devices : num_devices * 2},
                {input_device_tensor});
        },
        {input_tensor},
        output_tensors);

    return output_tensors.at(0);
}

Tensor all_gather_async_2D(
    const Tensor& input_tensor,
    const int32_t dim,
    const int32_t rank,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    std::vector<GlobalSemaphore> semaphores,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    bool transpose_mesh_dimension) {
    /*
    //swap to this when we have barrier on semaphores
    // Call the operation on the rows/cols
    auto output_tensor = all_gather_2D_helper(
        input_tensor,
        dim,
        rank,
        mesh_device,
        topology,
        semaphores,
        memory_config,
        num_preferred_links,
        !transpose_mesh_dimension,
        true);
    // Call the operation on the cols/rows
    return all_gather_2D_helper(
        output_tensor,
        dim,
        rank,
        mesh_device,
        topology,
        semaphores,
        memory_config,
        num_preferred_links,
        transpose_mesh_dimension
        false);
    */
    return all_gather_2D_helper(
        input_tensor,
        dim,
        rank,
        mesh_device,
        topology,
        semaphores,
        memory_config,
        num_preferred_links,
        !transpose_mesh_dimension,
        true);
}

AllGatherAsync create_all_gather_async_struct(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<SubDeviceId> sub_device_id,
    bool enable_persistent_fabric_mode) {
    uint32_t num_devices = devices.size();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    std::optional<GlobalSemaphore> semaphore = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < num_devices; ++i) {
        if (devices.at(i) == input_tensor.device()) {
            device_index = i;
            semaphore = semaphores.at(i);  // Get raw pointer
            if (i != 0) {
                backward_device = devices.at(i - 1);
            }
            if (i != num_devices - 1) {
                forward_device = devices.at(i + 1);
            }
        }
    }

    return ttnn::AllGatherAsync{
        forward_device,
        backward_device,
        dim,
        num_links,
        num_devices,
        device_index,
        memory_config.value_or(input_tensor.memory_config()),
        topology,
        semaphore.value(),
        sub_device_id,
        enable_persistent_fabric_mode};
}

}  // namespace all_gather_detail
}  // namespace ccl

void AllGather2D::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_gather need to be allocated in buffers on device!");
    auto num_devices = this->num_devices;
    TT_FATAL(num_devices > 0, "Error, num_links should be more than 0 but has {}", num_devices);
    TT_FATAL(
        num_devices <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout);
}

void AllGatherAsync::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout);
}

std::vector<ttnn::TensorSpec> AllGather2D::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.get_padded_shape();  // TODO: Replace with get_logical_shape()
    shape[this->dim] *= this->num_devices;
    return {TensorSpec(
        shape,
        TensorLayout(input_tensor.get_dtype(), input_tensor.get_tensor_spec().page_config(), output_mem_config))};
}

std::vector<ttnn::TensorSpec> AllGatherAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.get_padded_shape();  // TODO: Replace with get_logical_shape()
    shape[this->dim] *= this->ring_size;
    return {TensorSpec(
        shape,
        TensorLayout(input_tensor.get_dtype(), input_tensor.get_tensor_spec().page_config(), output_mem_config))};
}

AllGather2DVersion AllGather2D::select_version(const Tensor& input_tensor) const { return AllGather2DVersion::GENERIC; }

AllGatherAsyncVersion AllGatherAsync::select_version(const Tensor& input_tensor) const {
    auto input_tensor_shape = input_tensor.get_padded_shape();
    auto input_tensor_buffer_layout = input_tensor.buffer()->buffer_layout();
    auto input_tensor_page_layout = input_tensor.layout();
    auto input_tensor_memory_config = input_tensor.memory_config();
    bool input_is_sharded = input_tensor_memory_config.shard_spec.has_value();
    bool output_is_sharded = output_mem_config.shard_spec.has_value();
    uint32_t input_shard_num_cores = 0;
    uint32_t output_shard_num_cores = 0;
    if (input_is_sharded) {
        input_shard_num_cores = input_tensor_memory_config.shard_spec->grid.num_cores();
        log_trace(
            tt::LogOp,
            "[select_version] input_tensor_memory_config.shard_spec->shape: {}",
            input_tensor_memory_config.shard_spec->shape);
    }
    if (output_is_sharded) {
        output_shard_num_cores = output_mem_config.shard_spec->grid.num_cores();
        log_trace(tt::LogOp, "[select_version] output_mem_config.shard_spec->shape: {}", output_mem_config.shard_spec->shape);
    }

    log_trace(tt::LogOp, "[select_version] input_tensor_shape: {}", input_tensor_shape);
    log_trace(tt::LogOp, "[select_version] input_tensor_memory_config: {}", input_tensor_memory_config);
    log_trace(tt::LogOp, "[select_version] output_mem_config: {}", output_mem_config);
    log_trace(tt::LogOp, "[select_version] input_shard_num_cores: {}", input_shard_num_cores);
    log_trace(tt::LogOp, "[select_version] output_shard_num_cores: {}", output_shard_num_cores);

    // Check for minimal interleaved case
    if (input_tensor_shape[0] == 1 && input_tensor_shape[1] == 1 && input_tensor_shape[2] == 32 &&
        input_tensor_buffer_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
        input_tensor_page_layout == tt::tt_metal::Layout::TILE && this->enable_persistent_fabric_mode) {
        return AllGatherAsyncVersion::MINIMAL_INTERLEAVED_32;
    }

    log_trace(tt::LogOp, "[select_version] input_is_sharded: {}", input_is_sharded);
    log_trace(tt::LogOp, "[select_version] output_is_sharded: {}", output_is_sharded);

    if (input_is_sharded && output_is_sharded) {
        // Check for first llama post binary matmul case
        if (input_tensor_shape[0] == 1 && input_tensor_shape[1] == 1 && input_tensor_shape[2] == 32 &&
            input_tensor_shape[3] == 960 && input_tensor_memory_config.buffer_type == BufferType::L1 &&
            output_mem_config.buffer_type == BufferType::L1 &&
            input_tensor_memory_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED &&
            output_mem_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED &&
            input_tensor_memory_config.shard_spec->shape[0] == 32 &&
            input_tensor_memory_config.shard_spec->shape[1] == 32 &&
            output_mem_config.shard_spec->shape[0] == 32 &&
            output_mem_config.shard_spec->shape[1] == 160 && input_shard_num_cores == 30 &&
            output_shard_num_cores == 24) {
            return AllGatherAsyncVersion::LLAMA_POST_BINARY_MATMUL;
        }

        // Check for second llama post binary matmul case
        if (input_tensor_shape[0] == 1 && input_tensor_shape[1] == 8 && input_tensor_shape[2] == 32 &&
            input_tensor_shape[3] == 128 && input_tensor_memory_config.buffer_type == BufferType::L1 &&
            output_mem_config.buffer_type == BufferType::L1 &&
            input_tensor_memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED &&
            output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED &&
            input_tensor_memory_config.shard_spec->shape[0] == 32 &&
            input_tensor_memory_config.shard_spec->shape[1] == 128 &&
            output_mem_config.shard_spec->shape[0] == 32 &&
            output_mem_config.shard_spec->shape[1] == 128 && input_shard_num_cores == 8 &&
            output_shard_num_cores == 32) {
            log_trace(tt::LogOp, "All conditions matched for LLAMA_POST_BINARY_MATMUL case");
            return AllGatherAsyncVersion::LLAMA_POST_BINARY_MATMUL;
        }
    }
    log_trace(tt::LogOp, "All conditions matched for generic case");
    return AllGatherAsyncVersion::GENERIC;
}

operation::ProgramWithCallbacks AllGather2D::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    tt::log_debug(tt::LogOp, "DEBUG: create_program is called");

    AllGather2DVersion version = select_version(input_tensors[0]);

    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));
    const auto num_devices = this->num_devices;

    switch (version) {
        case AllGather2DVersion::GENERIC:
        default:
            log_trace(tt::LogOp, "Running generic all_gather_async_multi_core_with_workers");
            return all_gather_2D_multi_core_with_workers(
                input_tensors[0],
                output_tensors[0],
                this->device_coord,
                this->grid_size,
                this->num_rows,
                this->num_cols,
                this->output_mem_config,
                this->topology,
                this->semaphore,
                this->lower_pages,
                this->higher_pages,
                num_devices,
                this->page_size,
                this->is_horizontal,
                this->semaphore_target_value);
    }
}

operation::ProgramWithCallbacks AllGatherAsync::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    tt::log_debug(tt::LogOp, "DEBUG: create_program is called");

    AllGatherAsyncVersion version = select_version(input_tensors[0]);

    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));

    switch (version) {
        case AllGatherAsyncVersion::MINIMAL_INTERLEAVED_32:
            log_trace(
                tt::LogOp,
                "Detected all gather specialized shape. all_gather_async_minimal_interleaved_dim3_1_1_32_any is "
                "called");
            return all_gather_async_minimal_interleaved_dim3_1_1_32_any(
                input_tensors[0],
                this->forward_device,
                this->backward_device,
                output_tensors[0],
                this->dim,
                this->num_links,
                this->ring_size,
                this->ring_index,
                this->topology,
                this->semaphore,
                this->sub_device_id,
                this->enable_persistent_fabric_mode);

        case AllGatherAsyncVersion::LLAMA_POST_BINARY_MATMUL:
            log_trace(
                tt::LogOp,
                "Detected all gather specialized shape. all_gather_async_llama_post_binary_matmul is called");
            return all_gather_async_llama_post_binary_matmul(
                input_tensors[0],
                this->forward_device,
                this->backward_device,
                output_tensors[0],
                this->dim,
                this->num_links,
                this->ring_size,
                this->ring_index,
                this->topology,
                this->semaphore,
                this->sub_device_id,
                this->enable_persistent_fabric_mode);

        case AllGatherAsyncVersion::GENERIC:
        default:
            log_trace(tt::LogOp, "Running generic all_gather_async_multi_core_with_workers");
            return all_gather_async_multi_core_with_workers(
                input_tensors[0],
                this->forward_device,
                this->backward_device,
                output_tensors[0],
                this->dim,
                this->num_links,
                this->ring_size,
                this->ring_index,
                this->topology,
                this->semaphore,
                this->sub_device_id,
                this->enable_persistent_fabric_mode);
    }
}

const operation::Hash AllGather2D::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    AllGather2DVersion version = select_version(input_tensors[0]);
    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));
    auto input_shape = input_tensors[0].get_padded_shape();
    auto input_memory_layout = input_tensors[0].get_layout();
    auto input_dtype = input_tensors[0].get_dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    // TODO figure out if we need to hash device coordinate

    // An IF condition can be made if the models team adds custom 2D versions
    uint32_t semaphore_address = this->semaphore.address();
    return operation::hash_operation<AllGatherAsync>(
        this->grid_size,
        this->num_rows,
        this->num_cols,
        this->num_devices,
        this->dim,
        this->lower_pages,
        this->higher_pages,
        this->page_size,
        this->is_horizontal,
        this->topology,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config,
        semaphore_address);
}

const operation::Hash AllGatherAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    AllGatherAsyncVersion version = select_version(input_tensors[0]);
    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));
    auto input_shape = input_tensors[0].get_padded_shape();
    auto input_memory_layout = input_tensors[0].get_layout();
    auto input_dtype = input_tensors[0].get_dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    if (version == AllGatherAsyncVersion::GENERIC) {
        // Generic version should hash semaphore address as well
        uint32_t semaphore_address = this->semaphore.address();
        return operation::hash_operation<AllGatherAsync>(
            this->dim,
            this->num_links,
            this->ring_size,
            this->ring_index,
            this->output_mem_config,
            this->topology,
            input_shape,
            input_memory_layout,
            input_dtype,
            input_memory_config,
            semaphore_address);
    }
    return operation::hash_operation<AllGatherAsync>(
        this->dim,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->output_mem_config,
        this->topology,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

namespace operations {
namespace experimental {
namespace ccl {

Tensor all_gather_async(
    const Tensor& input_tensor,
    const uint32_t dim,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<SubDeviceId> sub_device_id,
    bool enable_persistent_fabric_mode,
    bool uses_2d_fabric) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "all_gather_async op is only supported for Fast Dispatch");
    auto devices = input_tensor.get_workers();
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "all_gather_async op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;

    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    tt::log_debug(
        tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", devices.size(), num_links);
    tt::log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    // create this semaphore for all cores since we don't know which core will be used for teardown draining
    std::vector<GlobalSemaphore> semaphores = multi_device_global_semaphore.global_semaphores;

    if (!uses_2d_fabric) {
        operation::launch_op(
            [dim,
             num_links,
             num_devices,
             memory_config,
             devices,
             ccl_topology,
             semaphores,
             sub_device_id,
             enable_persistent_fabric_mode](
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                const auto& input_tensor = input_tensors.at(0);

                return operation::run(
                    ttnn::ccl::all_gather_detail::create_all_gather_async_struct(
                        input_tensor,
                        dim,
                        num_links,
                        memory_config,
                        devices,
                        ccl_topology,
                        semaphores,
                        sub_device_id,
                        enable_persistent_fabric_mode),
                    {input_tensor});
            },
            {input_tensor},
            output_tensors);
    } else {
        TT_FATAL(false, "Can't perform 2D fabric without supplying the mesh device");
    }
    return output_tensors.at(0);
}

Tensor all_gather_async(
    const Tensor& input_tensor,
    const int32_t dim,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    bool transpose_mesh_dimension,
    bool uses_2d_fabric) {
    return all_gather_async(
        input_tensor,
        dim,
        0,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        memory_config,
        num_preferred_links,
        std::nullopt,
        true,
        transpose_mesh_dimension,
        uses_2d_fabric);
}

Tensor all_gather_async(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<SubDeviceId> sub_device_id,
    bool enable_persistent_fabric_mode,
    bool transpose_mesh_dimension,
    bool uses_2d_fabric) {
    int32_t rank = input_tensor.get_logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);
    std::vector<GlobalSemaphore> semaphores = multi_device_global_semaphore.global_semaphores;
    if (!uses_2d_fabric) {
        const auto mesh_view = mesh_device.get_view();
        std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
        TT_FATAL(
            topology == ttnn::ccl::Topology::Linear,
            "This all_gather API with cluster_axis is currently supported only for the Linear topology");
        std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
        operation::launch_op(
            [gather_dim,
             num_preferred_links,
             memory_config,
             mesh_view,
             cluster_axis,
             num_devices,
             topology,
             semaphores,
             sub_device_id,
             enable_persistent_fabric_mode](
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                const auto& input_device_tensor = input_tensors.at(0);

                TT_FATAL(
                    mesh_view.is_mesh_2d(),
                    "all-gather invoked with cluster_axis API on >2D mesh, which is currently unsupported");
                const auto coordinate = mesh_view.find_device(input_device_tensor.device()->id());
                std::vector<IDevice*> devices = (cluster_axis == 0) ? mesh_view.get_devices_on_column(coordinate[1])
                                                                    : mesh_view.get_devices_on_row(coordinate[0]);

                const auto& input_tensor = input_tensors.at(0);

                return operation::run(
                    ttnn::ccl::all_gather_detail::create_all_gather_async_struct(
                        input_device_tensor,
                        gather_dim,
                        num_preferred_links.has_value() ? num_preferred_links.value() : 1,
                        memory_config,
                        devices,
                        topology,
                        semaphores,
                        sub_device_id,
                        enable_persistent_fabric_mode),
                    {input_tensor});
            },
            {input_tensor},
            output_tensors);
        return output_tensors.at(0);
    }
    else {
      TT_FATAL(enable_persistent_fabric_mode, "Persistent fabric mode is necessary for 2D fabric");
      return ttnn::ccl::all_gather_detail::all_gather_async_2D(
          input_tensor,
          gather_dim,
          rank,
          mesh_device,
          topology,
          semaphores,
          memory_config,
          num_preferred_links,
          transpose_mesh_dimension);
    }
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
