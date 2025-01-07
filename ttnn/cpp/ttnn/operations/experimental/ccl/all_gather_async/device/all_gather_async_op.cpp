// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async_op.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/cpp/ttnn/global_semaphore.hpp"

#include "tt_metal/host_api.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {
namespace ccl {
namespace all_gather_detail {

AllGatherAsync create_all_gather_async_struct(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<Device*>& devices,
    const ttnn::ccl::Topology topology,
    const std::optional<std::vector<GlobalSemaphore>>& semaphores,
    bool enable_persistent_fabric_mode) {
    uint32_t num_devices = devices.size();

    std::optional<Device*> forward_device = std::nullopt;
    std::optional<Device*> backward_device = std::nullopt;
    std::optional<GlobalSemaphore> semaphore = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < num_devices; ++i) {
        if (devices.at(i) == input_tensor.device()) {
            device_index = i;
            if (semaphores.has_value()) {
                semaphore = semaphores.value().at(i);  // Get raw pointer
            }
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
        semaphore,
        enable_persistent_fabric_mode};
}

std::optional<std::vector<GlobalSemaphore>> get_global_semaphores(
    const std::vector<Device*>& devices,
    const CoreRange& core_range,
    std::optional<SubDeviceId> subdevice_id,
    bool create_semaphore_handles) {
    std::optional<std::vector<GlobalSemaphore>> semaphores_opt;
    if (create_semaphore_handles) {
        std::vector<GlobalSemaphore> semaphores;
        for (const auto& device : devices) {
            auto worker_subdevice_id =
                subdevice_id.has_value() ? std::vector<SubDeviceId>{subdevice_id.value()} : std::vector<SubDeviceId>{};

            auto sem =
                global_semaphore::create_global_semaphore(device, core_range, 0, BufferType::L1, worker_subdevice_id);
            log_trace(tt::LogOp, "Created semaphore at address {} for device {}", sem.address(), device->id());
            semaphores.push_back(std::move(sem));
        }
        // HACK: assert every address is the same
        TT_FATAL(
            std::all_of(
                semaphores.begin(),
                semaphores.end(),
                [&](const auto& sem) { return sem.address() == semaphores.front().address(); }),
            "[Hack] All semaphores should have the same address");
        semaphores_opt = std::move(semaphores);
    } else {
        semaphores_opt = std::nullopt;
    }

    return semaphores_opt;
}

}  // namespace all_gather_detail
}  // namespace ccl

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

static void validate_output_tensor_allocation(const std::vector<Tensor>& output_tensors) {
    for (const auto& output_tensor : output_tensors) {
        const auto& buffers = output_tensor.buffers();
        const auto first_address = buffers.front()->address();
        TT_FATAL(
            std::all_of(
                buffers.begin(),
                buffers.end(),
                [&first_address](const auto& buffer) {
                    return buffer != nullptr && buffer->address() == first_address;
                }),
            "Output buffers for all_gather async must be lock-step allocated but some of the tensors were allocated at "
            "different addresses across devices.");
    }
}

std::vector<ttnn::SimpleShape> AllGatherAsync::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    auto shape = input_tensors[0].get_padded_shape();  // TODO: Replace with get_logical_shape()
    shape[this->dim] *= this->ring_size;
    return std::vector<ttnn::SimpleShape>(input_tensors.size(), shape);
}

std::vector<Tensor> AllGatherAsync::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto output_tensors = std::vector<Tensor>();
    output_tensors.reserve(1);
    auto tile = input_tensor.get_tensor_spec().tile();
    if (this->output_mem_config.is_sharded()) {
        output_tensors.push_back(create_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            input_tensor.get_dtype(),
            input_tensor.get_layout(),
            input_tensor.device(),
            this->output_mem_config,
            tile));
    } else {
        output_tensors = operation::generic_create_output_tensors(
            *this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config, tile);
    }
    log_debug(tt::LogOp, "DEBUG: output_tensors[0] address: {}", output_tensors.at(0).buffer()->address());
    return output_tensors;
}

operation::ProgramWithCallbacks AllGatherAsync::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    tt::log_debug(tt::LogOp, "DEBUG: create_program is called");
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
        this->enable_persistent_fabric_mode);
}

const operation::Hash AllGatherAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    return operation::hash_operation<AllGatherAsync>(
        this->dim, this->num_links, this->ring_size, this->ring_index, this->output_mem_config, this->topology);
}



namespace operations {
namespace experimental {
namespace ccl {

Tensor all_gather_async(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<SubDeviceId> subdevice_id,
    bool enable_persistent_fabric_mode,
    bool create_semaphore_handles) {
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
    CoreCoord grid_size = devices[0]->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});

    std::optional<std::vector<GlobalSemaphore>> semaphores_opt =
        ttnn::ccl::all_gather_detail::get_global_semaphores(devices, core_grid, subdevice_id, create_semaphore_handles);

    operation::launch_op(
        [dim,
         num_links,
         num_devices,
         memory_config,
         devices,
         ccl_topology,
         semaphores_opt,
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
                    semaphores_opt,
                    enable_persistent_fabric_mode),
                {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

Tensor all_gather_async(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<SubDeviceId> subdevice_id,
    bool enable_persistent_fabric_mode,
    bool create_semaphore_handles) {
    TT_FATAL(
        topology == ttnn::ccl::Topology::Linear,
        "This all_gather API with cluster_axis is currently supported only for the Linear topology");
    const auto mesh_view = mesh_device.get_view();
    auto devices = input_tensor.get_workers();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    int32_t rank = input_tensor.get_logical_shape().rank();

    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    CoreCoord grid_size = devices[0]->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    std::optional<std::vector<GlobalSemaphore>> semaphores_opt =
        ttnn::ccl::all_gather_detail::get_global_semaphores(devices, core_grid, subdevice_id, create_semaphore_handles);

    operation::launch_op(
        [gather_dim,
         num_preferred_links,
         memory_config,
         mesh_view,
         cluster_axis,
         num_devices,
         topology,
         semaphores_opt,
         enable_persistent_fabric_mode](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_device_tensor = input_tensors.at(0);

            const auto coordinate = mesh_view.find_device(input_device_tensor.device()->id());
            std::vector<Device*> devices = (cluster_axis == 0) ? mesh_view.get_devices_on_column(coordinate.col)
                                                               : mesh_view.get_devices_on_row(coordinate.row);

            const auto& input_tensor = input_tensors.at(0);

            return operation::run(
                ttnn::ccl::all_gather_detail::create_all_gather_async_struct(
                    input_device_tensor,
                    gather_dim,
                    num_preferred_links.has_value() ? num_preferred_links.value() : 1,
                    memory_config,
                    devices,
                    topology,
                    semaphores_opt,
                    enable_persistent_fabric_mode),
                {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
