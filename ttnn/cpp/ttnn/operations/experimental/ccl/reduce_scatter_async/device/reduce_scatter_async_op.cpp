// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/global_semaphore.hpp"

#include <ranges>
#include <algorithm>
#include <cstdint>
#include <optional>

using namespace tt::tt_metal;

namespace ttnn {

void ReduceScatterAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].layout();
    const auto& dtype = input_tensors[0].dtype();
    for (auto const& t : input_tensors) {
        TT_FATAL(
            t.padded_shape()[this->scatter_dim] / this->ring_size > 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size",
            this->scatter_dim);
        TT_FATAL(
            t.padded_shape()[this->scatter_dim] % this->ring_size == 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size",
            this->scatter_dim);
    }
    if (output_tensors.size() > 0 && output_tensors[0].has_value()) {
        TT_FATAL(
            output_tensors.size() == 5,
            "Error, Number of output tensors should be 5 but has {}",
            output_tensors.size());
        for (size_t t = 0; t < output_tensors.size(); ++t) {
            const auto& output_tensor = output_tensors[t];
            TT_FATAL(
                output_tensor.value().storage_type() == StorageType::DEVICE,
                "Operands to all_gather need to be on device!");
            TT_FATAL(
                output_tensor.value().layout() == layout,
                "Error, Output tensor layout should be same as input tensor layout but has {}",
                output_tensor.value().layout());
            TT_FATAL(
                output_tensor.value().dtype() == dtype,
                "Error, Output tensor dtype should be same as input tensor dtype but has {}",
                output_tensor.value().dtype());
            TT_FATAL(
                output_tensor.value().tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
                "Error, Output tensor page config should be same as input tensor page config but has {}",
                output_tensor.value().tensor_spec().page_config());
            TT_FATAL(
                output_tensor.value().memory_config() == this->output_mem_config,
                "Error, Output tensor memory config should be same as output_mem_config but has {}",
                output_tensor.value().memory_config());

            // check memory layout
            TT_FATAL(
                output_tensor.value().memory_config().memory_layout() == input_tensor.memory_config().memory_layout(),
                "Error, Output tensor memory layout should be same as input tensor memory layout but has {}",
                output_tensor.value().memory_config().memory_layout());

            // check the output tensor size
            auto output_shape = output_tensor.value().padded_shape();
            auto input_shape = input_tensor.padded_shape();

            TT_FATAL(
                output_shape.size() == input_shape.size(),
                "Error, Output tensor shape should have same number of dimensions as input tensor but has {}",
                output_shape.size());
            for (size_t i = 0; i < input_shape.size(); ++i) {
                if (i == this->scatter_dim && (t == 0 || t == 3 || t == 4)) {
                    TT_FATAL(
                        output_shape[i] == input_shape[i] / this->ring_size,
                        "Error, Output tensor shape at dimension {} should be {} but has {}",
                        i,
                        input_shape[i] / this->ring_size,
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
        }
    }
}

std::vector<ttnn::TensorSpec> ReduceScatterAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shape = input_tensor.logical_shape();
    TT_FATAL(
        shape[this->scatter_dim] % this->ring_size == 0,
        "The size of the scatter dimension must be a multiple of the ring size. Dimension size: {}, ring Size: {}",
        shape[this->scatter_dim],
        this->ring_size);
    shape[this->scatter_dim] /= this->ring_size;

    // output tensors
    // 0. final (real) output_tensor
    // 1. input_tensor_from_remote_forward_direction (shape of input tensor)
    // 2. input_tensor_from_remote_backward_direction (shape of input tensor)
    // 3. partial_output_tensor_forward_direction (shape of output tensor)
    // 4. partial_output_tensor_backward_direction (shape of output tensor)

    bool is_tile_layout = input_tensor.layout() == Layout::TILE;
    std::optional<tt::tt_metal::Tile> tile =
        is_tile_layout ? input_tensor.tensor_spec().tile() : std::optional<tt::tt_metal::Tile>(std::nullopt);

    std::vector<TensorSpec> output_tensors;
    output_tensors.reserve(5);
    // real_output_tensor
    output_tensors.emplace_back(TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout(), tile), output_mem_config)));
    // temporary_input_from_remote_tensor_for_forward_direction
    output_tensors.emplace_back(input_tensor.tensor_spec());
    // temporary_input_from_remote_tensor_for_backward_direction
    output_tensors.emplace_back(input_tensor.tensor_spec());
    // temporary_partial_output_tensor_for_forward_direction
    output_tensors.emplace_back(TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout(), tile), output_mem_config)));
    // temporary_partial_output_tensor_for_backward_direction
    output_tensors.emplace_back(TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout(), tile), this->output_mem_config)));

    return output_tensors;
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks ReduceScatterAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
};

operation::ProgramWithCallbacks ReduceScatterAsync::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    std::vector<IDevice*> devices;
    if (this->cluster_axis.has_value()) {
        const auto& mesh_view = mesh_device->get_view();
        devices =
            (cluster_axis == 0) ? mesh_view.get_devices_on_column(coord[1]) : mesh_view.get_devices_on_row(coord[0]);
    } else {
        devices = this->devices;
    }

    auto target_device =
        input_tensors[0].mesh_device() ? input_tensors[0].mesh_device()->get_device(coord) : input_tensors[0].device();

    ttnn::ccl::SenderRecieverConfig config =
        ttnn::ccl::get_device_sender_receiver_config(target_device, devices, this->topology);

    TT_FATAL(
        config.receiver_device_id != std::nullopt || config.sender_device_id != std::nullopt,
        "Error, Reduce-scatter was unable to identify either a sender or receiver device ID and atleast one must be "
        "identified for a valid Reduce-scatter configuration. The input mesh tensor or Reduce-scatter arguments may be "
        "incorrect");

    auto find_device = [](const std::vector<IDevice*>& devices,
                          std::optional<chip_id_t> id) -> std::optional<IDevice*> {
        if (id == std::nullopt) {
            return std::nullopt;
        }
        auto device = std::find_if(
            devices.begin(), devices.end(), [id_ = id.value()](const IDevice* d) { return d->id() == id_; });
        TT_FATAL(
            device != devices.end(),
            "Device with ID {} not found in the list of devices, but it should be here since it was provided "
            "previously",
            id.value());
        return *device;
    };

    std::optional<Tensor> foreward_direction_remote_output_tensor = std::nullopt;
    std::optional<Tensor> backward_direction_remote_output_tensor = std::nullopt;
    return ccl::reduce_scatter_detail::build_reduce_scatter_async_program(
        input_tensors.at(0),   // true input_tensor
        output_tensors.at(0),  // final output_tensor
        output_tensors.at(1),  // input_tensor_from_remote_forward_direction
        output_tensors.at(2),  // input_tensor_from_remote_backward_direction
        output_tensors.at(3),  // partial_output_tensor_forward_direction
        output_tensors.at(4),  // partial_output_tensor_backward_direction
        foreward_direction_remote_output_tensor,
        backward_direction_remote_output_tensor,
        target_device,
        find_device(devices, config.receiver_device_id),
        find_device(devices, config.sender_device_id),
        this->binary_op_type,
        this->scatter_dim,
        this->ring_size,
        config.device_index,
        this->topology,
        this->num_links_preferred,
        this->from_remote_sem,
        this->to_remote_sem,
        this->sub_device_id);
}

operation::Hash ReduceScatterAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    return operation::hash_operation<ReduceScatterAsync>(
        this->binary_op_type,
        this->scatter_dim,
        this->ring_size,
        this->topology,
        this->cluster_axis,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
ttnn::operations::binary::BinaryOpType convert_reduce_type_to_eltwise_type(
    ttnn::operations::reduction::ReduceType reduce_op) {
    // Leaving switch statement for future support of additional types.
    switch (reduce_op) {
        case ttnn::operations::reduction::ReduceType::Sum: return ttnn::operations::binary::BinaryOpType::ADD;
        default:
            TT_THROW("Reduce scatter only supports reduce_type Sum. Op type {} not supported.", reduce_op);
            return ttnn::operations::binary::BinaryOpType::ADD;
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace operations {
namespace experimental {
namespace ccl {

namespace {
Tensor reduce_scatter_impl(
    const Tensor& input_tensor,
    const int32_t dim,
    const GlobalSemaphore& from_remote_multi_device_global_semaphore,
    const GlobalSemaphore& to_remote_multi_device_global_semaphore,
    ttnn::operations::reduction::ReduceType math_op,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_links_preferred,
    std::optional<SubDeviceId> worker_subdevice_id_opt,
    const std::vector<IDevice*>& devices) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(math_op);
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "reduce_scatter op is only supported for Fast Dispatch");

    ttnn::ccl::Topology ccl_topology = topology;
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "reduce_scatter op will only work for num_devices > 1, but has {}", num_devices);
    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }

    int16_t rank = input_tensor.logical_shape().rank();
    int16_t scatter_dim = (dim < 0) ? rank + dim : dim;
    TT_FATAL(
        scatter_dim >= -rank && scatter_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    return operation::run(
               ttnn::ReduceScatterAsync(
                   devices,
                   /*mesh_device=*/nullptr,
                   binary_op_type,
                   scatter_dim,
                   num_devices,
                   output_mem_config,
                   ccl_topology,
                   num_links_preferred,
                   from_remote_multi_device_global_semaphore,
                   to_remote_multi_device_global_semaphore,
                   worker_subdevice_id_opt,
                   /*cluster_axis=*/std::nullopt),
               {input_tensor},
               {},
               {})
        .at(0);
}
Tensor reduce_scatter_impl(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& from_remote_multi_device_global_semaphore,
    const GlobalSemaphore& to_remote_multi_device_global_semaphore,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_tensors,
    ttnn::operations::reduction::ReduceType reduce_op,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_links_preferred,
    std::optional<SubDeviceId> worker_subdevice_id_opt /* TODO make reference */) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(reduce_op);
    int16_t rank = input_tensor.logical_shape().rank();
    int16_t scatter_dim = (dim < 0) ? rank + dim : dim;
    const auto mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "reduce-scatter invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    const uint32_t num_devices = cluster_axis == 0 ? mesh_view.num_rows() : mesh_view.num_cols();

    std::vector<std::optional<Tensor>> optional_output_tensors =
        persistent_output_tensors
            ? std::vector<std::optional<Tensor>>(persistent_output_tensors->begin(), persistent_output_tensors->end())
            : std::vector<std::optional<Tensor>>{};
    return operation::run(
               ttnn::ReduceScatterAsync(
                   /*devices=*/{},
                   &mesh_device,
                   binary_op_type,
                   scatter_dim,
                   num_devices,
                   output_mem_config,
                   topology,
                   num_links_preferred,
                   from_remote_multi_device_global_semaphore,
                   to_remote_multi_device_global_semaphore,
                   worker_subdevice_id_opt,
                   /*cluster_axis=*/cluster_axis),
               {input_tensor},
               {},
               optional_output_tensors)
        .at(0);
}
}  // namespace

Tensor reduce_scatter(
    const Tensor& input_tensor,
    const int32_t dim,
    const GlobalSemaphore& from_remote_multi_device_global_semaphore,
    const GlobalSemaphore& to_remote_multi_device_global_semaphore,
    ttnn::operations::reduction::ReduceType math_op,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_links_preferred,
    std::optional<SubDeviceId> worker_subdevice_id_opt) {
    return reduce_scatter_impl(
        input_tensor,
        dim,
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        math_op,
        output_mem_config,
        topology,
        num_links_preferred,
        worker_subdevice_id_opt,
        ttnn::ccl::get_active_physical_devices(input_tensor));
}

std::vector<Tensor> reduce_scatter(
    const std::vector<Tensor>& input_tensors,
    const int32_t dim,
    const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
    const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
    ttnn::operations::reduction::ReduceType math_op,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_links_preferred,
    std::optional<SubDeviceId> worker_subdevice_id_opt) {
    std::vector<IDevice*> devices;
    devices.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        devices.push_back(input_tensor.device());
    }
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        output_tensors.push_back(reduce_scatter_impl(
            input_tensors[i],
            dim,
            from_remote_multi_device_global_semaphore.global_semaphores[i],
            to_remote_multi_device_global_semaphore.global_semaphores[i],
            math_op,
            output_mem_config,
            topology,
            num_links_preferred,
            worker_subdevice_id_opt,
            devices));
    }
    return output_tensors;
}

Tensor reduce_scatter(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& from_remote_multi_device_global_semaphore,
    const GlobalSemaphore& to_remote_multi_device_global_semaphore,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_tensors,
    ttnn::operations::reduction::ReduceType reduce_op,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_links_preferred,
    std::optional<SubDeviceId> worker_subdevice_id_opt /* TODO make reference */) {
    return reduce_scatter_impl(
        input_tensor,
        dim,
        cluster_axis,
        mesh_device,
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        persistent_output_tensors,
        reduce_op,
        output_mem_config,
        topology,
        num_links_preferred,
        worker_subdevice_id_opt);
}

std::vector<Tensor> reduce_scatter(
    const std::vector<Tensor>& input_tensors,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
    const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_tensors,
    ttnn::operations::reduction::ReduceType reduce_op,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_links_preferred,
    std::optional<SubDeviceId> worker_subdevice_id_opt /* TODO make reference */) {
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        output_tensors.push_back(reduce_scatter_impl(
            input_tensors[i],
            dim,
            cluster_axis,
            mesh_device,
            from_remote_multi_device_global_semaphore.global_semaphores[i],
            to_remote_multi_device_global_semaphore.global_semaphores[i],
            persistent_output_tensors,
            reduce_op,
            output_mem_config,
            topology,
            num_links_preferred,
            worker_subdevice_id_opt));
    }
    return output_tensors;
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

};  // namespace ttnn
