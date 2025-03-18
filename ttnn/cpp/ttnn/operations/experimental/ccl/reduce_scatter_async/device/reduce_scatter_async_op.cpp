// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include "cpp/ttnn/global_semaphore.hpp"

#include <ranges>
#include <algorithm>
#include <cstdint>
#include <optional>

using namespace tt::tt_metal;

namespace ttnn {
namespace ccl {
namespace reduce_scatter_detail {

ReduceScatterAsync create_reduce_scatter_struct(
    const Tensor& input_tensor,
    const ttnn::operations::binary::BinaryOpType binary_op_type,
    const uint32_t scatter_dim,
    const MemoryConfig& output_mem_config,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology,
    std::optional<std::vector<Tensor>> forward_output_tensors,
    std::optional<std::vector<Tensor>> backward_output_tensors,
    std::optional<size_t> num_links_preferred,
    const std::vector<GlobalSemaphore>& from_remote_sems,
    const std::vector<GlobalSemaphore>& to_remote_sems,
    std::optional<SubDeviceId> sub_device_id,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& fabric_handle) {
    uint32_t num_devices = devices.size();

    auto [device_index, sender_device_id, receiver_device_id] =
        get_device_index_and_sender_receiver_ids(input_tensor, devices, topology);

    TT_FATAL(
        receiver_device_id != std::nullopt || sender_device_id != std::nullopt,
        "Error, Reduce-scatter was unable to identify either a sender or receiver device ID and atleast one must be "
        "identified for a valid Reduce-scatter configuration. The input mesh tensor or Reduce-scatter arguments may be "
        "incorrect");

    auto find_device = [](const std::vector<IDevice*>& devices, std::optional<chip_id_t> id) -> std::optional<IDevice*> {
        if (id == std::nullopt) {
            return std::nullopt;
        }
        auto device = std::find_if(
            devices.begin(), devices.end(), [id_ = id.value()](IDevice const* d) { return d->id() == id_; });
        TT_FATAL(
            device != devices.end(),
            "Device with ID {} not found in the list of devices, but it should be here since it was provided "
            "previously",
            id.value());
        return *device;
    };

    GlobalSemaphore from_remote_sem = from_remote_sems.at(device_index);
    GlobalSemaphore to_remote_sem = to_remote_sems.at(device_index);

    return ttnn::ReduceScatterAsync{
        binary_op_type,
        scatter_dim,
        num_devices,
        device_index,
        find_device(devices, receiver_device_id),
        find_device(devices, sender_device_id),
        output_mem_config,
        topology,
        forward_output_tensors,
        backward_output_tensors,
        num_links_preferred,
        from_remote_sem,
        to_remote_sem,
        sub_device_id,
        fabric_handle};
}
}  // namespace reduce_scatter_detail
}  // namespace ccl

void ReduceScatterAsync::validate(const std::vector<Tensor>& input_tensors) const {
    for (auto const& t : input_tensors) {
        TT_FATAL(
            t.get_padded_shape()[this->scatter_dim] / this->ring_size > 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size",
            this->scatter_dim);
        TT_FATAL(
            t.get_padded_shape()[this->scatter_dim] % this->ring_size == 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size",
            this->scatter_dim);
    }
}

std::vector<ttnn::TensorSpec> ReduceScatterAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shape = input_tensor.get_logical_shape();
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

    bool is_tile_layout = input_tensor.get_layout() == Layout::TILE;
    std::optional<tt::tt_metal::Tile> tile =
        is_tile_layout ? input_tensor.get_tensor_spec().tile() : std::optional<tt::tt_metal::Tile>(std::nullopt);

    std::vector<TensorSpec> output_tensors;
    output_tensors.reserve(5);
    // real_output_tensor
    output_tensors.emplace_back(TensorSpec(
        shape, TensorLayout(input_tensor.get_dtype(), PageConfig(input_tensor.get_layout(), tile), output_mem_config)));
    // temporary_input_from_remote_tensor_for_forward_direction
    output_tensors.emplace_back(input_tensor.get_tensor_spec());
    // temporary_input_from_remote_tensor_for_backward_direction
    output_tensors.emplace_back(input_tensor.get_tensor_spec());
    // temporary_partial_output_tensor_for_forward_direction
    output_tensors.emplace_back(TensorSpec(
        shape, TensorLayout(input_tensor.get_dtype(), PageConfig(input_tensor.get_layout(), tile), output_mem_config)));
    // temporary_partial_output_tensor_for_backward_direction
    output_tensors.emplace_back(TensorSpec(
        shape,
        TensorLayout(input_tensor.get_dtype(), PageConfig(input_tensor.get_layout(), tile), this->output_mem_config)));

    return output_tensors;
}

operation::ProgramWithCallbacks ReduceScatterAsync::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
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
        this->forward_device,
        this->backward_device,
        this->binary_op_type,
        this->scatter_dim,
        this->ring_size,
        this->ring_index,
        this->topology,
        this->num_links_preferred,
        this->from_remote_sem,
        this->to_remote_sem,
        this->sub_device_id,
        this->fabric_handle);
}

operation::Hash ReduceScatterAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    auto input_shape = input_tensors[0].get_padded_shape();
    auto input_memory_layout = input_tensors[0].get_layout();
    auto input_dtype = input_tensors[0].get_dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    return operation::hash_operation<ReduceScatterAsync>(
        this->binary_op_type,
        this->scatter_dim,
        this->ring_size,
        this->ring_index,
        this->topology,
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
Tensor reduce_scatter(
    const Tensor& input_tensor,
    const int32_t dim,
    const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
    const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
    ttnn::operations::reduction::ReduceType math_op,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_links_preferred,
    std::optional<SubDeviceId> worker_subdevice_id_opt,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(math_op);
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "reduce_scatter op is only supported for Fast Dispatch");

    ttnn::ccl::Topology ccl_topology = topology;
    auto devices = input_tensor.get_workers();
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "reduce_scatter op will only work for num_devices > 1, but has {}", num_devices);
    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }

    int16_t rank = input_tensor.get_logical_shape().rank();
    int16_t scatter_dim = (dim < 0) ? rank + dim : dim;
    TT_FATAL(
        scatter_dim >= -rank && scatter_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<GlobalSemaphore> from_remote_inputs_semaphores =
        from_remote_multi_device_global_semaphore.global_semaphores;

    std::vector<GlobalSemaphore> to_remote_inputs_semaphores =
        to_remote_multi_device_global_semaphore.global_semaphores;

    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({input_tensor})),
        Tensor(operation::get_workers_for_op_output({input_tensor})),
        Tensor(operation::get_workers_for_op_output({input_tensor})),
        Tensor(operation::get_workers_for_op_output({input_tensor})),
        Tensor(operation::get_workers_for_op_output({input_tensor}))};
    TT_FATAL(
        output_tensors.size() == 5,
        "Reduce scatter requires 5 output tensors. 1 is real and the others are temporaries");
    operation::launch_op(
        [binary_op_type,
         from_remote_inputs_semaphores,
         to_remote_inputs_semaphores,
         scatter_dim,
         output_mem_config,
         ccl_topology,
         devices,
         num_links_preferred,
         output_tensors,
         worker_subdevice_id_opt,
         fabric_handle](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor = input_tensors.at(0);

            return operation::run(
                ttnn::ccl::reduce_scatter_detail::create_reduce_scatter_struct(
                    input_tensor,
                    binary_op_type,
                    scatter_dim,
                    output_mem_config,
                    devices,
                    ccl_topology,
                    std::nullopt,
                    std::nullopt,
                    num_links_preferred,
                    from_remote_inputs_semaphores,
                    to_remote_inputs_semaphores,
                    worker_subdevice_id_opt,
                    fabric_handle),
                {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

Tensor reduce_scatter(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
    const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
    ttnn::operations::reduction::ReduceType reduce_op,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_links_preferred,
    std::optional<SubDeviceId> worker_subdevice_id_opt,  // TODO make reference
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(reduce_op);
    int16_t rank = input_tensor.get_logical_shape().rank();
    int16_t scatter_dim = (dim < 0) ? rank + dim : dim;
    const auto mesh_view = mesh_device.get_view();
    auto devices = input_tensor.get_workers();

    std::vector<GlobalSemaphore> from_remote_inputs_semaphores =
        from_remote_multi_device_global_semaphore.global_semaphores;

    std::vector<GlobalSemaphore> to_remote_inputs_semaphores =
        to_remote_multi_device_global_semaphore.global_semaphores;

    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({input_tensor})),
        Tensor(operation::get_workers_for_op_output({input_tensor})),
        Tensor(operation::get_workers_for_op_output({input_tensor})),
        Tensor(operation::get_workers_for_op_output({input_tensor})),
        Tensor(operation::get_workers_for_op_output({input_tensor}))};
    TT_FATAL(
        output_tensors.size() == 5,
        "Reduce scatter requires 5 output tensors. 1 is real and the others are temporaries");
    operation::launch_op(
        [binary_op_type,
         from_remote_inputs_semaphores,
         to_remote_inputs_semaphores,
         scatter_dim,
         output_mem_config,
         mesh_view,
         cluster_axis,
         topology,
         devices,
         num_links_preferred,
         output_tensors,
         worker_subdevice_id_opt,
         fabric_handle](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_device_tensor = input_tensors.at(0);

            TT_FATAL(
                mesh_view.is_mesh_2d(),
                "reduce-scatter invoked with cluster_axis API on >2D mesh, which is currently unsupported");
            const auto coordinate = mesh_view.find_device(input_device_tensor.device()->id());
            std::vector<IDevice*> devices = (cluster_axis == 0) ? mesh_view.get_devices_on_column(coordinate[1])
                                                                : mesh_view.get_devices_on_row(coordinate[0]);

            const auto& input_tensor = input_tensors.at(0);

            return operation::run(
                ttnn::ccl::reduce_scatter_detail::create_reduce_scatter_struct(
                    input_tensor,
                    binary_op_type,
                    scatter_dim,
                    output_mem_config,
                    devices,
                    topology,
                    std::nullopt,
                    std::nullopt,
                    num_links_preferred,
                    from_remote_inputs_semaphores,
                    to_remote_inputs_semaphores,
                    worker_subdevice_id_opt,
                    fabric_handle),
                {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

};  // namespace ttnn
