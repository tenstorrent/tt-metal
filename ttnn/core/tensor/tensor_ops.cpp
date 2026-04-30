// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include "tensor/tensor_ops.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <ranges>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tracy/Tracy.hpp>
#include "ttnn/graph/graph_serialization.hpp"

#include <tt-metalium/experimental/tensor/tensor_apis.hpp>

namespace tt::tt_metal {

Tensor allocate_tensor_on_host(const TensorSpec& tensor_spec, distributed::MeshDevice* device) {
    auto distributed_host_buffer = DistributedHostBuffer::create(device->get_view());

    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(device->shape().mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(device->shape())) {
        coords.push_back(coord);
    }

    distributed_host_buffer.emplace_shards(
        coords,
        [&](const auto&) { return tensor_impl::allocate_host_buffer(tensor_spec); },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    // TODO (#25340): Implement correct logic and add test for this
    return Tensor(HostTensor(std::move(distributed_host_buffer), tensor_spec, TensorTopology{}));
}

Tensor create_device_tensor(
    const TensorSpec& tensor_spec, distributed::MeshDevice* mesh_device, std::optional<TensorTopology> tensor_topology) {
    GraphTracker::instance().track_function_start(
        "tt::tt_metal::create_device_tensor",
        tensor_spec.logical_shape(),
        tensor_spec.tensor_layout().get_data_type(),
        tensor_spec.tensor_layout().get_layout(),
        mesh_device,
        tensor_spec.tensor_layout().get_memory_config());

    Tensor output;
    auto topology = std::invoke([&]() {
        if (tensor_topology.has_value()) {
            return std::move(*tensor_topology);
        }
        // TODO (#25340): Implement correct logic and add test for this
        // River: why are we constructing the topology here like this instead of using the
        // TensorTopology::create_fully_replicated_tensor_topology function?
        //
        // Use Replicate as default value for placements in MeshMapperConfig
        const auto& mesh_shape = mesh_device->shape();
        ttsl::SmallVector<distributed::MeshMapperConfig::Placement> placements(
            mesh_shape.dims(), tt::tt_metal::distributed::MeshMapperConfig::Replicate{});

        std::vector<distributed::MeshCoordinate> coordinates;
        coordinates.reserve(mesh_shape.mesh_size());
        for (const auto& coord : distributed::MeshCoordinateRange(mesh_shape)) {
            coordinates.push_back(coord);
        }

        return TensorTopology{mesh_shape, placements, std::move(coordinates)};
    });

    output = Tensor(MeshTensor::allocate_on_device(*mesh_device, tensor_spec, topology));
    output = tt::tt_metal::set_tensor_id(output);

    GraphTracker::instance().track_function_end(output);

    return output;
}
}  // namespace tt::tt_metal

namespace tt::tt_metal {

Tensor to_device(
    const Tensor& input_tensor,
    distributed::MeshDevice* mesh_device,
    ttsl::optional_reference<const MemoryConfig> mem_config,
    std::optional<QueueId> cq_id) {
    GraphTracker::instance().track_function_start("Tensor::to_device", input_tensor, mesh_device, mem_config);
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        TT_ASSERT(input_tensor.device() == mesh_device, "Currently do not support moving between devices");
        GraphTracker::instance().track_function_end(input_tensor);
        return input_tensor;
    }
    auto& cq = mesh_device->mesh_command_queue(raw_optional(cq_id));
    Tensor device_tensor;
    if (is_uniform_write(input_tensor.host_tensor(), *mesh_device)) {
        device_tensor = Tensor(enqueue_write_tensor(cq, input_tensor.host_tensor(), *mesh_device, mem_config));
    } else {
        auto [mesh_tensor, coords] =
            non_uniform_data_movement::enqueue_write_tensor(cq, input_tensor.host_tensor(), *mesh_device, mem_config);
        device_tensor = Tensor(DeviceStorage(std::move(mesh_tensor), std::move(coords)));
    }
    GraphTracker::instance().track_function_end(device_tensor);
    return device_tensor;
}

void copy_to_device(const Tensor& host_tensor, Tensor& device_tensor, std::optional<tt::tt_metal::QueueId> cq_id) {
    GraphTracker::instance().track_function_start("tt::tt_metal::copy_to_device", host_tensor, device_tensor, cq_id);
    auto& cq = device_tensor.device()->mesh_command_queue(raw_optional(cq_id));
    if (is_uniform_write(host_tensor.host_tensor(), *device_tensor.device())) {
        enqueue_write_tensor(cq, host_tensor.host_tensor(), device_tensor.device_storage().get_mesh_tensor());
    } else {
        auto coords = non_uniform_data_movement::enqueue_write_tensor(
            cq, host_tensor.host_tensor(), device_tensor.device_storage().get_mesh_tensor());
        device_tensor.device_storage() = DeviceStorage(device_tensor.device_storage(), std::move(coords));
    }
    device_tensor = tt::tt_metal::set_tensor_id(device_tensor);
    GraphTracker::instance().track_function_end(device_tensor);
}

void copy_to_device(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    Tensor& device_tensor,
    const std::optional<BufferRegion>& region) {
    GraphTracker::instance().track_function_start("tt::tt_metal::copy_to_device", queue, src, device_tensor, region);
    enqueue_write_tensor(queue, src, device_tensor.device_storage().get_mesh_tensor(), region);
    GraphTracker::instance().track_function_end(device_tensor);
}

void copy_to_host(
    distributed::MeshCommandQueue& queue,
    const Tensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region,
    bool blocking) {
    GraphTracker::instance().track_function_start(
        "tt::tt_metal::copy_to_host", queue, device_tensor, dst, region, blocking);
    enqueue_read_tensor(queue, device_tensor.mesh_tensor(), dst, region, blocking);
    GraphTracker::instance().track_function_end(device_tensor);
}

void copy_to_host(const Tensor& device_tensor, Tensor& host_tensor, bool blocking, std::optional<QueueId> cq_id) {
    GraphTracker::instance().track_function_start(
        "tt::tt_metal::copy_to_host", device_tensor, host_tensor, blocking, cq_id);
    auto& cq = device_tensor.device()->mesh_command_queue(raw_optional(cq_id));
    if (device_tensor.device_storage().is_uniform_storage()) {
        enqueue_read_tensor(cq, device_tensor.mesh_tensor(), host_tensor.host_storage().host_tensor(), blocking);
    } else {
        auto coords = device_tensor.device_storage().get_coords();
        non_uniform_data_movement::enqueue_read_tensor(
            cq, device_tensor.mesh_tensor(), host_tensor.host_storage().host_tensor(), coords, blocking);
    }
    GraphTracker::instance().track_function_end(host_tensor);
}

Tensor cpu(const Tensor& input_tensor, bool blocking, std::optional<QueueId> cq_id) {
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        return input_tensor;
    }

    GraphTracker::instance().track_function_start("Tensor::cpu", input_tensor, blocking);

    auto& cq = input_tensor.device()->mesh_command_queue(raw_optional(cq_id));
    Tensor output;
    if (input_tensor.device_storage().is_uniform_storage()) {
        output = Tensor(enqueue_read_tensor(cq, input_tensor.mesh_tensor(), blocking));
    } else {
        auto coords = input_tensor.device_storage().get_coords();
        output =
            Tensor(non_uniform_data_movement::enqueue_read_tensor(cq, input_tensor.mesh_tensor(), coords, blocking));
    }
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor to_layout(const Tensor& input_tensor, Layout target_layout) {
    GraphTracker::instance().track_function_start("Tensor::to_layout", input_tensor, target_layout);
    TT_FATAL(is_cpu_tensor(input_tensor), "Tensor must be on host for to_layout conversion");
    Tensor output = Tensor(tt::tt_metal::to_layout(input_tensor.host_tensor(), target_layout));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor pad(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value) {
    GraphTracker::instance().track_function_start(
        "Tensor::pad", input_tensor, output_padded_shape, input_tensor_start, pad_value);
    TT_FATAL(is_cpu_tensor(input_tensor), "Tensor must be on host for padding");
    // TODO: Flip to assert when we remove use cases in python and c++
    if (input_tensor.layout() != Layout::ROW_MAJOR) {
        log_warning(
            tt::LogOp,
            "Tensor layout {} must be ROW_MAJOR for padding! Returning original tensor!",
            input_tensor.layout());
        return input_tensor;
    }

    auto output =
        Tensor(tt::tt_metal::pad(input_tensor.host_tensor(), output_padded_shape, input_tensor_start, pad_value));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor unpad(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
    GraphTracker::instance().track_function_start(
        "Tensor::unpad", input_tensor, output_tensor_start, output_tensor_end);
    TT_FATAL(is_cpu_tensor(input_tensor), "Tensor must be on host for unpadding");
    auto output = Tensor(tt::tt_metal::unpad(input_tensor.host_tensor(), output_tensor_start, output_tensor_end));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor pad_to_tile(const Tensor& input_tensor, float pad_value) {
    // TODO: Flip to assert when we remove use cases in python and c++
    if (input_tensor.layout() != Layout::ROW_MAJOR) {
        log_warning(
            tt::LogOp,
            "Tensor layout {} must be ROW_MAJOR for padding! Returning original tensor!",
            input_tensor.layout());
        return input_tensor;
    }

    GraphTracker::instance().track_function_start("Tensor::pad_to_tile", input_tensor, pad_value);
    TT_FATAL(is_cpu_tensor(input_tensor), "Tensor must be on host for pad_to_tile conversion");
    auto output = Tensor(tt::tt_metal::pad_to_tile(input_tensor.host_tensor(), pad_value));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor unpad_from_tile(const Tensor& input_tensor, const tt::tt_metal::Shape& output_tensor_shape) {
    GraphTracker::instance().track_function_start("Tensor::unpad_from_tile", input_tensor, output_tensor_shape);
    TT_FATAL(is_cpu_tensor(input_tensor), "Tensor must be on host for unpad_from_tile conversion");
    auto output = Tensor(tt::tt_metal::unpad_from_tile(input_tensor.host_tensor(), output_tensor_shape));
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

// ======================================================================================
//                                  .tensor_view()
// ======================================================================================

Tensor view_device(const Tensor& input_tensor, const Shape& new_logical_shape, const Shape& new_padded_shape) {
    // Just edit shape if shape has a 0 dimension
    if (input_tensor.logical_volume() == 0) {
        TT_FATAL(new_logical_shape.volume() == 0, "Tensor volume is 0, but shape's volume is not");
    }
    const auto& input_memory_config = input_tensor.memory_config();
    auto output_memory_config = input_memory_config;
    bool is_row_major = input_tensor.layout() == Layout::ROW_MAJOR;
    bool changing_last_dim = false;

    if (input_memory_config.memory_layout() == TensorMemoryLayout::ND_SHARDED) {
        const auto old_rank = input_tensor.padded_shape().rank();
        const auto& old_nd_spec = input_memory_config.nd_shard_spec().value();

        // Rank-expansion of a 0D/1D tensor into a 2D shape — the original allow-listed
        // case. Requires the expanded 2D shape to still match the input's logical footprint.
        bool is_rank_expansion_to_2d = old_rank < 2 && new_padded_shape.rank() == 2 && new_padded_shape[0] == 1 &&
                                       (old_rank == 0 || new_padded_shape[1] == input_tensor.padded_shape()[-1]);

        // Logical-shape-only update: same rank and same padded shape as the input. No bytes
        // move, the physical tile layout is unchanged, and the per-core shard location and
        // size are unchanged. The caller is just reinterpreting which logical elements live
        // in which physical positions (e.g. a reduction op trimming logical dim after keeping
        // the padded tile intact). Safe for ND-sharded tensors.
        bool is_same_physical_shape =
            new_padded_shape.rank() == old_rank && new_padded_shape == input_tensor.padded_shape();

        TT_FATAL(
            is_rank_expansion_to_2d || is_same_physical_shape,
            "View is not supported for ND sharded tensors except for rank expansion to 2D "
            "or same-physical-shape (logical-only) metadata updates. Input shape: {}, New shape: {}",
            input_tensor.padded_shape(),
            new_padded_shape);

        if (is_same_physical_shape) {
            // Keep the input's MemoryConfig as-is (including nd_shard_spec + flags) — no
            // physical layout change means no metadata adjustment needed.
            output_memory_config = input_memory_config;
        } else {
            // Rank-expansion-to-2D path: synthesize a new nd_shard_spec for the expanded shape.
            ttsl::SmallVector<uint32_t> new_shard_shape =
                old_rank == 0 ? ttsl::SmallVector<uint32_t>{1, 1}
                              : ttsl::SmallVector<uint32_t>{1, old_nd_spec.shard_shape[-1]};
            output_memory_config =
                MemoryConfig(input_memory_config.buffer_type(), old_nd_spec.with_shard_shape(Shape(new_shard_shape)));
        }
    } else {
        changing_last_dim = new_padded_shape[-1] != input_tensor.padded_shape()[-1];
        TT_FATAL(
            !input_memory_config.is_sharded() || !changing_last_dim ||
                input_memory_config.shard_spec()->shape[1] == input_tensor.padded_shape()[-1],
            "Changing the last dimension of a sharded tensor is not supported unless the shard width matches the "
            "input last dimension. "
            "Input shape: {}, New shape: {}, Shard width: {}",
            input_tensor.padded_shape(),
            new_padded_shape,
            input_memory_config.shard_spec()->shape[1]);
        if (is_row_major && input_memory_config.is_sharded() && changing_last_dim) {
            auto shard_spec = input_memory_config.shard_spec().value();
            auto shard_volume = shard_spec.numel();
            shard_spec.shape[1] = new_padded_shape[-1];
            shard_spec.shape[0] = shard_volume / shard_spec.shape[1];
            output_memory_config =
                MemoryConfig{input_memory_config.memory_layout(), input_memory_config.buffer_type(), shard_spec};
        }
    }

    auto new_spec = tt::tt_metal::TensorSpec(
        new_logical_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            input_tensor.tensor_spec().page_config(),
            output_memory_config,
            new_logical_shape,
            new_padded_shape));

    // TODO (#25340): Review tensor topology logic for reshape
    if (input_tensor.layout() != Layout::ROW_MAJOR || !changing_last_dim) {
        const auto& input_buffer = input_tensor.device_storage().get_mesh_buffer();

        auto view_mesh_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            input_buffer.global_config(),
            input_buffer.device_local_config(),
            input_buffer.device(),
            input_buffer.address());

        MeshTensor view_mesh_tensor(std::move(view_mesh_buffer), new_spec, input_tensor.tensor_topology());
        DeviceStorage view_storage(input_tensor.device_storage(), std::move(view_mesh_tensor));
        return Tensor(std::move(view_storage));
    }
    if (!input_tensor.memory_config().is_sharded()) {
        const auto& input_buffer = input_tensor.device_storage().get_mesh_buffer();

        auto new_device_config = input_buffer.device_local_config();
        new_device_config.page_size = new_spec.compute_page_size_bytes();

        auto view_mesh_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            input_buffer.global_config(), new_device_config, input_buffer.device(), input_buffer.address());

        MeshTensor view_mesh_tensor(std::move(view_mesh_buffer), new_spec, input_tensor.tensor_topology());
        DeviceStorage view_storage(input_tensor.device_storage(), std::move(view_mesh_tensor));
        return Tensor(std::move(view_storage));
    }

    tt::tt_metal::ShardSpec new_shard_spec = output_memory_config.shard_spec().value();
    std::array<uint32_t, 2> shard_page_shape = {1, new_shard_spec.shape[1]};
    std::array<uint32_t, 2> tensor2d_shape_in_pages = {
        new_spec.physical_shape().height() / shard_page_shape[0],
        new_spec.physical_shape().width() / shard_page_shape[1]};
    tt::tt_metal::ShardSpecBuffer new_shard_spec_buffer =
        tt::tt_metal::ShardSpecBuffer(new_shard_spec, shard_page_shape, tensor2d_shape_in_pages);

    tt::tt_metal::Shape tensor_shape_pages(tensor2d_shape_in_pages);
    tt::tt_metal::Shape shard_shape_pages(new_shard_spec_buffer.shape_in_pages());
    tt::tt_metal::BufferDistributionSpec new_buffer_dist_spec = tt::tt_metal::BufferDistributionSpec(
        tensor_shape_pages, shard_shape_pages, new_shard_spec.grid, new_shard_spec.orientation);

    auto device_local_config = input_tensor.mesh_buffer().device_local_config();
    auto& sharding_args = device_local_config.sharding_args;
    tt::tt_metal::BufferShardingArgs new_sharding_args(
        new_buffer_dist_spec, new_shard_spec_buffer, sharding_args.buffer_layout());

    tt::tt_metal::distributed::DeviceLocalBufferConfig new_device_config = {
        .page_size = new_spec.compute_page_size_bytes(),
        .buffer_type = device_local_config.buffer_type,
        .sharding_args = new_sharding_args,
        .bottom_up = device_local_config.bottom_up};

    auto view_mesh_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        input_tensor.mesh_buffer().global_config(),
        new_device_config,
        input_tensor.device(),
        input_tensor.mesh_buffer().address());

    tt::tt_metal::DeviceStorage view_storage(
        input_tensor.device_storage(), MeshTensor(view_mesh_buffer, new_spec, input_tensor.tensor_topology()));
    return Tensor(std::move(view_storage));
}

Tensor view(const Tensor& input_tensor, const Shape& new_logical_shape, const Shape& new_padded_shape) {
    tt::tt_metal::GraphTracker::instance().track_function_start(
        "Tensor::reshape", input_tensor, new_logical_shape, new_padded_shape);

    Tensor output;
    if (is_cpu_tensor(input_tensor)) {
        output = Tensor(tensor_impl::view(input_tensor.host_tensor(), new_logical_shape, new_padded_shape));
    } else {
        output = view_device(input_tensor, new_logical_shape, new_padded_shape);
    }

    output = tt::tt_metal::set_tensor_id(output);
    tt::tt_metal::GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor view(const Tensor& input_tensor, const Shape& new_shape) { return view(input_tensor, new_shape, new_shape); }

Tensor unchecked_reinterpret_layout(const Tensor& input_tensor, Layout target_layout) {
    const auto& old_spec = input_tensor.tensor_spec();
    const auto& old_layout = old_spec.tensor_layout();

    TensorLayout new_tensor_layout(
        old_layout.get_data_type(), PageConfig(target_layout, old_layout.get_tile()), old_layout.get_memory_config());
    TensorSpec new_spec(old_spec.logical_shape(), new_tensor_layout);
    const auto& topology = input_tensor.tensor_topology();

    if (is_cpu_tensor(input_tensor)) {
        return Tensor(HostTensor(input_tensor.host_tensor().buffer(), new_spec, topology));
    }

    const auto& input_buffer = input_tensor.device_storage().get_mesh_buffer();
    auto new_mesh_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        input_buffer.global_config(),
        input_buffer.device_local_config(),
        input_buffer.device(),
        input_buffer.address());

    MeshTensor reinterpreted(std::move(new_mesh_buffer), new_spec, topology);
    DeviceStorage reinterpreted_storage(input_tensor.device_storage(), std::move(reinterpreted));
    return Tensor(std::move(reinterpreted_storage));
}

// ======================================================================================
//                                  .tensor_reshape()
// ======================================================================================
Tensor reshape(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape) {
    return view(input_tensor, new_logical_shape, new_padded_shape);
}

Tensor reshape(const Tensor& input_tensor, const tt::tt_metal::Shape& new_shape) {
    return reshape(input_tensor, new_shape, new_shape);
}

Tensor to_dtype(const Tensor& input_tensor, DataType dtype) {
    GraphTracker::instance().track_function_start("tt::tt_metal::to_dtype", input_tensor, dtype);
    auto output_tensor = Tensor(tt::tt_metal::to_dtype(input_tensor.host_tensor(), dtype));
    GraphTracker::instance().track_function_end(output_tensor);
    return output_tensor;
}

std::string to_string(const Tensor& tensor) { return tensor_impl::to_string(tensor); }

}  // namespace tt::tt_metal
