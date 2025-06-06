// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/all_reduce/device/all_reduce_op.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include <cstdint>

namespace ttnn {

void AllReduce::validate(const std::vector<Tensor>& input_tensors) const {
    for (auto const& t : input_tensors) {
        TT_FATAL(!t.is_sharded(), "Sharded tensors are not supported for all reduce currently");
    }
}

std::vector<ttnn::TensorSpec> AllReduce::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shape = input_tensor.logical_shape();
    TensorSpec spec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), output_mem_config));
    return std::vector<ttnn::TensorSpec>(input_tensors.size(), spec);
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks AllReduce::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
};

tt::tt_metal::operation::ProgramWithCallbacks AllReduce::create_program_at(
    const ttnn::MeshCoordinate& coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto target_device =
        input_tensors[0].mesh_device() ? input_tensors[0].mesh_device()->get_device(coord) : input_tensors[0].device();
    ttnn::ccl::SenderRecieverConfig config =
        ttnn::ccl::get_device_sender_receiver_config(target_device, this->devices, this->topology);

    return ccl::reduce_scatter_detail::reduce_scatter_with_workers(
        input_tensors.at(0),
        output_tensors.at(0),
        this->binary_op_type,
        0,
        this->num_links,
        this->ring_size,
        config.device_index,
        target_device->id(),
        config.receiver_device_id,
        config.sender_device_id,
        this->topology,
        this->user_defined_num_workers,
        this->user_defined_num_buffers_per_channel);
}

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
ttnn::operations::binary::BinaryOpType convert_reduce_type_to_eltwise_type(
    ttnn::operations::reduction::ReduceType reduce_op) {
    // Leaving switch statement for future support of additional types.
    switch (reduce_op) {
        case ttnn::operations::reduction::ReduceType::Sum: return ttnn::operations::binary::BinaryOpType::ADD;
        default:
            TT_THROW("All reduce only supports reduce_type Sum. Op type {} not supported.", reduce_op);
            return ttnn::operations::binary::BinaryOpType::ADD;
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace operations {
namespace experimental {
namespace ccl {

static AllReduceStrategy choose_all_reduce_strategy(
    const Tensor& input_tensor, uint32_t num_devices, uint32_t num_links, ttnn::ccl::Topology topology) {
    auto shape = input_tensor.logical_shape();
    auto rank = shape.rank();

    uint32_t all_reduce_dim = -1;
    bool optimized_version = false;

    if (num_devices == 2) {
        // 2 devices == n300 == linear topology
        topology = ttnn::ccl::Topology::Linear;
    }

    for (uint32_t i = 0; i < rank; ++i) {
        if (shape[i] % num_devices == 0) {
            all_reduce_dim = i;
            optimized_version = true;
        }
    }

    if (topology == ttnn::ccl::Topology::Linear) {
        // reduce scatter doesn't reliably support line topology yet
        optimized_version = false;
    }

    if (optimized_version) {
        if (shape[2] == tt::constants::TILE_HEIGHT || shape[3] == tt::constants::TILE_WIDTH) {
            optimized_version = false;  // Reduce scatter hangs for this shape
        }

        if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
            if ((all_reduce_dim == 2 && shape[all_reduce_dim] % tt::constants::TILE_HEIGHT != 0) ||
                (all_reduce_dim == 3 && shape[all_reduce_dim] % tt::constants::TILE_WIDTH != 0)) {
                optimized_version = false;
            }
        }
    }

    if (optimized_version) {
        return AllReduceStrategy::ReduceScatterAllGather;
    } else {
        return AllReduceStrategy::AllGatherLocalReduce;
    }

    return AllReduceStrategy::Invalid;
}

static Tensor all_gather_local_reduce(
    const Tensor& input_tensor,
    uint32_t num_devices,
    uint32_t num_links,
    const MemoryConfig& output_mem_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::vector<IDevice*>& devices,
    ttnn::ccl::Topology topology) {
    auto shape = input_tensor.logical_shape();
    auto rank = shape.rank();
    log_warning(
        tt::LogOp,
        "Falling back to unoptimized version (all_gather + local reduce) as the input tensor shape {} is not handled "
        "by optimized version",
        shape);
    if (num_devices == 2) {
        // 2 devices == n300 == linear topology
        topology = ttnn::ccl::Topology::Linear;
    }

    TT_FATAL(rank == 4, "Tensor rank must be 4, but has {} ", rank);
    uint32_t merged_dim_size = 1;
    for (uint32_t i = 2; i < rank; ++i) {
        merged_dim_size *= shape[i - 2];
    }

    std::vector<int32_t> new_shape{1, merged_dim_size, shape[rank - 2], shape[rank - 1]};
    auto reshaped_tensor = ttnn::reshape(input_tensor, new_shape);
    const auto& gathered_tensor = tt::tt_metal::operation::run(
        ttnn::AllGather{
            .dim = 0,
            .num_links = num_links,
            .ring_size = num_devices,
            .user_defined_num_workers = user_defined_num_workers,
            .user_defined_num_buffers_per_channel = user_defined_num_buffers_per_channel,
            .output_mem_config = output_mem_config,
            .topology = topology,
            .cluster_axis = std::nullopt,
            .devices = devices},
        {reshaped_tensor});

    auto sum_tensor = ttnn::sum(gathered_tensor.at(0), 0);
    return ttnn::reshape(sum_tensor, shape);
}

static std::vector<Tensor> all_gather_local_reduce(
    const std::vector<Tensor>& input_tensors,
    uint32_t num_devices,
    uint32_t num_links,
    const MemoryConfig& output_mem_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::vector<IDevice*>& devices,
    ttnn::ccl::Topology topology) {
    auto shape = input_tensors.at(0).logical_shape();
    auto rank = shape.rank();
    log_warning(
        tt::LogOp,
        "Falling back to unoptimized version (all_gather + local reduce) as the input tensor shape {} is not handled "
        "by optimized version",
        shape);
    if (num_devices == 2) {
        // 2 devices == n300 == linear topology
        topology = ttnn::ccl::Topology::Linear;
    }

    TT_FATAL(rank == 4, "Tensor rank must be 4, but has {} ", rank);
    uint32_t merged_dim_size = 1;
    for (uint32_t i = 2; i < rank; ++i) {
        merged_dim_size *= shape[i - 2];
    }

    std::vector<int32_t> new_shape{1, merged_dim_size, shape[rank - 2], shape[rank - 1]};
    std::vector<Tensor> reshaped_tensors;
    reshaped_tensors.reserve(input_tensors.size());
    for (const auto& input_tensor : input_tensors) {
        reshaped_tensors.push_back(ttnn::reshape(input_tensor, new_shape));
    }
    std::vector<Tensor> gathered_tensors;
    gathered_tensors.reserve(input_tensors.size());
    for (const auto& reshaped_tensor : reshaped_tensors) {
        const auto& gathered_tensor = tt::tt_metal::operation::run(
            ttnn::AllGather{
                .dim = 0,
                .num_links = num_links,
                .ring_size = num_devices,
                .user_defined_num_workers = user_defined_num_workers,
                .user_defined_num_buffers_per_channel = user_defined_num_buffers_per_channel,
                .output_mem_config = output_mem_config,
                .topology = topology,
                .cluster_axis = std::nullopt,
                .devices = devices},
            {reshaped_tensor});
        gathered_tensors.push_back(gathered_tensor.at(0));
    }
    std::vector<Tensor> reduced_tensors;
    reduced_tensors.reserve(input_tensors.size());
    for (const auto& gathered_tensor : gathered_tensors) {
        reduced_tensors.push_back(ttnn::sum(gathered_tensor, 0));
    }
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (const auto& reduced_tensor : reduced_tensors) {
        output_tensors.push_back(ttnn::reshape(reduced_tensor, shape));
    }
    return output_tensors;
}

static Tensor reduce_scatter_all_gather(
    const Tensor& input_tensor,
    const ttnn::operations::binary::BinaryOpType binary_op_type,
    uint32_t num_devices,
    uint32_t num_links,
    const MemoryConfig& output_mem_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology& topology) {
    auto shape = input_tensor.logical_shape();
    auto rank = shape.rank();

    uint32_t all_reduce_dim = -1;
    for (uint32_t i = 0; i < rank; ++i) {
        if (shape[i] % num_devices == 0) {
            all_reduce_dim = i;
        }
    }

    const auto& reduced_tensor = tt::tt_metal::operation::run(
        ttnn::ReduceScatter{
            .binary_op_type = binary_op_type,
            .scatter_dim = all_reduce_dim,
            .num_links = num_links,
            .ring_size = num_devices,
            .output_mem_config = output_mem_config,
            .topology = topology,
            .user_defined_num_workers = user_defined_num_workers,
            .user_defined_num_buffers_per_channel = user_defined_num_buffers_per_channel,
            .cluster_axis = std::nullopt,
            .devices = devices},
        {input_tensor});

    const auto& gathered_tensor = tt::tt_metal::operation::run(
        ttnn::AllGather{
            .dim = all_reduce_dim,
            .num_links = num_links,
            .ring_size = num_devices,
            .user_defined_num_workers = user_defined_num_workers,
            .user_defined_num_buffers_per_channel = user_defined_num_buffers_per_channel,
            .output_mem_config = output_mem_config,
            .topology = topology,
            .cluster_axis = std::nullopt,
            .devices = devices},
        {reduced_tensor.at(0)});

    return gathered_tensor.at(0);
}

static std::vector<Tensor> reduce_scatter_all_gather(
    const std::vector<Tensor>& input_tensors,
    const ttnn::operations::binary::BinaryOpType binary_op_type,
    uint32_t num_devices,
    uint32_t num_links,
    const MemoryConfig& output_mem_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology& topology) {
    auto shape = input_tensors.at(0).logical_shape();
    auto rank = shape.rank();

    uint32_t all_reduce_dim = -1;
    for (uint32_t i = 0; i < rank; ++i) {
        if (shape[i] % num_devices == 0) {
            all_reduce_dim = i;
        }
    }

    std::vector<Tensor> reduced_tensors;
    reduced_tensors.reserve(input_tensors.size());
    for (const auto& input_tensor : input_tensors) {
        const auto& reduced_tensor = tt::tt_metal::operation::run(
            ttnn::ReduceScatter{
                .binary_op_type = binary_op_type,
                .scatter_dim = all_reduce_dim,
                .num_links = num_links,
                .ring_size = num_devices,
                .output_mem_config = output_mem_config,
                .topology = topology,
                .user_defined_num_workers = user_defined_num_workers,
                .user_defined_num_buffers_per_channel = user_defined_num_buffers_per_channel,
                .cluster_axis = std::nullopt,
                .devices = devices},
            {input_tensor});
        reduced_tensors.push_back(reduced_tensor.at(0));
    }
    std::vector<Tensor> gathered_tensors;
    gathered_tensors.reserve(input_tensors.size());
    for (const auto& reduced_tensor : reduced_tensors) {
        const auto& gathered_tensor = tt::tt_metal::operation::run(
            ttnn::AllGather{
                .dim = all_reduce_dim,
                .num_links = num_links,
                .ring_size = num_devices,
                .user_defined_num_workers = user_defined_num_workers,
                .user_defined_num_buffers_per_channel = user_defined_num_buffers_per_channel,
                .output_mem_config = output_mem_config,
                .topology = topology,
                .cluster_axis = std::nullopt,
                .devices = devices},
            {reduced_tensor});
        gathered_tensors.push_back(gathered_tensor.at(0));
    }
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (const auto& gathered_tensor : gathered_tensors) {
        output_tensors.push_back(ttnn::reshape(gathered_tensor, shape));
    }
    return output_tensors;
}

Tensor run_all_reduce(
    AllReduceStrategy strategy,
    const Tensor& input_tensor,
    const ttnn::operations::binary::BinaryOpType binary_op_type,
    uint32_t num_devices,
    uint32_t num_links,
    const MemoryConfig& output_mem_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology& topology) {
    switch (strategy) {
        case AllReduceStrategy::AllGatherLocalReduce:
            return all_gather_local_reduce(
                input_tensor,
                num_devices,
                num_links,
                output_mem_config,
                user_defined_num_workers,
                user_defined_num_buffers_per_channel,
                devices,
                topology);
        case AllReduceStrategy::ReduceScatterAllGather:
            return reduce_scatter_all_gather(
                input_tensor,
                binary_op_type,
                num_devices,
                num_links,
                output_mem_config,
                user_defined_num_workers,
                user_defined_num_buffers_per_channel,
                devices,
                topology);
        case AllReduceStrategy::Invalid:
        default:
            TT_FATAL(
                false,
                "Invalid strategy selected {} for input tensor shape: {}",
                strategy,
                input_tensor.logical_shape());
    }
}

std::vector<Tensor> run_all_reduce(
    AllReduceStrategy strategy,
    const std::vector<Tensor>& input_tensors,
    const ttnn::operations::binary::BinaryOpType binary_op_type,
    uint32_t num_devices,
    uint32_t num_links,
    const MemoryConfig& output_mem_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology& topology) {
    switch (strategy) {
        case AllReduceStrategy::AllGatherLocalReduce:
            return all_gather_local_reduce(
                input_tensors,
                num_devices,
                num_links,
                output_mem_config,
                user_defined_num_workers,
                user_defined_num_buffers_per_channel,
                devices,
                topology);
        case AllReduceStrategy::ReduceScatterAllGather:
            return reduce_scatter_all_gather(
                input_tensors,
                binary_op_type,
                num_devices,
                num_links,
                output_mem_config,
                user_defined_num_workers,
                user_defined_num_buffers_per_channel,
                devices,
                topology);
        case AllReduceStrategy::Invalid:
        default:
            TT_FATAL(
                false,
                "Invalid strategy selected {} for input tensor shape: {}",
                strategy,
                input_tensors.at(0).logical_shape());
    }
}

Tensor all_reduce(
    const Tensor& input_tensor,
    ttnn::operations::reduction::ReduceType math_op,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(math_op);
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "All Reduce op is only supported for Fast Dispatch");

    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "all_reduce op will only work for num_devices > 1, but has {}", num_devices);

    bool is_linear = topology == ttnn::ccl::Topology::Linear;

    // Choose the appropriate strategy
    AllReduceStrategy strategy = choose_all_reduce_strategy(input_tensor, num_devices, num_links, topology);

    // Run the selected all-reduce operation
    return run_all_reduce(
        strategy,
        input_tensor,
        binary_op_type,
        num_devices,
        num_links,
        output_mem_config,
        user_defined_num_workers,
        user_defined_num_buffers_per_channel,
        devices,
        topology);
}

std::vector<Tensor> all_reduce(
    const std::vector<Tensor>& input_tensors,
    ttnn::operations::reduction::ReduceType math_op,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(math_op);
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "All Reduce op is only supported for Fast Dispatch");

    std::vector<IDevice*> devices;
    devices.reserve(input_tensors.size());
    for (const auto& input_tensor : input_tensors) {
        devices.push_back(input_tensor.device());
    }
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "all_reduce op will only work for num_devices > 1, but has {}", num_devices);

    bool is_linear = topology == ttnn::ccl::Topology::Linear;

    // Choose the appropriate strategy
    AllReduceStrategy strategy = choose_all_reduce_strategy(input_tensors.at(0), num_devices, num_links, topology);

    // Run the selected all-reduce operation
    return run_all_reduce(
        strategy,
        input_tensors,
        binary_op_type,
        num_devices,
        num_links,
        output_mem_config,
        user_defined_num_workers,
        user_defined_num_buffers_per_channel,
        devices,
        topology);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

};  // namespace ttnn
