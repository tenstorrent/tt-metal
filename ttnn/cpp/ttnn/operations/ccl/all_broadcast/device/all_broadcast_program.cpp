// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/ccl/all_broadcast/device/all_broadcast_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

using namespace tt::constants;

namespace ttnn {

using namespace ccl;

tt::tt_metal::operation::ProgramWithCallbacks all_broadcast_multicore(
    const Tensor& input_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    std::vector<Tensor>& output_tensors,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    tt::tt_metal::Program program{};

    auto mesh_device = input_tensor.device();

    // Get OP Config, topology config
    auto [num_targets_forward, num_targets_backward] =
        ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, true);

    // Get worker cores, assuming 1 worker per link
    CoreCoord core = {0, 0};
    CoreRange sender_worker_core_range = {core};
    // KERNEL CREATION

    std::vector<uint32_t> writer_compile_args = {
        num_targets_forward,   // num_targets_forward_direction
        num_targets_backward,  // num_targets_backward_direction
    };

    std::vector<uint32_t> mcast_forward_args(2, 0);
    std::vector<uint32_t> mcast_backward_args(2, 0);
    if (forward_coord.has_value()) {
        mcast_forward_args[0] = 1;
        mcast_forward_args[1] = num_targets_forward;
    }
    if (backward_coord.has_value()) {
        mcast_backward_args[0] = 1;
        mcast_backward_args[1] = num_targets_backward;
    }
    writer_compile_args.insert(writer_compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
    writer_compile_args.insert(writer_compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());
    // This is the same in 1D and 2D fabric and verified to be correct
    std::cout << "Writer compile args are:";
    for (std::size_t i = 0; i < writer_compile_args.size(); ++i) {
        std::cout << writer_compile_args[i] << (i + 1 < writer_compile_args.size() ? ", " : "\n");
    }
    std::map<std::string, std::string> kernel_defines;

    // Writer
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_tile_writer.cpp",
        sender_worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args, kernel_defines));

    // Kernel Runtime Args
    CoreCoord barrier_core;

    barrier_core = mesh_device->worker_core_from_logical_core(core);

    // Set writer runtime args
    std::vector<uint32_t> writer_rt_args = {
        barrier_semaphore.address(),  // barrier_sem
        barrier_core.x,               // barrier_sem_noc0_x
        barrier_core.y                // barrier_sem_noc0_y
    };
    auto num_connections = (int)forward_coord.has_value() + (int)backward_coord.has_value();
    writer_rt_args.push_back(num_connections);
    const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
    std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
    dst_nodes.reserve(num_connections);
    if (forward_coord.has_value()) {
        const auto forward_coord_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
        dst_nodes.push_back(forward_coord_fabric_node_id);
    }
    if (backward_coord.has_value()) {
        const auto backward_coord_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
        dst_nodes.push_back(backward_coord_fabric_node_id);
    }

    append_routing_plane_connection_manager_rt_args(
        sender_fabric_node_id, dst_nodes, {0}, program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
    tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);

    return {.program = std::move(program)};
}

}  // namespace ttnn
