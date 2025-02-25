// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include <tt-metalium/mesh_graph.hpp>

#include "cpp/ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "cpp/ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

// #include "fabric/hw/inc/tt_fabric_interface.h"

#include <tt-metalium/mesh_graph.hpp>

#include <tt-metalium/mesh_device.hpp>

#include <tt-metalium/mesh_device_view.hpp>

using namespace tt::constants;

namespace ttnn {

void getNeighborsCountAndOffset(
    const tt::tt_metal::distributed::Coordinate& device_coord,
    const size_t num_rows,
    const size_t num_cols,
    bool row_order,
    uint32_t& eastCount,
    uint32_t& westCount,
    uint32_t& upCount,
    uint32_t& downCount,
    uint32_t& offset) {
    // Calculate East (right direction)
    if (device_coord.col < num_cols - 1) {
        eastCount = num_cols - device_coord.col - 1;
    }

    // Calculate West (left direction)
    if (device_coord.col > 0) {
        westCount = device_coord.col;
    }

    // Calculate Up (above direction)
    if (device_coord.row > 0) {
        upCount = device_coord.row;
    }

    // Calculate Down (below direction)
    if (device_coord.row < num_rows - 1) {
        downCount = num_rows - device_coord.row - 1;  // Devices below
    }

    if (row_order) {  // Row-major order
        offset = device_coord.row * num_cols + device_coord.col;
    } else {  // Column-major order
        offset = device_coord.col * num_rows + device_coord.row;
    }
}

// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)

/*
E devices and we are on device e

E num_devices
HP = higher_pages = a1*a2*a3
LP = lower_pages
PS = page_size

(a1,a2,a3,a4,b,c,d) -> AG(E) -> (a,e*b,c,d)

Lower_Pages: Tile: b*ceil(c/32) * ceil(d/32), Row Major b*c

for i in range (a):
    for k in range (Lower_Pages):
        read_from_shard (i*LP+k)
        write_to_tensor (i*E*LP+k+e*LP)

*/

operation::ProgramWithCallbacks all_gather_2D_multi_core_with_workers(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const MeshDevice& mesh_device,
    const MemoryConfig output_mem_config,
    const ccl::Topology topology,
    const GlobalSemaphore semaphore,
    const uint32_t lower_pages,
    const uint32_t higher_pages,
    const uint32_t num_devices,
    const uint32_t page_size,
    bool is_horizontal) {
    tt::tt_metal::Program program{};

    IDevice* device = input_tensor.device();
    auto device_id = device->id();

    auto mesh_device_view = mesh_device.get_view();
    tt::tt_metal::distributed::Coordinate device_coord = mesh_device_view.find_device(device_id);
    auto grid_size = mesh_device.shape();
    auto num_rows = mesh_device.num_rows();
    auto num_cols = mesh_device.num_cols();

    uint32_t eastCount, westCount, upCount, downCount, device_offset = 0;
    getNeighborsCountAndOffset(
        device_coord, num_rows, num_cols, row_order, eastCount, westCount, upCount, downCount, device_offset);

    // bool is_first_chip = ring_index == 0;
    // bool is_last_chip = ring_index == ring_size - 1;

    tt::tt_fabric::RoutingDirection direction0 = tt::tt_fabric::RoutingDirection::E;
    tt::tt_fabric::RoutingDirection direction1 = tt::tt_fabric::RoutingDirection::W;
    tt::tt_fabric::RoutingDirection direction2 = tt::tt_fabric::RoutingDirection::N;
    tt::tt_fabric::RoutingDirection direction3 = tt::tt_fabric::RoutingDirection::S;

    tt::tt_fabric::ControlPlane* control_plane = tt::DevicePool::instance().get_control_plane();
    std::pair<tt::tt_fabric::mesh_id_t, chip_id_t> device_mesh_chip_id =
        control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());

    auto neighbors_dir0 =
        control_plane->get_intra_chip_neighbors(device_mesh_chip_id.first, device_mesh_chip_id.second, direction0);
    auto neighbors_dir1 =
        control_plane->get_intra_chip_neighbors(device_mesh_chip_id.first, device_mesh_chip_id.second, direction1);
    auto neighbors_dir2 =
        control_plane->get_intra_chip_neighbors(device_mesh_chip_id.first, device_mesh_chip_id.second, direction2);
    auto neighbors_dir3 =
        control_plane->get_intra_chip_neighbors(device_mesh_chip_id.first, device_mesh_chip_id.second, direction3);
    return {.program = std::move(program)};
}

}  // namespace ttnn
