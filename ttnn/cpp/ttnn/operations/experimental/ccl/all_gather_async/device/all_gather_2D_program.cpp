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
    const tt::tt_metal::distributed::MeshCoordinate& device_coord,
    const size_t num_rows,
    const size_t num_cols,
    bool is_horizontal,
    uint32_t& eastCount,
    uint32_t& westCount,
    uint32_t& upCount,
    uint32_t& downCount,
    uint32_t& offset) {
    // Calculate East (right direction)
    if (device_coord.coords()[1] < num_cols - 1) {
        eastCount = num_cols - device_coord.coords()[1] - 1;
    }

    // Calculate West (left direction)
    if (device_coord.coords()[1] > 0) {
        westCount = device_coord.coords()[1];
    }

    // Calculate Up (above direction)
    if (device_coord.coords()[0] > 0) {
        upCount = device_coord.coords()[0];
    }

    // Calculate Down (below direction)
    if (device_coord.coords()[0] < num_rows - 1) {
        downCount = num_rows - device_coord.coords()[0] - 1;  // Devices below
    }

    if (is_horizontal) {
        offset = device_coord.coords()[0];
    } else {
        offset = device_coord.coords()[1];
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
    tt::tt_metal::distributed::MeshCoordinate device_coord,
    CoreCoord grid_size,
    size_t num_rows,
    size_t num_cols,
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

    uint32_t eastCount, westCount, upCount, downCount, device_offset = 0;
    //I might not need this
    getNeighborsCountAndOffset(
        device_coord, num_rows, num_cols, is_horizontal, eastCount, westCount, upCount, downCount, device_offset);

    // bool is_first_chip = ring_index == 0;
    // bool is_last_chip = ring_index == ring_size - 1;

    tt::tt_fabric::RoutingDirection direction0;
    tt::tt_fabric::RoutingDirection direction1;
    if(is_horizontal) {
        direction0 = tt::tt_fabric::RoutingDirection::E;
        direction1 = tt::tt_fabric::RoutingDirection::W;
    }
    else{
        direction0 = tt::tt_fabric::RoutingDirection::N;
        direction1 = tt::tt_fabric::RoutingDirection::S;
    }

    tt::tt_fabric::ControlPlane* control_plane = tt::DevicePool::instance().get_control_plane();
    std::pair<tt::tt_fabric::mesh_id_t, chip_id_t> device_mesh_chip_id =
        control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());

    auto neighbors_dir0 =
        control_plane->get_intra_chip_neighbors(device_mesh_chip_id.first, device_mesh_chip_id.second, direction0);
    auto neighbors_dir1 =
        control_plane->get_intra_chip_neighbors(device_mesh_chip_id.first, device_mesh_chip_id.second, direction1);

    bool first_device = neighbors_dir1.size() == 0;
    bool last_device = neighbors_dir0.size() == 0;

    chip_id_t chip_id_dir0 =0;
    chip_id_t chip_id_dir1 =0;
    std::pair<tt::tt_fabric::mesh_id_t, chip_id_t> mesh_chip_ids_dir0;
    std::pair<tt::tt_fabric::mesh_id_t, chip_id_t> mesh_chip_ids_dir1;
    tt::tt_fabric::mesh_id_t mesh_id_dir0 =0;
    tt::tt_fabric::mesh_id_t mesh_id_dir1 =0;

    if (!last_device) {
        chip_id_dir0 = neighbors_dir0[0];
        mesh_chip_ids_dir0 = control_plane->get_mesh_chip_id_from_physical_chip_id(chip_id_dir0);
        mesh_id_dir0 = mesh_chip_ids_dir0.first;
    }
    if (!first_device) {
        chip_id_dir1 = neighbors_dir1[0];
        mesh_chip_ids_dir1 = control_plane->get_mesh_chip_id_from_physical_chip_id(chip_id_dir1);
        mesh_id_dir1 = mesh_chip_ids_dir1.first;
    }

    uint32_t depth_dir0 = neighbors_dir0.size();
    uint32_t depth_dir1 = neighbors_dir1.size();

    std::vector<std::pair<tt::tt_fabric::routing_plane_id_t, CoreCoord>> routers_dir0 = control_plane->get_routers_to_chip(
        device_mesh_chip_id.first, device_mesh_chip_id.second, mesh_id_dir0, chip_id_dir0);
    std::vector<std::pair<tt::tt_fabric::routing_plane_id_t, CoreCoord>> routers_dir1 = control_plane->get_routers_to_chip(
        device_mesh_chip_id.first, device_mesh_chip_id.second, mesh_id_dir1, chip_id_dir1);

    auto& noc_xy_dir0 = routers_dir0[0].second;
    auto& noc_xy_dir1 = routers_dir1[0].second;
    auto router_noc_xy_dir0 = tt::tt_metal::hal.noc_xy_encoding(noc_xy_dir0.x, noc_xy_dir0.y);
    auto router_noc_xy_dir1 = tt::tt_metal::hal.noc_xy_encoding(noc_xy_dir1.x, noc_xy_dir1.y);


    CoreRange core({0, 0}, {0, 0});
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    uint32_t input_data_size = input_single_tile_size;

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);
    uint32_t output_data_size = output_single_tile_size;

    //uint32_t data_size = page_size; //assuming it is tt::constants::TILE_HW * sizeof(uint32_t);
    //const size_t packet_size_bytes = tt::tt_fabric::PACKET_HEADER_SIZE_BYTES + data_size;
    uint32_t cb_num_pages = lower_pages * higher_pages;

    uint32_t input_cb_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * page_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, page_size);
    CBHandle cb_worker = CreateCircularBuffer(program, core, cb_input_config);

    // Allocate space for the client interface
    uint32_t num_directions = 2;
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_directions * tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(program, core, client_interface_cb_config);

    Buffer* src0_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");


    uint32_t src_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> bd_reader_compile_time_args = {src_is_dram, input_cb_index, lower_pages, higher_pages};
    auto bd_reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/broadcast_kernel_2d_reader.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = bd_reader_compile_time_args});

    uint32_t dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> bd_writer_compile_time_args = {client_interface_cb_index, is_horizontal, dst_is_dram, input_cb_index, num_devices, device_id, output_tensor.element_size()};
    auto bd_writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/broadcast_kernel_2d.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = bd_writer_compile_time_args});

    std::vector<uint32_t> bd_reader_runtime_args = {
        src0_buffer->address(), //src_addr
        page_size * input_tensor.element_size() //num_bytes
    };
    std::vector<uint32_t> bd_writer_runtime_args = {
        src0_buffer->address(), //src_addr
        dst_buffer->address(), //dst_addr
        lower_pages,
        higher_pages,
        page_size * input_tensor.element_size(), //num_bytes
        page_size,
        mesh_id_dir0,
        chip_id_dir0,
        depth_dir0,
        router_noc_xy_dir0,
        mesh_id_dir1,
        chip_id_dir1,
        depth_dir1,
        router_noc_xy_dir1,
        first_device,
        last_device
    };
    tt::tt_metal::SetRuntimeArgs(program, bd_reader_kernel, core, bd_reader_runtime_args);
    tt::tt_metal::SetRuntimeArgs(program, bd_writer_kernel, core, bd_writer_runtime_args);

    return {.program = std::move(program)};
}

}  // namespace ttnn
