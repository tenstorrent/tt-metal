// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/deepseek_minimal_all_reduce_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/deepseek_minimal_all_reduce_program_factory.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

namespace ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce::program {

DeepseekMinimalAllReduceProgramFactory::cached_mesh_workload_t
DeepseekMinimalAllReduceProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    ttnn::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    auto init_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    auto final_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = DeepseekMinimalAllReduceProgramFactory::create_at(
            operation_attributes,
            coord,
            tensor_args,
            tensor_return_value,
            final_barrier_semaphore,
            init_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

DeepseekMinimalAllReduceProgramFactory::cached_program_t DeepseekMinimalAllReduceProgramFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const MeshCoordinate& self_coord,
    const tensor_args_t& tensor_args,
    Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore) {
    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::Program program{};

    uint32_t num_links = operation_attributes.num_links;
    uint32_t ring_size = operation_attributes.ring_size;
    uint32_t ring_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, self_coord, operation_attributes.cluster_axis);
    auto topology = operation_attributes.topology;
    auto cluster_axis = operation_attributes.cluster_axis;

    std::optional<MeshCoordinate> forward_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, self_coord, 1, topology, cluster_axis);

    std::optional<MeshCoordinate> backward_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, self_coord, -1, topology, cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    auto* mesh_device = input_tensor.device();
    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        self_coord,
        is_first_chip,
        is_last_chip);

    // Fatal if input is not tilized with tiny tiles (1, 32)
    bool tilized = input_tensor.layout() == ttnn::TILE_LAYOUT;
    const auto tile_width = input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = input_tensor.tensor_spec().tile().get_height();
    TT_FATAL(
        tilized,
        "all_reduce_batch1 op currently only supports TILE_LAYOUT input tensors. Got layout: {}",
        input_tensor.layout());
    TT_FATAL(
        tile_width == 32 && tile_height == 1,
        "all_reduce_batch1 op currently only supports TILE_LAYOUT input tensors with tile size (1, 32). Got tile size: "
        "({}, {})",
        tile_height,
        tile_width);

    // Extract shard grid from input tensor
    TT_FATAL(input_tensor.is_sharded(), "Input tensor must be sharded");
    const auto& shard_spec = input_tensor.shard_spec().value();
    const auto& shard_grid = shard_spec.grid;

    // Get all cores from the shard grid
    std::vector<CoreCoord> cores;
    for (const auto& core_range : shard_grid.ranges()) {
        auto c = corerange_to_cores(core_range, std::nullopt);
        cores.insert(cores.end(), c.begin(), c.end());
    }
    // data should be on a single core: Fatal if not
    TT_FATAL(cores.size() == 1, "Input tensor must be sharded to a single core");

    // need two worker cores for 2 links
    // each link for a direction
    // link 0: device 0 sends to device 1
    // link 1: device 1 sends to device 0
    std::vector<CoreCoord> sender_worker_cores;
    sender_worker_cores.push_back(cores[0]);
    if (num_links == 2) {
        CoreCoord next_core;
        if (cores[0].x > 0) {
            next_core = CoreCoord{cores[0].x - 1, cores[0].y};
        } else {
            next_core = CoreCoord{cores[0].x + 1, cores[0].y};
        }
        sender_worker_cores.push_back(next_core);
    }
    auto sender_worker_core_range = CoreRangeSet(sender_worker_cores);

    // Get OP Config, topology config
    auto [num_targets_forward, num_targets_backward] =
        ::ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, true);

    TT_FATAL(
        num_targets_backward + num_targets_forward == 1,
        "Send to a single neighbour device only. Got num_targets_forward: {}, num_targets_backward: {}",
        num_targets_forward,
        num_targets_backward);

    // Tensor Info - get num_pages early for CB config
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();

    const auto tiny_tile = tt::tt_metal::Tile({1, 32});

    // L1 Scratch CB Creation
    const uint32_t input_page_size_bytes = input_tensor.buffer()->aligned_page_size();
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const size_t packet_size_bytes = input_page_size_bytes * input_tensor_num_pages;
    printf("packet_size_bytes: %u\n", packet_size_bytes);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    // CB page size should be individual tile size, not the entire packet
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(input_tensor_num_pages * input_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, input_page_size_bytes)
            .set_tile_dims(src0_cb_index, tiny_tile);
    CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // two buffers for reduction inputs and one for output
    constexpr auto compute_cb_in1 = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig compute_cb_in1_config =
        tt::tt_metal::CircularBufferConfig(input_tensor_num_pages * input_page_size_bytes, {{compute_cb_in1, df}})
            .set_page_size(compute_cb_in1, input_page_size_bytes)
            .set_tile_dims(compute_cb_in1, tiny_tile);
    CreateCircularBuffer(program, sender_worker_core_range, compute_cb_in1_config);

    constexpr auto compute_cb_in2 = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig compute_cb_in2_config =
        tt::tt_metal::CircularBufferConfig(input_tensor_num_pages * input_page_size_bytes, {{compute_cb_in2, df}})
            .set_page_size(compute_cb_in2, input_page_size_bytes)
            .set_tile_dims(compute_cb_in2, tiny_tile);
    CreateCircularBuffer(program, sender_worker_core_range, compute_cb_in2_config);

    constexpr auto compute_cb_out = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig compute_cb_out_config =
        tt::tt_metal::CircularBufferConfig(input_tensor_num_pages * input_page_size_bytes, {{compute_cb_out, df}})
            .set_page_size(compute_cb_out, input_page_size_bytes)
            .set_tile_dims(compute_cb_out, tiny_tile);
    CreateCircularBuffer(program, sender_worker_core_range, compute_cb_out_config);

    constexpr auto packet_header_cb_id = tt::CBIndex::c_4;
    constexpr auto buffering_factor = 2;
    constexpr auto num_packet_headers_storable = 2;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id, packet_header_size_bytes)
            .set_tile_dims(packet_header_cb_id, tiny_tile);
    CreateCircularBuffer(program, sender_worker_core_range, cb_header_config);

    constexpr auto packet_cb_id = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig cb_packet_config =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id, df}})
            .set_page_size(packet_cb_id, packet_size_bytes)
            .set_tile_dims(packet_cb_id, tiny_tile);
    CreateCircularBuffer(program, sender_worker_core_range, cb_packet_config);

    // Sender info
    bool is_sender = false;
    auto data_core_coord = mesh_device->worker_core_from_logical_core(sender_worker_cores[0]);
    auto core_noc_x = data_core_coord.x;
    auto core_noc_y = data_core_coord.y;

    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> sender_reader_compile_args = {
        src0_cb_index,           // cb0_id
        input_tensor_num_pages,  // num_tiles
        input_page_size_bytes,   // tensor0_page_size
        core_noc_x,
        core_noc_y};

    std::vector<uint32_t> sender_writer_compile_args = {
        packet_header_cb_id,
        src0_cb_index,
        l1_alignment,
        input_tensor_num_pages,
        input_page_size_bytes,
        packet_size_bytes,
        core_noc_x,
        core_noc_y,
    };

    std::vector<uint32_t> receiver_reader_compile_args = {
        packet_header_cb_id,
        compute_cb_in1,
        l1_alignment,
        compute_cb_in2,
        input_tensor_num_pages,
        input_page_size_bytes,
        packet_size_bytes,
        core_noc_x,
        core_noc_y,
    };

    std::vector<uint32_t> receiver_writer_compile_args = {
        compute_cb_out,
        input_tensor_num_pages,
        input_page_size_bytes,
        core_noc_x,
        core_noc_y,
    };

    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/kernels/sender_reader.cpp",
        {sender_worker_cores[0]},
        tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args));

    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/kernels/sender_writer.cpp",
        {sender_worker_cores[0]},
        tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args));

    auto worker_receiver_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/kernels/receiver_reader.cpp",
        {sender_worker_cores[1]},
        tt::tt_metal::ReaderDataMovementConfig(receiver_reader_compile_args));

    auto worker_receiver_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/kernels/receiver_writer.cpp",
        {sender_worker_cores[1]},
        tt::tt_metal::WriterDataMovementConfig(receiver_writer_compile_args));

    auto compute_kernel_configuration = ttnn::init_device_compute_kernel_config(
        input_tensor.device()->arch(), std::nullopt, MathFidelity::HiFi4, true, false, false);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input_tensor.device()->arch(), compute_kernel_configuration);

    compute_ct_args = {compute_cb_in1, compute_cb_in2, compute_cb_out, 1, input_tensor_num_pages};
    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/kernels/reduction.cpp",
        {sender_worker_cores[1]},
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = true,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_ct_args,
        });

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    CoreCoord barrier_core;
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }

        if (link == 0) {
            std::vector<uint32_t> reader_rt_args = {
                input_tensor.buffer()->address(),  // tensor_address0
            };
            tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

            std::vector<uint32_t> writer_rt_args = {
                intermediate_tensor.buffer()->address(),
                semaphore1.address(),   // barrier_sem
                is_first_chip ? 1 : 0,  // dst is forward
            };

            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(self_coord);
            const auto dst_fabric_node_id = is_first_chip ? mesh_device->get_fabric_node_id(forward_coord.value())
                                                          : mesh_device->get_fabric_node_id(backward_coord.value());
            append_routing_plane_connection_manager_rt_args(
                sender_fabric_node_id,
                dst_fabric_node_id,
                {link},
                program,
                worker_sender_writer_kernel_id,
                {core},
                writer_rt_args);
            tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
        } else {
            // Set reader runtime args
            std::vector<uint32_t> reader_rt_args = {
                input_buffer()->address(),                // tensor_address0
                intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                semaphore2.address(),
                is_first_chip ? 0 : 1,  // src is forward
            };
            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(self_coord);
            const auto dst_fabric_node_id = is_first_chip ? mesh_device->get_fabric_node_id(forward_coord.value())
                                                          : mesh_device->get_fabric_node_id(backward_coord.value());
            append_routing_plane_connection_manager_rt_args(
                sender_fabric_node_id,
                dst_fabric_node_id,
                {link},
                program,
                worker_receiver_reader_kernel_id,
                {core},
                reader_rt_args);
            tt::tt_metal::SetRuntimeArgs(program, worker_receiver_reader_kernel_id, {core}, reader_rt_args);

            // Set writer runtime args
            std::vector<uint32_t> writer_rt_args = {output_tensor.buffer()->address()};
            tt::tt_metal::SetRuntimeArgs(program, worker_receiver_writer_kernel_id, {core}, writer_rt_args);
        }
    }

    shared_variables_t shared_variables{
        .sender_worker_cores = sender_worker_cores,
        .receiver_worker_cores = {},
        .worker_sender_reader_kernel_id = worker_sender_reader_kernel_id,
        .worker_sender_writer_kernel_id = worker_sender_writer_kernel_id,
        .worker_receiver_reader_kernel_id = worker_receiver_reader_kernel_id,
        .worker_receiver_writer_kernel_id = worker_receiver_writer_kernel_id,
        .semaphore1 = semaphore1,
        .semaphore2 = semaphore2,
        .ring_index = ring_index,
    };
    return {std::move(program), std::move(shared_variables)};
}

void DeepseekMinimalAllReduceProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = tensor_return_value;

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        // const auto& coord = coordinate_range.start_coord();
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        log_trace(tt::LogOp, "DEBUG: semaphore: {}", shared_vars.semaphore.address());
        log_trace(tt::LogOp, "DEBUG: barrier_semaphore: {}", shared_vars.barrier_semaphore.address());
        // update senders
        auto& worker_reader_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_reader_kernel_id);
        auto& worker_writer_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_writer_kernel_id);

        for (const auto& core : shared_vars.sender_worker_cores) {
            // reader
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args[0] = input.buffer()->address();

            // writer
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args[0] = output.buffer()->address();
            worker_writer_sender_runtime_args[1] = shared_vars.semaphore.address();
            worker_writer_sender_runtime_args[9] = shared_vars.barrier_semaphore.address();
        }
    }
}
}  // namespace ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce::program
