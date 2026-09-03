// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// broadcast_ring program factory: per-device program for a one-sender, bidirectional, pipelined ring relay
// on FABRIC_1D_RING. Fabric routes/connections mirror ring_attention_all_gather_writer.cpp.

#include "ttnn/operations/experimental/ccl/broadcast_ring/device/broadcast_ring_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <numeric>

namespace ttnn::prim {

BroadcastRingProgramFactory::cached_mesh_workload_t BroadcastRingProgramFactory::create_mesh_workload(
    const BroadcastRingParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const BroadcastRingInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // Per-chunk recv credits (upstream increments, this device waits) + a readiness barrier. Global
    // semaphores so a neighbour on another device can atomic-inc them over the fabric. The L1-relay path
    // needs two more (cred_fwd/cred_bwd): the backward slot-free credit for its bounded L1 recv buffer.
    auto recv_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    auto barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    std::optional<tt::tt_metal::GlobalSemaphore> cred_fwd_semaphore, cred_bwd_semaphore;
    if (operation_attributes.use_l1_relay) {
        cred_fwd_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
        cred_bwd_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    }
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program =
            operation_attributes.use_l1_relay
                ? create_at_l1(
                      operation_attributes,
                      coord,
                      tensor_args,
                      tensor_return_value,
                      recv_semaphore,
                      cred_fwd_semaphore.value(),
                      cred_bwd_semaphore.value())
                : create_at(
                      operation_attributes, coord, tensor_args, tensor_return_value, recv_semaphore, barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

BroadcastRingProgramFactory::cached_program_t BroadcastRingProgramFactory::create_at(
    const BroadcastRingParams& operation_attributes,
    const ttnn::MeshCoordinate& coord,
    const BroadcastRingInputs& tensor_args,
    Tensor& tensor_return_value,
    const tt::tt_metal::GlobalSemaphore& recv_semaphore,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;
    tt::tt_metal::Program program{};
    auto* mesh_device = input_tensor.device();

    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t ring_index =
        ::ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, coord, operation_attributes.cluster_axis);

    // Forward neighbour along the ring axis (Ring topology wraps last -> first, giving one-way coverage).
    std::optional<MeshCoordinate> forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, coord, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    std::optional<MeshCoordinate> backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, coord, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value(), "broadcast_ring needs a forward neighbour (Ring topology)");

    // 1-hop line-unicast route args to the forward (idx+1) and backward (idx-1) neighbours.
    auto [unicast_forward_args, unicast_backward_args] =
        ::ttnn::ccl::get_forward_backward_line_unicast_configuration(coord, forward_coord, backward_coord, mesh_device);

    // Bidirectional role (mirrors broadcast_ring_relay.cpp): sender sends both ways; the ring splits into a
    // forward arc (HF hops) and a backward arc (HB hops). Each non-sender relays away from the sender until
    // its arc's far end. send_fwd -> forwards to idx+1; send_bwd -> forwards to idx-1.
    const uint32_t fwd_hops = (ring_index + ring_size - operation_attributes.sender_ring_index) % ring_size;
    const uint32_t bwd_hops = (ring_size - fwd_hops) % ring_size;
    const uint32_t HF = ring_size / 2;
    const uint32_t HB = (ring_size - 1) / 2;
    const bool is_sender = (fwd_hops == 0);
    const bool on_fwd_arc = !is_sender && (fwd_hops <= HF);
    const bool on_bwd_arc = !is_sender && !on_fwd_arc;
    const bool send_fwd = is_sender || (on_fwd_arc && fwd_hops < HF);
    const bool send_bwd = is_sender || (on_bwd_arc && bwd_hops < HB);

    // Worker cores (1 per link).
    const uint32_t num_workers_per_link = 1;
    const auto [worker_core_range, worker_cores] = ::ttnn::ccl::choose_worker_cores(
        operation_attributes.num_links,
        num_workers_per_link,
        mesh_device,
        operation_attributes.sub_device_id,
        CoreCoord(0, 0),
        std::nullopt);

    // Staging CB: depth >= 2-3 chunks so receive(k+1) overlaps forward(k). Page = input aligned page.
    const uint32_t page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t input_num_pages = input_tensor.buffer()->num_pages();
    // Broadcast only [bcast_offset, bcast_offset+bcast_count) of the sender's shard (0 num = whole shard);
    // the rest of the output is untouched. Cuts data moved when the caller needs a sub-range.
    const uint32_t bcast_offset = operation_attributes.broadcast_offset_tiles;
    const uint32_t bcast_count =
        operation_attributes.broadcast_num_tiles > 0
            ? std::min(operation_attributes.broadcast_num_tiles, input_num_pages - bcast_offset)
            : input_num_pages;
    // Chunk size: the requested chunk_size_tiles if set, else the largest chunk whose triple-buffered CB
    // fits kChunkL1Budget, capped at kMaxChunkPages. Larger chunks amortize the per-chunk sem round-trip
    // (a tiny one-packet chunk is ~3x slower); the cap bounds L1 use. Kernel clamps to the tile count.
    constexpr uint32_t kCbDepthChunks = 3;
    constexpr uint32_t kChunkL1Budget = 768 * 1024;  // bytes for the staging CB (validated at 128 bf16 tiles)
    constexpr uint32_t kMaxChunkPages = 256;
    const uint32_t budget_chunk = std::max<uint32_t>(1, kChunkL1Budget / (kCbDepthChunks * page_size));
    const uint32_t auto_chunk_pages = std::min(budget_chunk, kMaxChunkPages);
    const uint32_t chunk_num_pages = std::min(
        bcast_count,
        operation_attributes.chunk_size_tiles > 0 ? operation_attributes.chunk_size_tiles : auto_chunk_pages);
    const uint32_t num_chunks = (bcast_count + chunk_num_pages - 1) / chunk_num_pages;
    const uint32_t cb_depth_pages = kCbDepthChunks * chunk_num_pages;

    const uint32_t cb_id = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto cb_config =
        tt::tt_metal::CircularBufferConfig(cb_depth_pages * page_size, {{cb_id, df}}).set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, worker_core_range, cb_config);

    // Packet-header CB: up to 4 headers (payload + sem-inc, per direction).
    const uint32_t packet_header_cb_id = tt::CB::c_in1;
    const uint32_t packet_header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    auto packet_header_cb_config =
        tt::tt_metal::CircularBufferConfig(4 * packet_header_size, {{packet_header_cb_id, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_id, packet_header_size);
    CreateCircularBuffer(program, worker_core_range, packet_header_cb_config);

    // CT layout must match broadcast_ring_relay.cpp exactly.
    std::vector<uint32_t> ct_args = {
        ring_size,
        operation_attributes.sender_ring_index,
        ring_index,
        input_num_pages,  // num_tiles (total per-device shard tiles; kernel chunks internally)
        page_size,
        chunk_num_pages,  // packet_size_in_pages (tiles per chunk)
        cb_id,
        packet_header_cb_id,
        unicast_forward_args[0],   // fwd_route_arg0 (to idx+1)
        unicast_forward_args[1],   // fwd_route_arg1
        unicast_backward_args[0],  // bwd_route_arg0 (to idx-1)
        unicast_backward_args[1],  // bwd_route_arg1
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(ct_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(ct_args);
    (void)num_chunks;

    auto relay_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/broadcast_ring/device/kernels/broadcast_ring_relay.cpp",
        worker_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = ct_args});

    // Runtime args (per worker core / link), matching broadcast_ring_relay.cpp:
    //   input_addr, output_addr, recv_sem_addr, ds_sem_noc_x, ds_sem_noc_y, ds_sem_addr,
    //   tile_start, tile_count, then fabric args.
    // Each link relays a disjoint slice of the shard's tiles (payload split -> ~num_links x bandwidth), and
    // targets the DOWNSTREAM's SAME-index worker core, so each link is an independent ring relay with its
    // own recv-semaphore counter (the global sem has an independent copy per core).
    const auto src_fabric_node_id = mesh_device->get_fabric_node_id(coord);
    const auto fwd_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
    const auto bwd_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
    // Split the broadcast sub-range [bcast_offset, +bcast_count) across links; each link's tiles are
    // absolute shard tile ids (bcast_offset + local).
    const uint32_t tiles_per_link = (bcast_count + operation_attributes.num_links - 1) / operation_attributes.num_links;
    for (uint32_t link = 0; link < operation_attributes.num_links; ++link) {
        const CoreCoord core = worker_cores[link];
        // Downstream same-index worker core: its noc coords (target of both directions' completion atomic-inc;
        // the same logical core on every device, so the coords are shared and only the route differs).
        const CoreCoord ds_core_noc = mesh_device->worker_core_from_logical_core(core);
        const uint32_t local_start = std::min(link * tiles_per_link, bcast_count);
        const uint32_t tile_start = bcast_offset + local_start;
        const uint32_t tile_count = std::min(tiles_per_link, bcast_count - local_start);
        std::vector<uint32_t> rt = {
            input_tensor.buffer()->address(),
            output_tensor.buffer()->address(),
            recv_semaphore.address(),
            static_cast<uint32_t>(ds_core_noc.x),
            static_cast<uint32_t>(ds_core_noc.y),
            recv_semaphore.address(),  // downstream recv-sem: same L1 offset (global semaphore)
            tile_start,
            tile_count,
        };
        // Fabric connection args, in the exact layout FabricConnectionManager::build_from_args expects:
        //   [forward_flag] [forward sender args if flag] [backward_flag] [backward args if flag].
        // Bidirectional: open the forward connection iff this device sends to idx+1, backward iff to idx-1.
        rt.push_back(send_fwd ? 1u : 0u);  // forward connection flag
        if (send_fwd) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_fabric_node_id, fwd_fabric_node_id, link, program, core, rt);
        }
        rt.push_back(send_bwd ? 1u : 0u);  // backward connection flag
        if (send_bwd) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_fabric_node_id, bwd_fabric_node_id, link, program, core, rt);
        }
        tt::tt_metal::SetRuntimeArgs(program, relay_kernel_id, core, rt);
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .worker_cores = worker_cores,
            .relay_kernel_id = relay_kernel_id,
            .recv_semaphore = recv_semaphore,
            .barrier_semaphore = barrier_semaphore,
            .ring_index = ring_index}};
}

// Number of L1 recv slots (== staging CB depth). The backward credit protocol keeps at most this many
// chunks in flight per device; must match kCbDepthChunks used for the CB and the kernel's num_slots CT arg.
static constexpr uint32_t kL1RelaySlots = 3;

BroadcastRingProgramFactory::cached_program_t BroadcastRingProgramFactory::create_at_l1(
    const BroadcastRingParams& operation_attributes,
    const ttnn::MeshCoordinate& coord,
    const BroadcastRingInputs& tensor_args,
    Tensor& tensor_return_value,
    const tt::tt_metal::GlobalSemaphore& recv_semaphore,
    const tt::tt_metal::GlobalSemaphore& cred_fwd_semaphore,
    const tt::tt_metal::GlobalSemaphore& cred_bwd_semaphore) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;
    tt::tt_metal::Program program{};
    auto* mesh_device = input_tensor.device();

    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t ring_index =
        ::ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, coord, operation_attributes.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, coord, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    std::optional<MeshCoordinate> backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, coord, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value(), "broadcast_ring needs a forward neighbour (Ring topology)");

    auto [unicast_forward_args, unicast_backward_args] =
        ::ttnn::ccl::get_forward_backward_line_unicast_configuration(coord, forward_coord, backward_coord, mesh_device);

    // Roles (same arc split as the DRAM path). L1 adds credit roles: a device that received data credits its
    // upstream one hop back (fwd-arc devices credit idx-1 via the backward conn; bwd-arc credit idx+1 via the
    // forward conn). A connection is opened if it carries data OR a credit.
    const uint32_t fwd_hops = (ring_index + ring_size - operation_attributes.sender_ring_index) % ring_size;
    const uint32_t bwd_hops = (ring_size - fwd_hops) % ring_size;
    const uint32_t HF = ring_size / 2;
    const uint32_t HB = (ring_size - 1) / 2;
    const bool is_sender = (fwd_hops == 0);
    const bool on_fwd_arc = !is_sender && (fwd_hops <= HF);
    const bool on_bwd_arc = !is_sender && !on_fwd_arc;
    const bool send_data_fwd = is_sender || (on_fwd_arc && fwd_hops < HF);
    const bool send_data_bwd = is_sender || (on_bwd_arc && bwd_hops < HB);
    const bool credit_via_forward = on_bwd_arc;   // credit idx+1 (my upstream on the backward arc)
    const bool credit_via_backward = on_fwd_arc;  // credit idx-1 (my upstream on the forward arc)
    const bool open_forward = send_data_fwd || credit_via_forward;
    const bool open_backward = send_data_bwd || credit_via_backward;

    const uint32_t num_workers_per_link = 1;
    const auto [worker_core_range, worker_cores] = ::ttnn::ccl::choose_worker_cores(
        operation_attributes.num_links,
        num_workers_per_link,
        mesh_device,
        operation_attributes.sub_device_id,
        CoreCoord(0, 0),
        std::nullopt);

    const uint32_t page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t input_num_pages = input_tensor.buffer()->num_pages();
    const uint32_t bcast_offset = operation_attributes.broadcast_offset_tiles;
    const uint32_t bcast_count =
        operation_attributes.broadcast_num_tiles > 0
            ? std::min(operation_attributes.broadcast_num_tiles, input_num_pages - bcast_offset)
            : input_num_pages;

    // Chunk sizing identical to the DRAM path; slots = kL1RelaySlots. The recv buffer is num_slots x chunk.
    constexpr uint32_t kChunkL1Budget = 768 * 1024;
    constexpr uint32_t kMaxChunkPages = 256;
    const uint32_t budget_chunk = std::max<uint32_t>(1, kChunkL1Budget / (kL1RelaySlots * page_size));
    const uint32_t auto_chunk_pages = std::min(budget_chunk, kMaxChunkPages);
    const uint32_t chunk_num_pages = std::min(
        bcast_count,
        operation_attributes.chunk_size_tiles > 0 ? operation_attributes.chunk_size_tiles : auto_chunk_pages);
    const uint32_t recv_buf_pages = kL1RelaySlots * chunk_num_pages;

    // Recv buffer: kL1RelaySlots slots the upstream fabric-writes into; the kernel indexes it by chunk%slots.
    const uint32_t cb_id = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto cb_config =
        tt::tt_metal::CircularBufferConfig(recv_buf_pages * page_size, {{cb_id, df}}).set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, worker_core_range, cb_config);

    // Packet headers: up to 2 per direction (payload + sem-inc). Credit-only directions use just the sem-inc.
    const uint32_t packet_header_cb_id = tt::CB::c_in1;
    const uint32_t packet_header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    auto packet_header_cb_config =
        tt::tt_metal::CircularBufferConfig(4 * packet_header_size, {{packet_header_cb_id, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_id, packet_header_size);
    CreateCircularBuffer(program, worker_core_range, packet_header_cb_config);

    // CT layout must match broadcast_ring_relay_l1.cpp exactly.
    std::vector<uint32_t> ct_args = {
        ring_size,
        operation_attributes.sender_ring_index,
        ring_index,
        input_num_pages,
        page_size,
        chunk_num_pages,  // tiles per chunk == slot size in pages
        cb_id,
        packet_header_cb_id,
        unicast_forward_args[0],
        unicast_forward_args[1],
        unicast_backward_args[0],
        unicast_backward_args[1],
        kL1RelaySlots,
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(ct_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(ct_args);

    auto relay_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/broadcast_ring/device/kernels/broadcast_ring_relay_l1.cpp",
        worker_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = ct_args});

    const auto src_fabric_node_id = mesh_device->get_fabric_node_id(coord);
    const auto fwd_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
    const auto bwd_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
    const uint32_t tiles_per_link = (bcast_count + operation_attributes.num_links - 1) / operation_attributes.num_links;
    for (uint32_t link = 0; link < operation_attributes.num_links; ++link) {
        const CoreCoord core = worker_cores[link];
        const CoreCoord ds_core_noc = mesh_device->worker_core_from_logical_core(core);
        const uint32_t local_start = std::min(link * tiles_per_link, bcast_count);
        const uint32_t tile_start = bcast_offset + local_start;
        const uint32_t tile_count = std::min(tiles_per_link, bcast_count - local_start);
        // recv_sem is data_ready (upstream inc); cred_* are the backward slot-free credits. All are global
        // sems at the same L1 offset on every core, so one address serves both the local wait and the
        // remote (same-logical-core) atomic-inc target.
        std::vector<uint32_t> rt = {
            input_tensor.buffer()->address(),
            output_tensor.buffer()->address(),
            recv_semaphore.address(),
            static_cast<uint32_t>(ds_core_noc.x),
            static_cast<uint32_t>(ds_core_noc.y),
            recv_semaphore.address(),
            cred_fwd_semaphore.address(),
            cred_bwd_semaphore.address(),
            tile_start,
            tile_count,
        };
        rt.push_back(open_forward ? 1u : 0u);
        if (open_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_fabric_node_id, fwd_fabric_node_id, link, program, core, rt);
        }
        rt.push_back(open_backward ? 1u : 0u);
        if (open_backward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_fabric_node_id, bwd_fabric_node_id, link, program, core, rt);
        }
        tt::tt_metal::SetRuntimeArgs(program, relay_kernel_id, core, rt);
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .worker_cores = worker_cores,
            .relay_kernel_id = relay_kernel_id,
            .recv_semaphore = recv_semaphore,
            .barrier_semaphore = recv_semaphore,
            .extra_semaphores = {cred_fwd_semaphore, cred_bwd_semaphore},
            .ring_index = ring_index}};
}

void BroadcastRingProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const BroadcastRingParams& /*operation_attributes*/,
    const BroadcastRingInputs& tensor_args,
    Tensor& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared = cached_workload.shared_variables.at(range);
        auto& rt_by_core = tt::tt_metal::GetRuntimeArgs(program, shared.relay_kernel_id);
        for (const auto& core : shared.worker_cores) {
            auto& rt = rt_by_core[core.x][core.y];
            rt[0] = tensor_args.input_tensor.buffer()->address();
            rt[1] = tensor_return_value.buffer()->address();
        }
    }
}

}  // namespace ttnn::prim
