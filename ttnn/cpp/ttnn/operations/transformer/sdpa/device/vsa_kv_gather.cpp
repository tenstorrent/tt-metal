// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/vsa_kv_gather.hpp"

#include <algorithm>
#include <set>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn::prim {

namespace {
constexpr uint32_t kTile = 32;
constexpr uint32_t kMaxScatterPagesPerPacket = 4;     // scatter writes carry up to 4 destination addresses
constexpr uint32_t kHeuristicMaxChunksPerSync = 160;  // all_gather_async's HEURISTIC_MAX_CHUNKS_PER_SYNC
constexpr const char* kReaderKernel =
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/vsa_kv_gather_reader.cpp";
constexpr const char* kWriterKernel =
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/vsa_kv_gather_writer.cpp";
}  // namespace

VsaKvGatherArtifacts build_vsa_kv_gather(
    tt::tt_metal::Program& program,
    const Tensor& k,
    const Tensor& v,
    const Tensor& gathered_k,
    const Tensor& gathered_v,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const ttnn::experimental::ccl::AllGatherFusedOpSignaler& fused_op_signaler,
    uint32_t num_workers_per_link,
    std::optional<uint32_t> chunks_per_sync) {
    using namespace tt::tt_metal;
    auto* mesh_device = k.device();
    TT_FATAL(mesh_device != nullptr, "vsa_kv_gather: mesh device not found");
    TT_FATAL(forward_coord.has_value() && backward_coord.has_value(), "vsa_kv_gather: a ring needs both neighbours");
    TT_FATAL(semaphore.size() >= 2, "vsa_kv_gather: two out_ready GlobalSemaphores are required");
    TT_FATAL(num_links >= 1 && num_workers_per_link >= 1, "vsa_kv_gather: links and workers must be >= 1");

    const auto ks = k.padded_shape();
    const auto gs = gathered_k.padded_shape();
    TT_FATAL(
        ks == v.padded_shape() && gs == gathered_v.padded_shape(),
        "vsa_kv_gather: K and V (and their gathered buffers) must have the same shape");
    const uint32_t H = ks[1];
    const uint32_t ht_local = ks[2] / kTile;
    const uint32_t dht = ks[3] / kTile;
    const uint32_t ht_total = gs[2] / kTile;
    TT_FATAL(
        gs[1] == H && gs[3] == ks[3] && ht_total == ring_size * ht_local,
        "vsa_kv_gather: gathered K/V ({}) must be the input ({}) x ring_size ({}) on dim 2",
        gs,
        ks,
        ring_size);
    const uint32_t page_size = k.buffer()->page_size();
    TT_FATAL(
        page_size == v.buffer()->page_size() && page_size == gathered_k.buffer()->page_size() &&
            page_size == gathered_v.buffer()->page_size(),
        "vsa_kv_gather: K/V page sizes differ");

    // --- topology ---
    const uint32_t num_directions_per_link = 2;
    const uint32_t num_mux = num_workers_per_link == 1 ? 0u : 1u;
    const uint32_t num_cores_per_link = num_directions_per_link * (num_mux + num_workers_per_link);
    auto [num_targets_forward, num_targets_backward] =
        ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, ccl::Topology::Ring, false);
    auto [unicast_forward_args, unicast_backward_args] = ccl::get_forward_backward_line_unicast_configuration(
        sender_device_coord, forward_coord, backward_coord, mesh_device);

    // --- cores: from (0, 0) row-major; per link: direction 0 (MUX, workers), direction 1 (MUX, workers) ---
    const auto [all_core_range, all_cores] =
        ccl::choose_worker_cores(num_links, num_cores_per_link, mesh_device, sub_device_id, CoreCoord{0, 0});
    std::vector<CoreRange> sender_worker_core_ranges, mux_core_ranges, termination_master_core_ranges;
    std::set<CoreRange> sender_forward_core_ranges, sender_backward_core_ranges;
    const auto mux_connection_valid = [&](uint32_t dir) {
        return (dir && backward_coord.has_value()) || (!dir && forward_coord.has_value());
    };
    {
        uint32_t core_id = 0;
        for (uint32_t link = 0; link < num_links; ++link) {
            for (uint32_t dir = 0; dir < num_directions_per_link; ++dir) {
                if (num_mux) {
                    const auto& mux_core = all_cores[core_id++];
                    if (mux_connection_valid(dir)) {
                        mux_core_ranges.emplace_back(mux_core);
                    }
                }
                for (uint32_t worker = 0; worker < num_workers_per_link; ++worker) {
                    const auto& worker_core = all_cores[core_id++];
                    if (num_mux && worker == 0) {
                        termination_master_core_ranges.emplace_back(worker_core);
                    }
                    (dir ? sender_forward_core_ranges : sender_backward_core_ranges).emplace(worker_core);
                    sender_worker_core_ranges.emplace_back(worker_core);
                }
            }
        }
    }
    const CoreRangeSet sender_worker_core_range_set(sender_worker_core_ranges);
    const CoreRangeSet mux_core_range_set(mux_core_ranges);

    // --- packets and the L1 staging CB ---
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    TT_FATAL(
        packet_size_bytes >= page_size,
        "vsa_kv_gather: fabric packet ({} B) smaller than a page ({} B)",
        packet_size_bytes,
        page_size);
    const uint32_t tiles_per_packet =
        std::min<uint32_t>(kMaxScatterPagesPerPacket, static_cast<uint32_t>(packet_size_bytes / page_size));
    const uint32_t cb_num_pages = 3 * tiles_per_packet;  // triple buffering
    const tt::DataFormat df = datatype_to_dataformat_converter(k.dtype());
    const uint32_t sender_cb_index = tt::CB::c_in0;
    CircularBufferConfig cb_sender_config = CircularBufferConfig(cb_num_pages * page_size, {{sender_cb_index, df}})
                                                .set_page_size(sender_cb_index, page_size);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_sender_config);

    // --- fused-op signalers: one per direction plus the direction-1 writers' local-slice pre-signal ---
    auto signaler_forward = fused_op_signaler;
    auto signaler_backward = fused_op_signaler;
    auto signaler_sender_workers = fused_op_signaler;
    {
        auto workers_forward = corerange_to_cores(CoreRangeSet(sender_forward_core_ranges), std::nullopt, true);
        auto workers_backward = corerange_to_cores(CoreRangeSet(sender_backward_core_ranges), std::nullopt, true);
        signaler_forward.init_all_gather(
            program, mesh_device, CoreRangeSet(sender_forward_core_ranges), workers_forward);
        signaler_backward.init_all_gather(
            program, mesh_device, CoreRangeSet(sender_backward_core_ranges), workers_backward);
        signaler_sender_workers.init_all_gather(
            program, mesh_device, CoreRangeSet(sender_forward_core_ranges), workers_forward);
    }

    // --- MUX config ---
    const uint32_t l1_unreserved_base_address = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_workers_per_link,
        0,
        /*num_buffers_full_size_channels=*/1,
        0,
        packet_size_bytes,
        l1_unreserved_base_address);

    // --- kernels ---
    const std::vector<uint32_t> common_ct = {
        ring_size,
        ring_index,
        sender_cb_index,
        tiles_per_packet,
        page_size,
        num_targets_forward,
        num_targets_backward,
        H,
        dht,
        ht_local,
        ht_total,
        /*fuse_op=*/1u,
    };
    std::vector<uint32_t> reader_ct = common_ct;
    TensorAccessorArgs(k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(v.buffer()).append_to(reader_ct);
    TensorAccessorArgs(gathered_k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(gathered_v.buffer()).append_to(reader_ct);
    const auto reader_kernel_id =
        CreateKernel(program, kReaderKernel, sender_worker_core_range_set, ReaderDataMovementConfig(reader_ct));

    std::vector<uint32_t> writer_ct = common_ct;
    std::map<std::string, std::string> writer_defines;
    if (num_mux) {
        ccl::fabric_mux_connection_ct_args(
            num_workers_per_link, tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, mux_kernel_config, writer_ct);
        writer_defines["USE_WORKER_MUX"] = "1";
    }
    writer_ct.insert(writer_ct.end(), unicast_forward_args.begin(), unicast_forward_args.end());
    writer_ct.insert(writer_ct.end(), unicast_backward_args.begin(), unicast_backward_args.end());
    TensorAccessorArgs(gathered_k.buffer()).append_to(writer_ct);
    TensorAccessorArgs(gathered_v.buffer()).append_to(writer_ct);
    const auto writer_kernel_id = CreateKernel(
        program, kWriterKernel, sender_worker_core_range_set, WriterDataMovementConfig(writer_ct, writer_defines));

    KernelHandle mux_kernel_id = 0;
    if (num_mux) {
        mux_kernel_id = CreateKernel(
            program,
            "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
            mux_core_range_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
                .opt_level = KernelBuildOptLevel::O3});
    }

    // --- per-worker rows and runtime args ---
    VsaKvGatherArtifacts a;
    a.reader_kernel_id = reader_kernel_id;
    a.writer_kernel_id = writer_kernel_id;
    a.all_cores = all_cores;
    a.num_links = num_links;
    a.num_directions_per_link = num_directions_per_link;
    a.num_workers_per_direction = num_workers_per_link;
    a.num_mux_cores_per_direction_per_link = num_mux;
    a.num_cores_per_link = num_cores_per_link;
    a.tiles_per_packet = tiles_per_packet;
    const uint32_t G = num_links * num_workers_per_link;
    TT_FATAL(ht_local >= G, "vsa_kv_gather: {} tile rows per shard cannot feed {} workers", ht_local, G);
    const uint32_t tiles_per_row = 2 * H * dht;
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t base = ht_local / G, rem = ht_local % G;
        a.row_ranges.emplace_back(g * base + std::min(g, rem), (g + 1) * base + std::min(g + 1, rem));
    }
    // the all-gather's chunks_per_sync heuristic on the largest range, the same value for every worker
    const uint32_t tiles_worker0 = (a.row_ranges[0].second - a.row_ranges[0].first) * tiles_per_row;
    a.chunks_per_sync = chunks_per_sync.value_or(
        std::min(std::max<uint32_t>(tiles_worker0 / tiles_per_packet, 1), kHeuristicMaxChunksPerSync));

    auto worker_core_iter = sender_worker_core_ranges.cbegin();
    auto mux_core_iter = mux_core_ranges.cbegin();
    auto termination_master_core_iter = termination_master_core_ranges.cbegin();
    for (uint32_t link = 0; link < num_links; ++link) {
        for (uint32_t dir = 0; dir < num_directions_per_link; ++dir) {
            CoreCoord termination_master_logical_core = {0, 0};
            CoreCoord mux_virtual_core = {0, 0};
            if (num_mux) {
                if (mux_connection_valid(dir)) {
                    const auto mux_logical_core = (mux_core_iter++)->start_coord;
                    mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);
                    const auto src_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
                    // direction 1 sends to the backward neighbour, direction 0 to the forward one
                    const auto dst_node_id =
                        mesh_device->get_fabric_node_id(dir ? backward_coord.value() : forward_coord.value());
                    const auto mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                    SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
                }
                termination_master_logical_core = (termination_master_core_iter++)->start_coord;
            }
            for (uint32_t worker = 0; worker < num_workers_per_link; ++worker) {
                const CoreCoord core = (worker_core_iter++)->start_coord;
                const CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
                const uint32_t g = link * num_workers_per_link + worker;
                const auto [row_start, row_end] = a.row_ranges[g];
                const uint32_t self_write_done_semaphore = CreateSemaphore(program, {core}, 0);

                std::vector<uint32_t> reader_rt = {
                    k.buffer()->address(),
                    v.buffer()->address(),
                    gathered_k.buffer()->address(),
                    gathered_v.buffer()->address(),
                    semaphore.at(dir).address(),
                    dir,
                    row_start,
                    row_end,
                    a.chunks_per_sync,
                };
                reader_rt.push_back(self_write_done_semaphore);
                (dir ? signaler_forward : signaler_backward).push_all_gather_fused_op_rt_args(reader_rt, G, g, dir);
                SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt);

                std::vector<uint32_t> writer_rt = {
                    gathered_k.buffer()->address(),
                    gathered_v.buffer()->address(),
                    virtual_core.x,
                    virtual_core.y,
                    semaphore.at(dir).address(),
                    dir,
                    row_start,
                    row_end,
                    a.chunks_per_sync,
                };
                if (num_mux) {
                    ccl::fabric_mux_connection_rt_args(
                        mux_connection_valid(dir),
                        worker == 0,
                        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        mux_virtual_core,
                        worker,
                        core,
                        mux_kernel_config,
                        program,
                        mesh_device->worker_core_from_logical_core(termination_master_logical_core),
                        writer_rt);
                } else {
                    // direct fabric connections: [forward valid, forward args][backward valid, backward args]
                    const auto src_node = mesh_device->get_fabric_node_id(sender_device_coord);
                    if (dir) {
                        writer_rt.push_back(false);
                        writer_rt.push_back(backward_coord.has_value());
                        if (backward_coord.has_value()) {
                            tt::tt_fabric::append_fabric_connection_rt_args(
                                src_node,
                                mesh_device->get_fabric_node_id(backward_coord.value()),
                                link,
                                program,
                                {core},
                                writer_rt);
                        }
                    } else {
                        writer_rt.push_back(forward_coord.has_value());
                        if (forward_coord.has_value()) {
                            tt::tt_fabric::append_fabric_connection_rt_args(
                                src_node,
                                mesh_device->get_fabric_node_id(forward_coord.value()),
                                link,
                                program,
                                {core},
                                writer_rt);
                        }
                        writer_rt.push_back(false);
                    }
                }
                writer_rt.push_back(self_write_done_semaphore);
                signaler_sender_workers.push_all_gather_fused_op_rt_args(writer_rt, G, g, 1);
                SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt);
            }
        }
    }
    return a;
}

void vsa_kv_gather_override_runtime_arguments(
    tt::tt_metal::Program& program,
    const VsaKvGatherArtifacts& a,
    const std::vector<GlobalSemaphore>& semaphore,
    const Tensor& k,
    const Tensor& v,
    const Tensor& gathered_k,
    const Tensor& gathered_v) {
    auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, a.reader_kernel_id);
    auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, a.writer_kernel_id);
    for (uint32_t link = 0; link < a.num_links; ++link) {
        for (uint32_t dir = 0; dir < a.num_directions_per_link; ++dir) {
            for (uint32_t worker = 0; worker < a.num_workers_per_direction; ++worker) {
                const auto core = a.worker_core(link, dir, worker);
                auto& r = reader_args[core.x][core.y];
                r[0] = k.buffer()->address();
                r[1] = v.buffer()->address();
                r[2] = gathered_k.buffer()->address();
                r[3] = gathered_v.buffer()->address();
                r[4] = semaphore.at(dir).address();
                auto& w = writer_args[core.x][core.y];
                w[0] = gathered_k.buffer()->address();
                w[1] = gathered_v.buffer()->address();
                w[4] = semaphore.at(dir).address();
            }
        }
    }
}

}  // namespace ttnn::prim
