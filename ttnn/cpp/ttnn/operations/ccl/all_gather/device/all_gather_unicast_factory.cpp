// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_unicast_factory.hpp"
#include "all_gather_device_operation.hpp"
#include "all_gather_test_knobs.hpp"  // TEST ONLY

#include <algorithm>
#include <map>
#include <set>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/common/host/descriptor_runtime_args.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::ccl {

using namespace ::ttnn::ccl;

namespace {

////////////////////////////////////////////////////////////////
// Store-and-forward AllGather (line/ring over a single mesh axis; for Fabric 1D and 2D)
//
// Every device relays stripes to its neighbor one hop at a time; a shard reaches far devices by being
// re-forwarded at each hop. Forward and backward directions run on separate cores. Per direction: the reader
// (CB producer, no fabric) reads iteration 0 from local input and later iterations from what upstream relayed
// into our output; the writer (CB consumer) unicasts each stripe one hop to the neighbor's output (same
// address on every device). Direction/topology are runtime args, so both kernels compile once and run on all
// cores. Two semaphores: barrier_sem (init handshake) and data_valid_sem (relay gate + completion).
////////////////////////////////////////////////////////////////

tt::tt_metal::ProgramDescriptor build_unicast_descriptor_at(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const AllGatherInputs& tensor_args,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& barrier_sem,
    const tt::tt_metal::GlobalSemaphore& data_valid_sem,
    const CoreRangeSet& available_cores) {
    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh_device = input_tensor.device();

    ////////////////////////////////////////////////////////////////
    // Fabric setup
    //
    // Glossary:
    //   relay          -- re-forward a received stripe from upstream one hop to downstream.
    //   slice          -- portion of tensor stripes allocated to this (link, worker)
    //   sink direction -- a direction that forwards nothing (num_iters == 0), e.g. a line endpoint's dead side.
    //   sink stripe    -- a stripe consumed here, not relayed onward (a line endpoint's incoming, or a ring
    //                     antipode).
    //   antipode       -- on a ring, the device N/2 hops away.
    ////////////////////////////////////////////////////////////////

    TT_FATAL(!operation_attributes.is_true_2d(), "all_gather unicast algorithm does not support true 2D topologies");

    const uint32_t axis = operation_attributes.get_1d_axis();
    const auto topology = operation_attributes.axis_topology[axis];
    const bool is_ring = tt::tt_fabric::is_ring_or_torus(topology);

    const uint32_t num_devices = operation_attributes.num_devices;
    const uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    auto fwd_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, sender_device_coord, 1, topology, axis);
    auto bwd_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, sender_device_coord, -1, topology, axis);

    // Stripes a direction sends from a device: ring -> N/2; line fwd -> d+1, bwd -> N-d; 0 at a dead endpoint.
    auto relay_iters = [&](uint32_t idx, bool is_forward) -> uint32_t {
        if (is_ring) {
            return num_devices / 2;
        }
        return is_forward ? (idx + 1 < num_devices ? idx + 1 : 0) : (idx > 0 ? num_devices - idx : 0);
    };
    const uint32_t fwd_iters = relay_iters(device_idx, true);
    const uint32_t bwd_iters = relay_iters(device_idx, false);
    TT_FATAL(fwd_iters > 0 || bwd_iters > 0, "device participates in neither direction");

    // Even-sized ring: for load balancing, the antipode device receives the antipode stripe as halves from both
    // forward and backward directions.
    const bool ring_even_split = is_ring && (num_devices % 2 == 0);

    const uint32_t packet_size = operation_attributes.packet_size;

    ////////////////////////////////////////////////////////////////
    // Page indexing
    //
    // The kernels walk this device's stripe of the output (all_gather_walk.hpp) in units of A bytes and pack the
    // runs that are contiguous in the output into packets of up to 4 segments. A walk element is one output page, or
    // one unit when a stripe row is not whole output pages; workers split the elements into contiguous slices.
    ////////////////////////////////////////////////////////////////

    const auto walk_geometry = compute_walk_geometry(
        input_tensor,
        output_tensor.tensor_spec(),
        operation_attributes.dim_from_end,
        num_devices,
        packet_size,
        /*unicast=*/true);
    TT_FATAL(
        walk_geometry.has_value(),
        "all_gather unicast cannot cut {} B input pages and {} B output pages on the relay's read alignment",
        input_tensor.buffer()->page_size(),
        output_tensor.buffer()->page_size());
    const all_gather_walk::WalkGeo& geo = *walk_geometry;
    const uint32_t unit_bytes = geo.unit_bytes;
    const uint32_t num_elements = input_tensor.buffer()->num_pages() * geo.in_page_units / geo.element_units;
    const uint32_t packet_cap = compute_packet_cap(geo, output_tensor);

    ////////////////////////////////////////////////////////////////
    // Core selection
    //
    // Each link runs two directions: forward (dir 0) and backward (dir 1). The workers of a direction can't each open
    // a direct connection (an ERISC exposes one worker sender channel per direction), so they share a fabric mux: one
    // mux core per direction per link owns the connection and multiplexes their traffic.
    //
    // Flat layout, per link:
    //   [dir 0: mux, worker 0 .. worker W-1][dir 1: mux, worker 0 .. worker W-1]
    ////////////////////////////////////////////////////////////////

    // Num worker cores per direction per link.
    // This is a major perf knob, below heuristic was determined from extensive test sweeps.
    uint32_t num_links = operation_attributes.axis_num_links[axis];
    const bool is_wormhole = input_tensor.device()->arch() == tt::ARCH::WORMHOLE_B0;
    const bool wh_tuned = is_wormhole && !ag_test::wh_legacy();  // TEST ONLY: TT_AG_WH_LEGACY=1 keeps the old WH values
    const uint64_t input_bytes = static_cast<uint64_t>(num_elements) * geo.element_units * unit_bytes;
    const uint64_t busiest_link_packets =
        tt::div_up(input_bytes * (num_devices - 1), uint64_t{is_ring ? 2u : 1u} * std::max(1u, num_links) * packet_cap);
    uint32_t workers_per_dir = 1;
    if (is_wormhole) {
        // 2 workers keep up with a link (1 link, 6144 B payload, ring: 3 are 3-13% slower on bf16 tiles at 2-4 MB)
        // unless short segments cut every packet below the payload (packet_cap < max_packet_bytes: row-major rows of
        // 1 KB or less per device, bfp8 tiles in one-tile lanes). There 3 are faster from about 400 packets on the
        // busiest link-direction (ring 1-8 MB: 1-9%; forced-unicast line: 5-19%) and up to 5% slower below 300.
        constexpr uint64_t three_worker_packets = 400;
        workers_per_dir =
            (wh_tuned && packet_cap < geo.max_packet_bytes && busiest_link_packets >= three_worker_packets) ? 3 : 2;
    } else if (input_tensor.device()->arch() == tt::ARCH::BLACKHOLE) {
        // 2 workers cannot keep up with a ring's link (3 are 1.4-7% faster at 256 KB - 9 MB); on a line 2 are 10%
        // faster at 121 packets on the busiest link-direction, 3 as fast or faster from 242.
        constexpr uint64_t line_three_worker_packets = 160;
        workers_per_dir = (is_ring || busiest_link_packets >= line_three_worker_packets) ? 3 : 2;
    }
    workers_per_dir = std::max(ag_test::env_u32("TT_AG_WORKERS").value_or(workers_per_dir), 1u);  // TEST ONLY env

    // Shrink core usage to fit available core grid. Shrink workers_per_link first, and then shrink num_links.
    const uint32_t num_available_cores = available_cores.num_cores();
    auto cores_per_link = [](uint32_t workers) { return 2u * (workers + 1u); };
    const uint32_t wanted_workers = workers_per_dir;
    const uint32_t wanted_links = num_links;
    while (workers_per_dir > 1 && num_links * cores_per_link(workers_per_dir) > num_available_cores) {
        --workers_per_dir;
    }
    if (num_links * cores_per_link(workers_per_dir) > num_available_cores) {
        // Even one worker per direction per link does not fit; drop links (workers_per_dir is 1 by now).
        num_links = num_available_cores / cores_per_link(workers_per_dir);
        TT_FATAL(
            num_links > 0,
            "all_gather needs at least {} worker cores but only {} are available; provide a larger sub_core_grid.",
            cores_per_link(workers_per_dir),
            num_available_cores);
    }
    if (workers_per_dir != wanted_workers || num_links != wanted_links) {
        log_warning(
            tt::LogOp,
            "all_gather scaled down from {} links x {} workers/direction to {} links x {} workers/direction to fit "
            "the {} available worker cores. This may lead to performance loss.",
            wanted_links,
            wanted_workers,
            num_links,
            workers_per_dir,
            num_available_cores);
    }

    constexpr uint32_t num_directions = 2;  // 0 = forward, 1 = backward
    const uint32_t cores_per_dir = workers_per_dir + 1;
    const uint32_t num_cores_per_link = num_directions * cores_per_dir;

    // all_cores contains workers + mux
    [[maybe_unused]] auto [all_core_range, all_cores] = ttnn::ccl::choose_worker_cores(
        num_links,
        num_cores_per_link,
        mesh_device,
        operation_attributes.subdevice_id,
        /*core_grid_offset=*/CoreCoord{0, 0},
        operation_attributes.sub_core_grid);
    TT_FATAL(
        all_cores.size() == static_cast<size_t>(num_links) * num_cores_per_link,
        "all_gather needs {} worker cores ({} links x {} cores/link) but only {} are available; provide a larger "
        "sub_core_grid.",
        static_cast<size_t>(num_links) * num_cores_per_link,
        num_links,
        num_cores_per_link,
        all_cores.size());

    // Put each direction's mux and its workers on one core row where the grid allows, directions on distinct rows
    // first, then rows double up; otherwise keep choose_worker_cores' row-major order. Much of a direction's traffic
    // runs along its row's NOC links (reads into its workers, packets to its mux and on to ethernet), so directions
    // sharing a row share that bandwidth: the row-major order puts three on one row (BH, 2 links: 2.7x slower on a
    // 26 MB ring, 1.3x on a line).
    auto group_by_row = [&]() {
        std::map<uint32_t, std::vector<CoreCoord>> free_by_row;
        for (const auto& core : corerange_to_cores(available_cores, std::nullopt, /*row_wise=*/true)) {
            free_by_row[core.y].push_back(core);
        }
        for (auto& [y, free] : free_by_row) {
            std::sort(free.begin(), free.end(), [](const CoreCoord& a, const CoreCoord& b) { return a.x < b.x; });
        }
        std::vector<CoreCoord> grouped;
        for (bool progress = true; progress && grouped.size() < all_cores.size();) {
            progress = false;
            for (auto& [y, free] : free_by_row) {
                if (grouped.size() < all_cores.size() && free.size() >= cores_per_dir) {
                    grouped.insert(grouped.end(), free.begin(), free.begin() + cores_per_dir);
                    free.erase(free.begin(), free.begin() + cores_per_dir);
                    progress = true;
                }
            }
        }
        if (grouped.size() == all_cores.size()) {
            all_cores = std::move(grouped);
            return true;
        }
        return false;
    };
    const bool no_row_group = ag_test::env_u32("TT_AG_NO_ROW_GROUP").value_or(0) != 0;  // TEST ONLY
    const bool row_grouped = !no_row_group && group_by_row();

    // Helpers to index into the flat core vector (dir: 0 = forward, 1 = backward).
    auto core_at = [&](uint32_t link, uint32_t dir, uint32_t idx_in_dir) -> const CoreCoord& {
        return all_cores[(link * num_cores_per_link) + (dir * cores_per_dir) + idx_in_dir];
    };
    auto mux_core = [&](uint32_t link, uint32_t dir) -> const CoreCoord& { return core_at(link, dir, 0); };
    auto worker_core = [&](uint32_t link, uint32_t dir, uint32_t w) -> const CoreCoord& {
        return core_at(link, dir, 1 + w);
    };
    auto dir_neighbor = [&](uint32_t dir) { return dir == 0 ? fwd_coord : bwd_coord; };
    auto dir_active = [&](uint32_t dir) { return dir_neighbor(dir).has_value(); };

    // Reader/writer kernels + CB run on worker cores only; the mux kernel runs on its own cores (created only
    // for a direction that has a neighbor).
    std::set<CoreRange> worker_core_set;
    for (uint32_t link = 0; link < num_links; ++link) {
        for (uint32_t dir = 0; dir < num_directions; ++dir) {
            for (uint32_t w = 0; w < workers_per_dir; ++w) {
                worker_core_set.emplace(worker_core(link, dir, w));
            }
        }
    }
    const CoreRangeSet worker_core_range(worker_core_set);

    // Fabric mux config. One worker per direction needs 4 buffers per channel to keep the link busy (2 are 12-14%
    // slower); from 2 workers, 2 buffers are within 3% of 4. The mux map runs up from the allocator base and stays
    // below the floor of the L1_SMALL region, where the semaphores live.
    const uint32_t mux_bufs = ag_test::env_u32("TT_AG_MUX_BUFS").value_or(workers_per_dir == 1 ? 4 : 2);  // TEST ONLY
    TT_FATAL(mux_bufs >= 1 && mux_bufs <= 255, "TT_AG_MUX_BUFS must be 1-255 (Mux V2 takes any count that fits L1)");
    const uint8_t num_buffers_per_channel = static_cast<uint8_t>(mux_bufs);
    const tt::tt_fabric::FabricMuxV2Config mux_config(
        *mesh_device,
        static_cast<uint8_t>(workers_per_dir),
        num_buffers_per_channel,
        tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*mesh_device),
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1),
        ::ttnn::ccl::l1_small_floor_address(*mesh_device));

    ////////////////////////////////////////////////////////////////
    // Circular Buffer and Kernel creation
    ////////////////////////////////////////////////////////////////

    // --- CB sizing ---
    // One CB page is a slot (all_gather_walk::slot_offsets): up to packets_per_slot packets with their records and the
    // writer's fabric headers.
    const uint32_t total_slices = num_links * workers_per_dir;
    const uint32_t packet_header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    // WH ring: at least 4 slots per worker iteration, so a short one still pipelines (1 MB, 6144 B payload: 11 packets
    // per iteration, 2 per slot instead of 5 are 11% faster; 4352 B payload, 1-1.5 MB: 3-4%). A unicast line gains at
    // 6144 B but is up to 10% slower at 4352 B (512 KB - 1 MB), so it keeps 2. From 20 packets per iteration (6144 B)
    // k keeps its 30 KB target.
    const uint32_t packets_per_slot = compute_packets_per_slot(
        geo,
        packet_cap,
        total_slices,
        packet_header_size,
        input_tensor,
        (wh_tuned && is_ring) ? 4u : all_gather_walk::kNumSlots);
    const uint32_t cb_page_size =
        all_gather_walk::slot_offsets(packets_per_slot, unit_bytes, packet_header_size, packet_cap).slot_bytes;

    // Input and relay CB. A slot is untyped bytes (records, headers, payload); the format only sizes the page.
    uint32_t cb0_id = tt::CB::c_in0;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = all_gather_walk::kNumSlots * cb_page_size,
        .core_ranges = worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb0_id),
            .data_format = tt::DataFormat::RawUInt8,
            .page_size = cb_page_size,
        }}},
    });

    // KERNEL CREATION
    // NOCs: the writer on NOC0 is 5-10% faster on a ring, 3-8% slower on a line (BH); the reader takes the other.
    const auto writer_noc =
        ag_test::env_noc("TT_AG_UC_WR_NOC", is_ring ? tt::tt_metal::NOC::NOC_0 : tt::tt_metal::NOC::NOC_1);
    const auto reader_noc =
        ag_test::env_noc("TT_AG_UC_RD_NOC", is_ring ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0);
    ag_test::check_distinct_nocs(reader_noc, writer_noc, "TT_AG_UC_RD_NOC / TT_AG_UC_WR_NOC");  // TEST ONLY

    // Reader
    std::vector<uint32_t> reader_compile_args = {
        unit_bytes,            // cut quantum
        geo.in_page_units,     // input page footprint (units)
        geo.out_page_units,    // output page footprint (units)
        geo.stripe_units,      // units this device contributes to one output row
        num_devices,           // device count (stripe indexing)
        geo.element_units,     // units per walk element
        geo.max_packet_bytes,  // fabric packet payload limit
        geo.dst_cross,         // runs may continue across output pages
        geo.src_cross,         // runs may continue across input pages
        packet_cap,            // packet payload cap
        packets_per_slot,      // packets per CB page
        packet_header_size,    // fabric packet header size
        cb0_id,                // cb id
        cb_page_size,          // cb entry size
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

    // Writer
    std::vector<uint32_t> writer_compile_args = {
        unit_bytes,               // cut quantum
        packet_cap,               // packet payload cap
        packets_per_slot,         // packets per CB page
        packet_header_size,       // fabric packet header size
        cb0_id,                   // cb id
        cb_page_size,             // cb entry size
        num_buffers_per_channel,  // mux buffers per channel
    };

    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_reader.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = worker_core_range;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_args);
    reader_kernel_desc.defines = ag_test::diag_defines();  // TEST ONLY
    reader_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = reader_noc};
    desc.kernels.push_back(std::move(reader_kernel_desc));

    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_writer.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = worker_core_range;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_args);
    writer_kernel_desc.defines = ag_test::diag_defines();  // TEST ONLY
    writer_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = writer_noc};
    desc.kernels.push_back(std::move(writer_kernel_desc));

    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 1;

    ////////////////////////////////////////////////////////////////
    // Runtime args
    //
    // The element split is per (link, worker) -- num_links * workers_per_dir slices -- and direction-independent,
    // so a (link, worker)'s forward and backward directions relay the same slice. Core assignment is
    // deterministic, so worker w has the same coords on every device: data_valid signals target the mirror
    // worker w on the neighbor.
    ////////////////////////////////////////////////////////////////

    const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);

    // Mux kernels: one fabric connection per active direction per link, to that direction's neighbor. The forwarder
    // writes every packet into an ethernet core, 1.8-2.3x faster on NOC0 (BH).
    const auto mux_noc = ag_test::env_noc("TT_AG_MUX_NOC", tt::tt_metal::NOC::NOC_0);  // TEST ONLY env
    for (uint32_t link = 0; link < num_links; ++link) {
        for (uint32_t dir = 0; dir < num_directions; ++dir) {
            if (dir_active(dir)) {
                tt::tt_fabric::add_fabric_mux_v2_to_program(
                    desc,
                    mux_config,
                    mux_core(link, dir),
                    sender_fabric_node_id,
                    mesh_device->get_fabric_node_id(*dir_neighbor(dir)),
                    link,
                    mux_noc);
            }
        }
    }

    // Data_valid cadence: about 4 signals per worker iteration, every 2 to 16 packets (WH: 64). Each signal is a
    // packet: fewer save link time, more start the downstream relay sooner (BH: 1-packet cadence up to 1.5x slower at
    // 8-26 MB; WH ring, 6144 B payload: 64 instead of 16 is 1.4-3.4% faster at 8-42 MB).
    const uint64_t signals_per_iter = ag_test::env_u32("TT_AG_SIG_PER_ITER").value_or(4),  // TEST ONLY env
        min_cadence_packets = ag_test::env_u32("TT_AG_SIG_MIN").value_or(2),
                   max_cadence_packets = ag_test::env_u32("TT_AG_SIG_MAX").value_or(wh_tuned ? 64 : 16);
    TT_FATAL(signals_per_iter > 0 && min_cadence_packets <= max_cadence_packets, "bad TT_AG_SIG_* values");
    auto cadence_for = [&](uint64_t slice_packets) {
        return std::clamp(tt::div_up(slice_packets, signals_per_iter), min_cadence_packets, max_cadence_packets);
    };

    if (ag_test::log_enabled() && device_idx == 0) {  // TEST ONLY; slice 0 is the largest slice
        const uint64_t iter_packets = tt::div_up(
            uint64_t{tt::div_up(num_elements, total_slices)} * geo.element_units * unit_bytes, uint64_t{packet_cap});
        log_info(
            tt::LogOp,
            "AGCFG factory=unicast arch={} topology={} num_devices={} num_links={} workers_per_dir={} mux_bufs={} k={} "
            "packet_cap={} max_packet_bytes={} unit_bytes={} cadence_pkts={} cadence_slots={} rd_noc={} wr_noc={} "
            "mux_noc={} row_group={} iter_packets={}",
            input_tensor.device()->arch() == tt::ARCH::WORMHOLE_B0 ? "wh" : "bh",
            is_ring ? "ring" : "line",
            num_devices,
            num_links,
            workers_per_dir,
            num_buffers_per_channel,
            packets_per_slot,
            packet_cap,
            geo.max_packet_bytes,
            unit_bytes,
            cadence_for(iter_packets),
            tt::div_up(cadence_for(iter_packets), uint64_t{packets_per_slot}),
            static_cast<int>(reader_noc),
            static_cast<int>(writer_noc),
            static_cast<int>(mux_noc),
            no_row_group ? "off" : (row_grouped ? "on" : "fallback"),
            iter_packets);
    }

    for (uint32_t link = 0; link < num_links; ++link) {
        for (uint32_t w = 0; w < workers_per_dir; ++w) {
            const uint32_t slice_idx = (link * workers_per_dir) + w;
            const uint32_t elements_per_slice = num_elements / total_slices;
            const uint32_t remainder = num_elements % total_slices;
            const uint32_t slice_start = (slice_idx * elements_per_slice) + std::min(slice_idx, remainder);
            const uint32_t slice_count = elements_per_slice + (slice_idx < remainder ? 1 : 0);
            const uint32_t half = slice_count / 2;
            const uint32_t iter_units = slice_count * geo.element_units;
            const uint64_t slice_packets = tt::div_up(uint64_t{iter_units} * unit_bytes, uint64_t{packet_cap});
            const uint64_t cadence_packets = cadence_for(slice_packets);
            const uint32_t data_valid_cadence = tt::div_up(cadence_packets, uint64_t{packets_per_slot});

            // Both directions (dir: 0 = forward, 1 = backward). mirror_core is this core's coords, reused as the
            // data_valid_sem target on the neighbor's mirror core; partner_core is the opposite-direction worker
            // (same index w), the barrier_sem target.
            for (uint32_t dir = 0; dir < num_directions; ++dir) {
                const bool is_forward = (dir == 0);
                const CoreCoord core = worker_core(link, dir, w);
                const CoreCoord partner = worker_core(link, 1 - dir, w);
                const CoreCoord mirror_core = mesh_device->worker_core_from_logical_core(core);
                const CoreCoord partner_core = mesh_device->worker_core_from_logical_core(partner);
                const auto neighbor = dir_neighbor(dir);

                const uint32_t stripe_step = is_forward ? num_devices - 1 : 1;
                const uint32_t num_iters = is_forward ? fwd_iters : bwd_iters;
                const uint32_t num_recv =
                    is_ring ? num_devices / 2 : (is_forward ? device_idx : num_devices - 1 - device_idx);
                const bool do_local_write = is_forward ? (fwd_iters > 0) : (fwd_iters == 0);

                // The last iteration walks [final_skip, final_skip + final_count) of the slice: on an even ring the
                // antipode stripe as a half, the forward direction the first.
                uint32_t final_skip = 0;
                uint32_t final_count = num_iters == 0 ? 0 : slice_count;
                if (ring_even_split) {
                    final_skip = is_forward ? 0 : half;
                    final_count = is_forward ? half : (slice_count - half);
                }

                // Units the upstream delivers into our output (relayed full stripes + sink). The even-ring
                // antipode arrives as a half, so it contributes final_count instead of a full stripe.
                const uint32_t total_units =
                    num_recv * iter_units - (ring_even_split ? (slice_count - final_count) * geo.element_units : 0);

                std::vector<uint32_t> reader_rt_args = {
                    0u,                        // input tensor address (Buffer* binding)
                    0u,                        // output tensor address (Buffer* binding)
                    device_idx,                // this device's index (initial stripe)
                    stripe_step,               // stripe index step per iteration
                    num_iters,                 // iterations this direction runs
                    total_units,               // units upstream delivers (completion wait)
                    slice_start,               // this worker's slice start (elements)
                    slice_count,               // this worker's slice length (elements)
                    final_skip,                // last-iteration slice skip (even-ring split)
                    final_count,               // last-iteration slice length (even-ring split)
                    do_local_write ? 1u : 0u,  // write local data into local output on iteration 0
                    data_valid_sem
                        .address(),  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
                };
                emplace_runtime_args_with_leading_buffers(
                    desc.kernels[reader_kernel_id],
                    core,
                    {input_tensor.buffer(), output_tensor.buffer()},
                    reader_rt_args);

                // route for Fabric_2D
                const auto route_node =
                    neighbor.has_value() ? mesh_device->get_fabric_node_id(*neighbor) : sender_fabric_node_id;
                std::vector<uint32_t> writer_rt_args = {
                    num_iters,    // iterations this direction runs
                    slice_count,  // this worker's slice length (elements)
                    final_count,  // last-iteration slice length (even-ring split)
                    barrier_sem
                        .address(),  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
                    data_valid_sem
                        .address(),  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
                    (uint32_t)partner_core.x,         // barrier_sem target (neighbor partner core x)
                    (uint32_t)partner_core.y,         // barrier_sem target (neighbor partner core y)
                    (uint32_t)mirror_core.x,          // data_valid_sem target (neighbor mirror core x)
                    (uint32_t)mirror_core.y,          // data_valid_sem target (neighbor mirror core y)
                    data_valid_cadence,               // signal data_valid every this many CB pages
                    (uint32_t)route_node.chip_id,     // neighbor chip id (packet header 2D route)
                    (uint32_t)(*route_node.mesh_id),  // neighbor mesh id (packet header 2D route)
                };
                TT_FATAL(num_iters == 0 || neighbor.has_value(), "an active direction must have a neighbor");
                if (num_iters > 0) {
                    // Connect this worker to its channel (== worker index w) on the direction's mux; two local
                    // semaphores back the connection.
                    const uint32_t flow_control_sem =
                        ttnn::experimental::ccl::add_semaphore_descriptor(desc, CoreRangeSet(CoreRange(core)));
                    const uint32_t teardown_sem =
                        ttnn::experimental::ccl::add_semaphore_descriptor(desc, CoreRangeSet(CoreRange(core)));
                    mux_config.append_client_connection_rt_args(
                        mesh_device->worker_core_from_logical_core(mux_core(link, dir)),
                        static_cast<uint8_t>(w),
                        tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{
                            .flow_control_sem_id = flow_control_sem, .teardown_sem_id = teardown_sem},
                        writer_rt_args);
                }
                desc.kernels[writer_kernel_id].runtime_args.emplace_back(core, std::move(writer_rt_args));
            }
        }
    }

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor AllGatherUnicastFactory::create_workload_descriptor(
    const AllGatherParams& operation_attributes,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_cores = get_available_worker_cores(operation_attributes, tensor_args.input_tensor);
    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // Kernel needs to wait to receive all remote data before exiting, and needs to wait for its neighbors to be ready
    // before sending them data.
    // Since Fabric doesn't provide such capability within kernels, we need to manually sync using global semaphores.
    // Allocate the semaphore in L1_SMALL to avoid fragmenting the larger L1 memory pool.
    // Two semaphores:
    // - barrier_sem: one-shot init handshake ("I'm alive") to the neighbor.
    // - data_valid_sem: units upstream has relayed into our output (relay gate + completion).
    const auto sem_buffer_type = ttnn::ccl::prefer_l1_small_buffer_type(*mesh_device);
    ttnn::ccl::warn_if_semaphores_fall_back_to_l1(sem_buffer_type);
    workload_descriptor.semaphores.push_back(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type));
    workload_descriptor.semaphores.push_back(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type));
    const auto& barrier_sem = workload_descriptor.semaphores[0];
    const auto& data_valid_sem = workload_descriptor.semaphores[1];
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    workload_descriptor.programs.reserve(tensor_coords.coords().size());
    for (const auto& coord : tensor_coords.coords()) {
        auto desc = build_unicast_descriptor_at(
            operation_attributes, coord, tensor_args, output_tensor, barrier_sem, data_valid_sem, available_cores);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return workload_descriptor;
}

}  // namespace ttnn::operations::ccl
