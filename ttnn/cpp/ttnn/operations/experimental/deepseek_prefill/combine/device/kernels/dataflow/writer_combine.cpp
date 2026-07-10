// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/dprint.h"
#include "api/debug/device_print.h"  // [debug] DEVICE_PRINT: supports uint64_t/float for the BW summary
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/overlap_config.hpp"

#if MOCK_COMBINE_INTERNALS
// The writer-only fabric-push mock reuses the trid+header overlap pool, computes routes via
// get_route/manhattan_distance (1D only), and needs a fabric destination. Fail loudly if the build
// configuration can't support it rather than silently falling back to the real path.
#if !OVERLAPING_TOKEN_WRITE
#error "MOCK_COMBINE_INTERNALS requires OVERLAPING_TOKEN_WRITE=1 (it reuses the trid+header overlap pool)"
#endif
#ifdef FABRIC_2D
#error "MOCK_COMBINE_INTERNALS is 1D-fabric only (uses get_route/manhattan_distance)"
#endif
#ifndef DEST_CHIP_ID
#error "MOCK_COMBINE_INTERNALS requires a multi-chip build (DEST_CHIP_ID)"
#endif
#endif

// FABRIC_2D vs 1D dispatch is handled portably via ccl_routing_utils::fabric_set_line_unicast_route
// (templated on packet-header type). Under 1D the helper consumes route_info.distance_in_hops,
// under 2D it consumes route_info.dst_chip_id + dst_mesh_id. The 2D fabric_route (EDM index)
// still has to be recomputed from dest_mesh_ids/dest_chip_ids — see note in writer_dispatch.cpp.

// [debug] eRISC fabric-telemetry mailbox: FabricTelemetry struct (for offsetof(scratch)) and the
// ACTIVE_ETH telemetry L1 base address the router uses.
#include "hostdev/fabric_telemetry_msgs.h"
#include "dev_mem_map.h"

#define ENABLE_COMBINE_DEBUG 0
#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_COMBINE(...)
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile Time Args =====
    // CB IDs (indices 0-4)
    constexpr uint32_t cb_dispatched_buffer_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dispatched_metadata_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_experts_tok_counter_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(4);

    // Page counts (indices 5-8)
    constexpr uint32_t dispatched_buffer_pages = get_compile_time_arg_val(5);
    constexpr uint32_t dispatched_metadata_pages = get_compile_time_arg_val(6);
    constexpr uint32_t experts_tok_counter_pages = get_compile_time_arg_val(7);
    constexpr uint32_t output_pages = get_compile_time_arg_val(8);

    // Page sizes (indices 9-12)
    constexpr uint32_t dispatched_buffer_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t dispatched_metadata_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t experts_tok_counter_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(12);

    // Operation parameters (indices 13-16)
    constexpr uint32_t num_chips = get_compile_time_arg_val(13);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(14);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(15);
    constexpr uint32_t seq_len_per_chip = get_compile_time_arg_val(16);

    // Hidden dimension (index 17)
    constexpr uint32_t hidden_size = get_compile_time_arg_val(17);

    // Aligned page sizes (indices 18-21)
    constexpr uint32_t aligned_dispatched_buffer_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t aligned_dispatched_metadata_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t aligned_experts_tok_counter_page_size = get_compile_time_arg_val(20);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(21);

    // Mesh information (indices 22-26)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(22);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(23);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(24);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(25);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(26);

    // Fabric configuration (indices 27-30)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(27);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(28);
    constexpr uint32_t num_links = get_compile_time_arg_val(29);
    [[maybe_unused]] constexpr tt::tt_fabric::Topology topology =
        (tt::tt_fabric::Topology)get_compile_time_arg_val(30);  // used by the FABRIC_1D #else handshake

    // Batch configuration (index 31)
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(31);
    // Number of dispatch groups (index 32)
    constexpr uint32_t num_dispatch_groups = get_compile_time_arg_val(32);

    // Expert region offsets tensor metadata (indices 33-36)
    constexpr uint32_t cb_expert_region_offsets_id = get_compile_time_arg_val(33);
    constexpr uint32_t expert_region_offsets_pages = get_compile_time_arg_val(34);
    constexpr uint32_t expert_region_offsets_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_expert_region_offsets_page_size = get_compile_time_arg_val(36);

    // Index 37 (max_dispatch_buffer_token_size) is consumed by reader_combine only;
    // writer_combine skips over it and continues with TensorAccessorArgs at index 38.

    // TensorAccessorArgs for all 5 tensors (starting at index 38)
    constexpr auto dispatched_buffer_args = TensorAccessorArgs<38>();
    constexpr auto dispatched_metadata_args =
        TensorAccessorArgs<dispatched_buffer_args.next_compile_time_args_offset()>();
    constexpr auto experts_tok_counter_args =
        TensorAccessorArgs<dispatched_metadata_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<experts_tok_counter_args.next_compile_time_args_offset()>();

    // ===== Runtime Args =====
    size_t rt_args_idx = 0;
    uint32_t dispatched_buffer_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t dispatched_metadata_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t experts_tok_counter_addr = get_arg_val<uint32_t>(rt_args_idx++);
    rt_args_idx++;  // expert_region_offsets_addr — consumed by reader only
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_init_complete_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    // Separate semaphore for the exit handshake. Reusing init_semaphore_address
    // for both phases is racy
    uint32_t exit_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_init_barrier_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t expert_start_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t expert_end_idx = get_arg_val<uint32_t>(rt_args_idx++);

    uint32_t output_init_complete_semaphore_address = get_semaphore(output_init_complete_semaphore_id);
    uint32_t output_init_barrier_l1_offset = get_semaphore(output_init_barrier_semaphore_id);

    // Read NOC coordinates for all cores (for inter-core barrier signaling).
    // num_cores = effective_num_links = min(num_links, 4).
    constexpr uint32_t MAX_WORKER_CORES = 4;
    ASSERT(num_cores <= MAX_WORKER_CORES);
    uint64_t all_core_barrier_noc_addrs[MAX_WORKER_CORES];
    for (uint32_t c = 0; c < num_cores; c++) {
        uint32_t noc_x = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t noc_y = get_arg_val<uint32_t>(rt_args_idx++);
        all_core_barrier_noc_addrs[c] = get_noc_addr(noc_x, noc_y, output_init_barrier_l1_offset);
    }

    // [debug] per-sender index (RT arg appended by the program factory); used for the eRISC combine marker.
    [[maybe_unused]] const uint32_t combine_sender_index = get_arg_val<uint32_t>(rt_args_idx++);

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t combine_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t combine_devices = num_chips;
#endif

    DPRINT_COMBINE(
        "Combine Writer: experts=[{}, {}) linearized_mesh_coord={}\n",
        expert_start_idx,
        expert_end_idx,
        linearized_mesh_coord);

    // [debug] BW measurement window: wall clock is captured just before this sender writes the START
    // marker to its eth core and just after the END marker; bw_total_payload_bytes accumulates every
    // payload byte injected into fabric in between. Reported as achieved BW at kernel exit (see END
    // marker block). Declared at function scope so both marker sites can reach them; only populated
    // under DEST_CHIP_ID.
    [[maybe_unused]] uint64_t bw_t_start = 0;
    [[maybe_unused]] uint64_t bw_total_payload_bytes = 0;

    // [debug] Inter-step gap histogram. We snapshot the wall clock at every step of the send pipeline
    // (before the START signal, before each fabric send, before and after the STOP signal) and bin the
    // delta since the previous snapshot. Expectation under healthy flow: every gap lands in the two
    // smallest buckets. Fat tail => the sender is stalling (e.g. spinning in wait_for_empty_write_slot).
    // Thresholds are in cycles @ 1.35 GHz: 100ns=135, 1us=1350, 10us=13500, 100us=135000, 1ms=1350000,
    // 10ms=13500000.
    [[maybe_unused]] uint64_t hist_last_ts = 0;
    [[maybe_unused]] uint32_t hist_buckets[7] = {0, 0, 0, 0, 0, 0, 0};

    // [debug] Outlier token capture. The handful of tokens whose per-token gap lands in the 1-10ms
    // bucket or higher (d >= 1ms = 1350000 cycles @ 1.35 GHz, i.e. hist_buckets[5] or [6]) dominate the
    // total send time and thus dictate achieved BW, so record which tokens they are to probe them.
    // We keep only the first MAX_OUTLIERS such indices. hist_token_idx is the running per-token counter
    // (incremented once per real token in the send loop). Token index 0 is skipped: its gap is measured
    // against the uninitialized hist_last_ts=0 baseline, so it is a fixed artifact, not a real stall.
    static constexpr uint32_t MAX_OUTLIERS = 10;
    [[maybe_unused]] uint32_t hist_token_idx = 0;
    [[maybe_unused]] uint32_t outlier_token_indices[MAX_OUTLIERS] = {0};
    [[maybe_unused]] uint32_t outlier_count = 0;

    [[maybe_unused]] auto hist_record = [&hist_last_ts, &hist_buckets, &outlier_token_indices, &outlier_count](
                                            uint64_t now, uint32_t token_idx) {
        const uint64_t d = now - hist_last_ts;
        hist_last_ts = now;
        if (d < 135ULL) {
            hist_buckets[0]++;
        } else if (d < 1350ULL) {
            hist_buckets[1]++;
        } else if (d < 13500ULL) {
            hist_buckets[2]++;
        } else if (d < 135000ULL) {
            hist_buckets[3]++;
        } else if (d < 1350000ULL) {
            hist_buckets[4]++;
        } else if (d < 13500000ULL) {
            hist_buckets[5]++;
        } else {
            hist_buckets[6]++;
        }
        // [debug] gap >= 1ms → 1-10ms bucket ([5]) or higher ([6]); stash the token index. Skip
        // token 0 (baseline artifact) and stop after the first MAX_OUTLIERS.
        if (d >= 1350000ULL && token_idx != 0 && outlier_count < MAX_OUTLIERS) {
            outlier_token_indices[outlier_count++] = token_idx;
        }
    };

#if MOCK_COMBINE_INTERNALS
    // [debug] Mock-only decomposition of per-token time into the two remaining stall sources:
    //   mock_wait_buckets: time in the LAGGED wait_on_flush_for_trid (completion of the send issued
    //                      OVERLAP_POOL_DEPTH tokens ago). Should collapse into the smallest bucket
    //                      once the pipeline is primed — that means the overlap fully hides write
    //                      completion.
    //   mock_send_buckets: time in the send call itself, dominated by wait_for_empty_write_slot()
    //                      (waiting for a fabric slot to drain). If this is where the time goes and
    //                      mock_wait is ~0, we are truly only blocking on fabric slots.
    // Same cycle thresholds @ 1.35 GHz as hist_record.
    uint32_t mock_wait_buckets[7] = {0, 0, 0, 0, 0, 0, 0};
    uint32_t mock_send_buckets[7] = {0, 0, 0, 0, 0, 0, 0};
    auto mock_bin = [](uint32_t* b, uint64_t d) {
        if (d < 135ULL) {
            b[0]++;
        } else if (d < 1350ULL) {
            b[1]++;
        } else if (d < 13500ULL) {
            b[2]++;
        } else if (d < 135000ULL) {
            b[3]++;
        } else if (d < 1350000ULL) {
            b[4]++;
        } else if (d < 13500000ULL) {
            b[5]++;
        } else {
            b[6]++;
        }
    };
#endif

#if INIT_ZEROS
    // Wait for reader to complete output-zeroing
    volatile tt_l1_ptr uint32_t* output_init_complete_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_init_complete_semaphore_address);
    noc_semaphore_wait(output_init_complete_sem_ptr, 1);
    noc_semaphore_set(output_init_complete_sem_ptr, 0);
#endif

#ifdef DEST_CHIP_ID
    constexpr uint32_t total_mesh_devices = mesh_rows * mesh_cols;
    constexpr uint8_t dest_chip_ids[total_mesh_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[total_mesh_devices] = DEST_MESH_ID;

    uint32_t packet_header_buffer_address = get_read_ptr(cb_packet_header_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);

#ifdef FABRIC_2D
    // Portable FABRIC_2D connections: one RoutingPlaneConnectionManager slot per combine-axis
    // neighbor (each a distinct fabric direction), opened on its correct forwarding routing plane by
    // the host. Required for multi-hop forwarding along the combine axis; the legacy fixed-link
    // array connection only forwards a single hop (deadlocks on e.g. the 4-device column of a 4x2
    // mesh). FABRIC_1D keeps the legacy array + per-target unicast handshake in the #else branch.
    static_assert(axis != ReplicateGroup::NONE, "FABRIC_2D combine requires a concrete cluster_axis");
    uint32_t num_connections = get_arg_val<uint32_t>(rt_args_idx++);
    // sem_route_id holds the multicast handshake headers (non-ring path); the ring path uses
    // per-destination mesh-API unicast and reuses unicast_packet_header instead.
    uint8_t sem_route_id = PacketHeaderPool::allocate_header_n(static_cast<uint8_t>(num_connections));
    auto fabric_connections = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION>(rt_args_idx, num_connections);

    // dir_to_slot maps a fabric routing direction -> the connection slot opened in that direction.
    // DIR_TO_SLOT_EMPTY (> MaxConnections, so never a valid slot index) marks "no connection opened in
    // this direction"; it must never be used as a slot index at send time.
    constexpr uint8_t DIR_TO_SLOT_EMPTY = 0xFF;
    uint8_t dir_to_slot[eth_chan_directions::COUNT];
    for (uint32_t d = 0; d < static_cast<uint32_t>(eth_chan_directions::COUNT); ++d) {
        dir_to_slot[d] = DIR_TO_SLOT_EMPTY;
    }
    for (uint32_t i = 0; i < num_connections; ++i) {
        dir_to_slot[fabric_connections.get_tag(i)] = static_cast<uint8_t>(i);
    }

    // Bidirectional multicast handshake (hs_ = handshake): each device increments the init/exit
    // semaphore on every other combine-axis device exactly once. Per-slot hop range is chosen by THIS
    // device's position (hs_axis_pos) along the combine axis: the forward slot covers the (len-1-pos)
    // devices ahead, the backward slot the pos devices behind — together exactly combine_devices-1
    // peers. The neighbor's position only selects which slot (forward vs backward), not the range.
    // (FABRIC_2D combine is always a Linear axis here; no FABRIC_2D ring config exists.)
    constexpr bool hs_is_cols = (axis == ReplicateGroup::COLS);
    constexpr uint32_t hs_axis_pos =
        hs_is_cols ? (linearized_mesh_coord / mesh_cols) : (linearized_mesh_coord % mesh_cols);
    constexpr uint32_t hs_axis_len = hs_is_cols ? mesh_rows : mesh_cols;
    constexpr uint32_t hs_pos_range = (hs_axis_len - 1) - hs_axis_pos;  // devices ahead on the axis
    constexpr uint32_t hs_neg_range = hs_axis_pos;                      // devices behind on the axis

    uint8_t hs_starts[tt::tt_fabric::RoutingPlaneConnectionManager::MaxConnections];
    uint8_t hs_ranges[tt::tt_fabric::RoutingPlaneConnectionManager::MaxConnections];
    for (uint32_t i = 0; i < num_connections; ++i) {
        const uint16_t nbr_chip = fabric_connections.get(i).dst_dev_id;
        const uint16_t nbr_mesh = fabric_connections.get(i).dst_mesh_id;
        // Reverse-map this neighbor's (mesh, chip) to its linearized mesh index to read its axis
        // position. The neighbor must be one of this device's combine targets (host guarantees it).
        uint32_t nbr_lin = 0;
        bool nbr_found = false;
        for (uint32_t d = 0; d < total_mesh_devices; ++d) {
            if (dest_chip_ids[d] == nbr_chip && dest_mesh_ids[d] == nbr_mesh) {
                nbr_lin = d;
                nbr_found = true;
                break;
            }
        }
        ASSERT(nbr_found);  // a connection neighbor missing from dest_chip_ids => mis-sized/garbled rt args
        const uint32_t nbr_axis_pos = hs_is_cols ? (nbr_lin / mesh_cols) : (nbr_lin % mesh_cols);
        const bool is_forward = (nbr_axis_pos > hs_axis_pos);
        hs_starts[i] = 1;  // start_distance=1: skip the issuing device, deliver from its first hop onward
        hs_ranges[i] = is_forward ? static_cast<uint8_t>(hs_pos_range) : static_cast<uint8_t>(hs_neg_range);
    }

    // flush=true holds the receiver EDM's atomic-inc until this sender's prior fabric writes to that chip
    // commit (needed at exit to order the inc behind the combine payload). init passes flush=false: it is
    // the first fabric traffic, so there are no prior writes to order against — matching the FABRIC_1D path.
    auto combine_2d_mcast_handshake = [&](uint64_t sem_addr, bool flush) {
        const auto cmd = tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_addr, 1, flush};
        tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_atomic_inc(
            fabric_connections, sem_route_id, cmd, hs_starts, hs_ranges);
    };
#else
    constexpr std::array<bool, 4> directions = DIRECTIONS;
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_args_idx);
    auto* sem_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));
    open_direction_connections_barrier(directions, fabric_connections);
#endif

    // Init semaphore exchange
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
#ifdef FABRIC_2D
    combine_2d_mcast_handshake(init_noc_semaphore_addr, /*flush=*/false);
#else
    send_init_semaphore_to_configured_targets<
        linearized_mesh_coord,
        topology,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        axis,
        total_mesh_devices>(
        fabric_connections, sem_packet_header, dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);
#endif

    volatile tt_l1_ptr uint32_t* init_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
    noc_semaphore_wait(init_sem_ptr, combine_devices - 1);
    noc_semaphore_set(init_sem_ptr, 0);

    DPRINT_COMBINE("Fabric setup complete\n");

    // [debug] Mark START of the combine data-send loop in each connected eth router's telemetry scratch[0].
    // Value = 100 + chip*10 + sender_index so eRISC logs can correlate to this sender without host-side
    // mapping. Single inline dword write (scratch[0] is 4-byte aligned); the router reads it locally.
    constexpr uint32_t combine_marker_l1_addr = MEM_AERISC_FABRIC_TELEMETRY_BASE + offsetof(FabricTelemetry, scratch);
    const uint32_t combine_marker_value = 100 + src_chip_id * 10 + combine_sender_index;
    for (uint32_t d = 0; d < 4; d++) {
        if (directions[d]) {
            noc_inline_dw_write<InlineWriteDst::L1>(
                get_noc_addr(fabric_connections[d].edm_noc_x, fabric_connections[d].edm_noc_y, combine_marker_l1_addr),
                combine_marker_value);
        }
    }
    // DPRINT << "[combine-marker] writer START value=" << combine_marker_value << ENDL();

    // [debug][cmb-place] Log this sender's placement + the eth (EDM) cores it injects tokens into, so
    // the combine placement can be reconstructed from HW. Coords are VIRTUAL (translated) NOC space:
    // my_x/my_y and the fabric adapter's edm_noc_x/y both live here, so each eth_virt below joins to
    // the [cmb-place eth] self_virt that eth core prints (which carries its logical coords). send_noc =
    // the NoC the sender->EDM payload writes use (the fabric worker NoC, == noc_index for combine).
    DEVICE_PRINT(
        "[cmb-place sender] idx={} self_virt=({},{}) self_logical=({},{}) send_noc={}\n",
        combine_sender_index,
        (uint32_t)my_x[noc_index],
        (uint32_t)my_y[noc_index],
        (uint32_t)get_absolute_logical_x(),
        (uint32_t)get_absolute_logical_y(),
        (uint32_t)noc_index);
    for (uint32_t d = 0; d < 4; d++) {
        if (directions[d]) {
            DEVICE_PRINT(
                "[cmb-place sender]   idx={} dir={} eth_virt=({},{})\n",
                combine_sender_index,
                d,
                (uint32_t)fabric_connections[d].edm_noc_x,
                (uint32_t)fabric_connections[d].edm_noc_y);
        }
    }
#endif

#if INIT_ZEROS
    // Signal ALL readers that global init exchange is done.
    // Each writer increments every reader's barrier sem so each reader
    // collects num_cores signals before proceeding.
    for (uint32_t c = 0; c < num_cores; c++) {
        noc_semaphore_inc(all_core_barrier_noc_addrs[c], 1);
    }
    noc_async_atomic_barrier();
#endif

    const auto output_addr_gen = TensorAccessor(output_args, output_addr);

#if OVERLAPING_TOKEN_WRITE && defined(DEST_CHIP_ID)
    // Lockstep trid + header pools (stage 3 / gap#1). The header pool lives right after the 2 base
    // headers (unicast + handshake) in c_5. Each token uses header pool[overlap_slot] and trid
    // (overlap_slot + 1), cycling overlap_slot over [0, OVERLAP_POOL_DEPTH). The wait is still
    // adjacent here (one in flight), so cycling is functionally a no-op — it sets up stage 4, where
    // the wait lags by the pool depth and these become the outstanding in-flight sends.
    const uint32_t overlap_hdr_pool_base = packet_header_buffer_address + 2u * sizeof(PACKET_HEADER_TYPE);
    uint32_t overlap_slot = 0;
#endif

#if MOCK_COMBINE_INTERNALS
    // ===== MOCK writer-only synthetic fabric send loop (CB-free) =====
    // The entire producer chain is gutted (see MOCK_COMBINE_INTERNALS in overlap_config.hpp), so we
    // fabricate tokens here and push them straight into fabric through the real trid+header overlap
    // pool. No cb_wait_front / cb_pop_front: the only per-token stalls are the LAGGED
    // wait_on_flush_for_trid and wait_for_empty_write_slot() inside the send. Everything around this
    // loop (fabric open, init/exit sem handshakes, START/END markers, drain, close) is unchanged.
    {
        // Precompute (route direction, distance-in-hops) for every peer on the 1D combine line, out
        // of the hot loop. route/distance are what reader_combine would have written into route_info.
        uint8_t mock_route[total_mesh_devices];
        uint16_t mock_distance[total_mesh_devices];
        uint32_t mock_num_peers = 0;
        for (uint32_t dst = 0; dst < total_mesh_devices; dst++) {
            if (dst == linearized_mesh_coord) {
                continue;  // never send to self
            }
            mock_route[mock_num_peers] =
                static_cast<uint8_t>(get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst));
            mock_distance[mock_num_peers] =
                static_cast<uint16_t>(manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst));
            mock_num_peers++;
        }

        // Static payload scratch: reuse the (now unused) cb_route_info L1 slot + alignment as a fixed
        // 14 KB source. Its bytes are never refreshed — content is garbage (perf only), but that means
        // there is no reader overwriting it, so the non-blocking overlapped reads are race-free.
        const uint32_t mock_payload_addr = get_write_ptr(cb_route_info_id) + l1_alignment;
        const uint32_t mock_total_tokens = mock_num_peers * MOCK_COMBINE_TOKENS_PER_DEST;

        uint32_t peer = 0;
        for (uint32_t i = 0; i < mock_total_tokens; i++) {
            ccl_routing_utils::line_unicast_route_info_t pkt_route_info{};
            pkt_route_info.distance_in_hops = mock_distance[peer];
            auto& payload_sender = fabric_connections[mock_route[peer]];
            const uint32_t output_page_idx = i % output_pages;  // spread dst pages, avoid same-addr serialize

            auto* overlap_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(
                overlap_hdr_pool_base + overlap_slot * sizeof(PACKET_HEADER_TYPE));
            const uint32_t overlap_trid = overlap_slot + 1;  // 1..OVERLAP_POOL_DEPTH (trid 0 reserved)

            const uint64_t t0 = get_timestamp();
            wait_on_flush_for_trid(overlap_trid);  // LAGGED: free this slot's DEPTH-ago use before reuse
            const uint64_t t1 = get_timestamp();

            ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_for_route_helper(overlap_hdr), pkt_route_info);
            fabric_send_noc_unicast_with_trid<fabric_max_packet_size>(
                output_addr_gen,
                payload_sender,
                overlap_hdr,
                mock_payload_addr,
                output_page_idx,
                (int)aligned_output_page_size,
                l1_alignment,
                overlap_trid);
            const uint64_t t2 = get_timestamp();

            mock_bin(mock_wait_buckets, t1 - t0);  // lagged trid-completion wait (want ~0)
            mock_bin(mock_send_buckets, t2 - t1);  // issue + wait_for_empty_write_slot (fabric slot)
            hist_record(t2, i);                    // total per-token cadence (shared histogram)

            bw_total_payload_bytes += aligned_output_page_size;
            overlap_slot = (overlap_slot + 1 == OVERLAP_POOL_DEPTH) ? 0u : overlap_slot + 1u;
            peer = (peer + 1 == mock_num_peers) ? 0u : peer + 1u;
        }
    }
#else
    {
        DeviceZoneScopedN("combine-ethernet-flow");
        //  Sentinel-terminated fabric send loop
        while (true) {
            cb_wait_front(cb_route_info_id, 1);
            uint32_t cb_base = get_read_ptr(cb_route_info_id);
            volatile tt_l1_ptr uint32_t* route_info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base);
            uint32_t route = route_info[0];
            {
                // DeviceZoneScopedN("combine-waiting-for-route-info");
                if (route == ROUTE_INFO_SENTINEL) {
                    cb_pop_front(cb_route_info_id, 1);
                    break;
                }
            }
            // [debug] Snapshot the wall clock once per token and bin the gap since the previous
            // token's send. Placed after the sentinel check so only real tokens are counted.
            // hist_token_idx identifies the token in the send stream; hist_record uses it to log
            // which tokens are the >=1ms outliers.
            hist_record(get_timestamp(), hist_token_idx);
            hist_token_idx++;

            uint32_t distance = route_info[1];
            uint32_t output_page_idx = route_info[2];
            uint32_t output_data_addr = cb_base + l1_alignment;

            DPRINT_COMBINE("Fabric send: route={} distance={} page_idx={}\n", route, distance, output_page_idx);

#ifdef DEST_CHIP_ID
            {
                // DeviceZoneScopedN("FABRIC-send");
                // CB layout (written by reader_combine): [0]=route (1D EDM index), [1]=distance_hops,
                // [2]=page_idx, [3]=dst_chip_index (2D only). Under 2D recompute the EDM direction
                // from the destination since route_info[0] is 1D-style and doesn't match the 2D index.
                ccl_routing_utils::line_unicast_route_info_t pkt_route_info{};
                uint32_t fabric_route;
                if constexpr (
                    std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader> ||
                    std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::UDMHybridMeshPacketHeader>) {
                    const uint32_t dst_chip_device_id = route_info[3];
                    pkt_route_info.dst_chip_id = dest_chip_ids[dst_chip_device_id];
                    pkt_route_info.dst_mesh_id = dest_mesh_ids[dst_chip_device_id];
                    // TODO(#46174): drop the private tt_fabric_api.h dependency once
                    // RoutingPlaneConnectionManager exposes a portable (mesh, chip) -> slot lookup.
                    fabric_route = static_cast<uint32_t>(get_next_hop_router_direction(
                        dest_mesh_ids[dst_chip_device_id], dest_chip_ids[dst_chip_device_id]));
                } else {
                    pkt_route_info.distance_in_hops = static_cast<uint16_t>(distance);
                    fabric_route = route;
                }

#ifdef FABRIC_2D
                ASSERT(dir_to_slot[fabric_route] != DIR_TO_SLOT_EMPTY);  // first-hop direction must have an open slot
                auto& payload_sender = fabric_connections.get(dir_to_slot[fabric_route]).sender;
#else
                auto& payload_sender = fabric_connections[fabric_route];
#endif
#if OVERLAPING_TOKEN_WRITE
                // Stage 4: LAGGED wait. Before reusing this slot's header + trid, wait for that slot's
                // PREVIOUS use (OVERLAP_POOL_DEPTH tokens ago) to complete. For the first DEPTH tokens
                // the trid is unused so the barrier returns immediately; thereafter it lags by DEPTH,
                // keeping up to DEPTH sends in flight. Only wait_for_empty_write_slot (inside the issue)
                // and this lagged barrier can stall.
                //
                // PERF-PROTOTYPE CONTENT CAVEAT: the CB slot is still popped per-iteration below, so the
                // reader can overwrite the payload while this non-blocking send is still reading it ->
                // torn payload bytes (wrong token CONTENT). route/distance/page are captured into
                // registers + the pooled header BEFORE the send, so routing, volume, and termination are
                // all correct — only content is wrong. Correct content needs a lagged pop + CB peek-ahead
                // (deferred; this stage is for the perf signal).
                auto* overlap_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(
                    overlap_hdr_pool_base + overlap_slot * sizeof(PACKET_HEADER_TYPE));
                const uint32_t overlap_trid = overlap_slot + 1;  // 1..OVERLAP_POOL_DEPTH (trid 0 reserved)
                wait_on_flush_for_trid(overlap_trid);            // LAGGED: free this slot's DEPTH-ago use before reuse
                ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_for_route_helper(overlap_hdr), pkt_route_info);
                fabric_send_noc_unicast_with_trid<fabric_max_packet_size>(
                    output_addr_gen,
                    payload_sender,
                    overlap_hdr,
                    output_data_addr,
                    output_page_idx,
                    (int)aligned_output_page_size,
                    l1_alignment,
                    overlap_trid);
                overlap_slot = (overlap_slot + 1 == OVERLAP_POOL_DEPTH) ? 0u : overlap_slot + 1u;
#else
                ccl_routing_utils::fabric_set_line_unicast_route(
                    pkt_hdr_for_route_helper(unicast_packet_header), pkt_route_info);
                fabric_send_noc_unicast<fabric_max_packet_size>(
                    output_addr_gen,
                    payload_sender,
                    unicast_packet_header,
                    output_data_addr,
                    output_page_idx,
                    (int)aligned_output_page_size,
                    l1_alignment);
                noc_async_writes_flushed();  // Ensure output data departed L1 before freeing CB slot
#endif

                bw_total_payload_bytes += aligned_output_page_size;
            }
#endif

            // Pop the route-info CB, which also carries the output payload (merged CB).
            cb_pop_front(cb_route_info_id, 1);
        }
    }
#endif  // MOCK_COMBINE_INTERNALS

#ifdef DEST_CHIP_ID
#if OVERLAPING_TOKEN_WRITE
    // Drain the up-to-OVERLAP_POOL_DEPTH still-in-flight overlapped sends before the exit handshake /
    // connection teardown. (noc_async_write_barrier below also drains, but be explicit per-trid.)
    for (uint32_t t = 1; t <= OVERLAP_POOL_DEPTH; t++) {
        wait_on_flush_for_trid(t);
    }
#endif
    // Defensive: drain pending local NOC writes before fabric atomic-inc traffic,
    // so the exit-sem signal cannot reach peers ahead of the last data writes.
    noc_async_write_barrier();

    // [debug] Mark END of the combine data-send loop: write 0 to the eth router's telemetry scratch[0]
    // (after the barrier above, so the last data packet has departed L1). Reuses combine_marker_l1_addr.
    for (uint32_t d = 0; d < 4; d++) {
        if (directions[d]) {
            noc_inline_dw_write<InlineWriteDst::L1>(
                get_noc_addr(fabric_connections[d].edm_noc_x, fabric_connections[d].edm_noc_y, combine_marker_l1_addr),
                0);
        }
    }
    // DPRINT << "[combine-marker] writer END value=0" << ENDL();

    // Exit semaphore exchange on a dedicated semaphore (exit_semaphore_address). The exit-inc must not
    // be observed before our prior fabric writes to that chip have landed, or a peer could see
    // sem-reached-threshold before the data is in DRAM. The two paths enforce this differently:
    //   - FABRIC_2D: the noc_async_write_barrier() above drains local writes, and the 2D multicast
    //     handshake is issued with flush=true so the receiver EDM also holds the atomic-inc until our
    //     prior fabric writes commit (init uses flush=false: no prior writes to order against).
    //   - FABRIC_1D (#else): send_init_semaphore_to_configured_targets is called with /*flush=*/true so
    //     the receiver EDM holds the atomic-inc until our prior writes commit (the init handshake uses
    //     the default flush=false since it has no prior writes to order against).
    {
        const uint64_t exit_noc_semaphore_addr = get_noc_addr(exit_semaphore_address);
#ifdef FABRIC_2D
        combine_2d_mcast_handshake(exit_noc_semaphore_addr, /*flush=*/true);
#else
        send_init_semaphore_to_configured_targets<
            linearized_mesh_coord,
            topology,
            src_chip_id,
            mesh_rows,
            mesh_cols,
            axis,
            total_mesh_devices>(
            fabric_connections,
            sem_packet_header,
            dest_chip_ids,
            dest_mesh_ids,
            exit_noc_semaphore_addr,
            /*flush=*/true);
#endif

        volatile tt_l1_ptr uint32_t* exit_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(exit_semaphore_address);
        noc_semaphore_wait(exit_sem_ptr, combine_devices - 1);
        noc_semaphore_set(exit_sem_ptr, 0);
    }

    // The atomic-inc handshake helpers (FABRIC_2D multicast and FABRIC_1D per-target unicast) send
    // via send_payload_flush_non_blocking_from_address — that confirms the write departed the worker's
    // NIU but not that the bytes landed in the EDM's L1 inbox. If we exit and close the fabric
    // connections while a send is mid-flight, the EDM might process its slot bookkeeping before the
    // bytes arrive. A full barrier ensures all writes (and atomics, defensively) complete before close.
    noc_async_full_barrier();

#ifdef FABRIC_2D
    fabric_connections.close();
#else
    close_direction_connections(directions, fabric_connections);
#endif
#endif

    // [debug] Dump the per-token inter-step gap histogram. Under healthy flow all counts land in the
    // two smallest buckets; a fat tail means the sender is stalling. Thresholds @ 1.35 GHz (see above).
    DPRINT(
        "[combine-hist] sender per-token gap histogram (counts): "
        "<100ns={} <1us={} <10us={} <100us={} <1ms={} <10ms={} >=10ms={}\n",
        hist_buckets[0],
        hist_buckets[1],
        hist_buckets[2],
        hist_buckets[3],
        hist_buckets[4],
        hist_buckets[5],
        hist_buckets[6]);

    // [debug] Dump the token indices of the outliers captured above (per-token gap >= 1ms, i.e. the
    // 1-10ms bucket or higher). Up to the first MAX_OUTLIERS; token index 0 is excluded (its gap is
    // measured against an uninitialized baseline, so it is not a real measurement). These are the
    // tokens to probe next.
    DPRINT("[combine-hist] outlier tokens (per-token gap>=1ms), count={}:\n", outlier_count);
    for (uint32_t i = 0; i < outlier_count; i++) {
        DPRINT("  outlier[{}] token_idx={}\n", i, outlier_token_indices[i]);
    }

#if MOCK_COMBINE_INTERNALS
    // [debug] Mock decomposition. If mock-wait is essentially all in <100ns/<1us while mock-send
    // carries the mass, the OVERLAP_POOL_DEPTH pipeline is hiding write completion and the residual
    // stall is purely wait_for_empty_write_slot() — i.e. we are only blocking on fabric slots.
    DPRINT(
        "[combine-mock] LAGGED trid-wait gap histogram (counts): "
        "<100ns={} <1us={} <10us={} <100us={} <1ms={} <10ms={} >=10ms={}\n",
        mock_wait_buckets[0],
        mock_wait_buckets[1],
        mock_wait_buckets[2],
        mock_wait_buckets[3],
        mock_wait_buckets[4],
        mock_wait_buckets[5],
        mock_wait_buckets[6]);
    DPRINT(
        "[combine-mock] SEND-call gap histogram (incl. wait_for_empty_write_slot) (counts): "
        "<100ns={} <1us={} <10us={} <100us={} <1ms={} <10ms={} >=10ms={}\n",
        mock_send_buckets[0],
        mock_send_buckets[1],
        mock_send_buckets[2],
        mock_send_buckets[3],
        mock_send_buckets[4],
        mock_send_buckets[5],
        mock_send_buckets[6]);
#endif
}
