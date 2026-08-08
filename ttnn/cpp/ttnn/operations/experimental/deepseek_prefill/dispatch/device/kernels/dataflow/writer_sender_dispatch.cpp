// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Sender RISCV_0 kernel — fabric writer.
//
// Drains the worker baton-ring CBs and sends tokens and metadata via fabric to
// their destination chips. Reads payload/metadata/route-info from the per-worker
// ring CBs whose base addresses are exchanged during the addr handshake.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/dprint.h"
#include "api/debug/assert.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

// FABRIC_2D vs 1D dispatch is handled portably via ccl_routing_utils::fabric_set_line_unicast_route
// (templated on packet-header type). Under 1D the helper consumes route_info.distance_in_hops,
// under 2D it consumes route_info.dst_chip_id + dst_mesh_id. The 2D fabric_route (EDM index)
// still has to be recomputed from dest_mesh_ids/dest_chip_ids because the reader's 1D-style
// route_info[0] doesn't match the 2D physical EDM index.

#define ENABLE_DISPATCH_DEBUG 0

#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_DISPATCH(...)
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile Time Args =====
    // CB IDs (indices 0-8)
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_payload_for_writer_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_metadata_for_writer_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_metadata_temp_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_dispatch_table_id = get_compile_time_arg_val(8);

    // Page counts (indices 9-14)
    constexpr uint32_t input_pages = get_compile_time_arg_val(9);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(10);
    constexpr uint32_t offsets_pages = get_compile_time_arg_val(11);
    constexpr uint32_t output_pages = get_compile_time_arg_val(12);
    constexpr uint32_t metadata_pages = get_compile_time_arg_val(13);
    constexpr uint32_t dispatch_table_pages = get_compile_time_arg_val(14);

    // Page sizes (indices 15-20)
    constexpr uint32_t input_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(16);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(17);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t dispatch_table_page_size = get_compile_time_arg_val(20);

    // Operation parameters (indices 21-27; only num_devices/hidden_size used by the writer)
    constexpr uint32_t num_devices = get_compile_time_arg_val(21);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(22);

    // Mesh information (indices 28-32)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(28);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(29);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(30);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(31);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(32);

    // Aligned page sizes (indices 33-38)
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(33);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(34);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(36);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(37);
    constexpr uint32_t aligned_dispatch_table_page_size = get_compile_time_arg_val(38);

    // Fabric configuration (indices 39-42)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(39);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(40);
    constexpr uint32_t num_links = get_compile_time_arg_val(41);
    [[maybe_unused]] constexpr tt::tt_fabric::Topology topology =
        (tt::tt_fabric::Topology)get_compile_time_arg_val(42);  // used by the FABRIC_1D #else handshake

    // TensorAccessorArgs for all 6 tensors (starting at index 45)
    constexpr auto input_args = TensorAccessorArgs<45>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto dispatch_table_args =
        TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();

    constexpr uint32_t writer_extra_args_base = dispatch_table_args.next_compile_time_args_offset();
    constexpr uint32_t writer_cb_size = get_compile_time_arg_val(writer_extra_args_base + 0);
    constexpr uint32_t num_workers = get_compile_time_arg_val(writer_extra_args_base + 1);
#ifdef SPARSE_MCAST_DISPATCH
    // Grouped record (must match dispatch_program_factory.cpp grouped_route_info_u32 and the layout
    // written by writer_worker_dispatch): [dir, num_dests, token_idx, page[4], dist[4], expert[4],
    // k[4], weight[4]] = 23 u32, aligned up to the L1 alignment for the ring slot stride.
    constexpr uint32_t grouped_route_info_u32 = 23;
    constexpr uint32_t route_info_slot_stride =
        ((grouped_route_info_u32 * 4u + l1_alignment - 1) / l1_alignment) * l1_alignment;
#else
    constexpr uint32_t route_info_slot_stride = l1_alignment;
#endif

    // ===== Runtime Args =====
    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t offsets_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t dispatch_table_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t cross_device_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t dispatch_core_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_dispatch_cores = get_arg_val<uint32_t>(rt_args_idx++);
    // Separate semaphore for the exit handshake. Reusing init_semaphore_address
    // for both phases is racy: a fast partner's exit-inc can land inside the
    // post-init noc_semaphore_set(0) window and get wiped, deadlocking the
    // pair on dispatch_devices==2 (mesh-2x4 column pair). Mirrors the combine fix.
    uint32_t exit_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);

    uint32_t addr_ready_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t cross_addr_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t space_avail_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);

    uint32_t ring_route_cb[num_workers];
    uint32_t ring_payload_cb[num_workers];
    uint32_t ring_meta_cb[num_workers];
    uint32_t ring_noc_x[num_workers];
    uint32_t ring_noc_y[num_workers];
    uint32_t ring_data_avail_id[num_workers];
    for (uint32_t s = 0; s < num_workers; s++) {
        ring_route_cb[s] = get_arg_val<uint32_t>(rt_args_idx++);
        ring_payload_cb[s] = get_arg_val<uint32_t>(rt_args_idx++);
        ring_meta_cb[s] = get_arg_val<uint32_t>(rt_args_idx++);
        ring_noc_x[s] = get_arg_val<uint32_t>(rt_args_idx++);
        ring_noc_y[s] = get_arg_val<uint32_t>(rt_args_idx++);
        ring_data_avail_id[s] = get_arg_val<uint32_t>(rt_args_idx++);
    }

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t dispatch_devices = num_devices;
#endif

    DPRINT_DISPATCH(
        "Writer kernel: dispatch_core={} / {} dispatch_devices={}\n",
        dispatch_core_idx,
        num_dispatch_cores,
        dispatch_devices);

    uint32_t ring_route_base[num_workers];
    uint32_t ring_payload_base[num_workers];
    uint32_t ring_meta_base[num_workers];
    Noc noc;
    Semaphore<> addr_ready_sem(addr_ready_semaphore_id);
    uint32_t cross_addr_sem_l1_offset = get_semaphore(cross_addr_semaphore_id);
    for (uint32_t s = 0; s < num_workers; s++) {
        ring_route_base[s] = get_write_ptr(ring_route_cb[s]);
        ring_payload_base[s] = get_write_ptr(ring_payload_cb[s]);
        ring_meta_base[s] = get_write_ptr(ring_meta_cb[s]);
        uint64_t mailbox = get_noc_addr(ring_noc_x[s], ring_noc_y[s], cross_addr_sem_l1_offset);
        noc_inline_dw_write(mailbox + 0 * sizeof(uint32_t), ring_route_base[s]);
        noc_inline_dw_write(mailbox + 1 * sizeof(uint32_t), ring_payload_base[s]);
        noc_inline_dw_write(mailbox + 2 * sizeof(uint32_t), ring_meta_base[s]);
        noc_async_write_barrier();  // all three addresses must land before addr_ready wakes worker s
        addr_ready_sem.up(noc, ring_noc_x[s], ring_noc_y[s], 1);
        noc_async_atomic_barrier();
        DPRINT_DISPATCH("Sender writer: addr handshake done ring={} u=({},{})\n", s, ring_noc_x[s], ring_noc_y[s]);
    }

#ifdef DEST_CHIP_ID
    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;

    uint32_t packet_header_buffer_address = get_read_ptr(cb_packet_header_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);

#ifdef FABRIC_2D
    // Portable FABRIC_2D connections: one RoutingPlaneConnectionManager slot per physical first-hop
    // direction needed by the dispatch group, opened on its correct routing plane by the host.
    // Required for multi-hop forwarding along the dispatch axis; the legacy fixed-link
    // array connection only forwards a single hop (deadlocks on e.g. the 4-device column of a 4x2
    // mesh). FABRIC_1D keeps the legacy array + per-target unicast handshake in the #else branch.
    static_assert(axis != ReplicateGroup::NONE, "FABRIC_2D dispatch requires a concrete cluster_axis");
    uint32_t num_connections = get_arg_val<uint32_t>(rt_args_idx++);
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

    // A logical dispatch axis can turn through the physical Fabric2D graph (for example, an 8x1 group
    // embedded in a 2x4 LoudBox). A multi-hop multicast range cannot describe that path reliably.
    // Explicitly unicast one semaphore increment to every peer in this dispatch group; each packet uses
    // the routing plane selected for its actual first hop and carries its final (mesh, chip) destination.
    constexpr uint32_t dispatch_group_begin = axis == ReplicateGroup::COLS
                                                  ? linearized_mesh_coord % mesh_cols
                                                  : (linearized_mesh_coord / mesh_cols) * mesh_cols;
    constexpr uint32_t dispatch_group_stride = axis == ReplicateGroup::COLS ? mesh_cols : 1;
    auto dispatch_2d_unicast_handshake = [&](uint64_t sem_addr, bool flush) {
        for (uint32_t peer = 0; peer < dispatch_devices; ++peer) {
            const uint32_t dst_chip_index = dispatch_group_begin + peer * dispatch_group_stride;
            if (dst_chip_index == linearized_mesh_coord) {
                continue;
            }

            ccl_routing_utils::line_unicast_route_info_t route_info{};
            route_info.dst_chip_id = dest_chip_ids[dst_chip_index];
            route_info.dst_mesh_id = dest_mesh_ids[dst_chip_index];
            const uint32_t fabric_route =
                static_cast<uint32_t>(get_next_hop_router_direction(route_info.dst_mesh_id, route_info.dst_chip_id));
            ASSERT(fabric_route < static_cast<uint32_t>(eth_chan_directions::COUNT));
            if (fabric_route >= static_cast<uint32_t>(eth_chan_directions::COUNT)) {
                return;
            }
            const uint8_t connection_slot = dir_to_slot[fabric_route];
            ASSERT(connection_slot != DIR_TO_SLOT_EMPTY);
            if (connection_slot == DIR_TO_SLOT_EMPTY) {
                return;
            }

            ccl_routing_utils::fabric_set_line_unicast_route(
                pkt_hdr_for_route_helper(unicast_packet_header), route_info);
            unicast_packet_header->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_addr, 1, flush});
            auto& sender = fabric_connections.get(connection_slot).sender;
            sender.wait_for_empty_write_slot();
            sender.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(unicast_packet_header), sizeof(PACKET_HEADER_TYPE));
        }
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
    dispatch_2d_unicast_handshake(init_noc_semaphore_addr, /*flush=*/false);
#else
    send_init_semaphore_to_configured_targets<
        linearized_mesh_coord,
        topology,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        axis,
        num_devices>(fabric_connections, sem_packet_header, dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);
#endif

    volatile tt_l1_ptr uint32_t* init_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
    noc_semaphore_wait(init_sem_ptr, dispatch_devices - 1);
    noc_semaphore_set(init_sem_ptr, 0);

    DPRINT_DISPATCH("Fabric setup complete\n");
#endif

    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address);

    volatile tt_l1_ptr uint32_t* ring_data_avail_ptr[num_workers];
    uint64_t ring_space_avail_noc[num_workers];
    uint32_t consumed[num_workers];
    bool done[num_workers];
    uint32_t num_done = 0;
    uint32_t space_avail_sem_l1_offset = get_semaphore(space_avail_semaphore_id);
    for (uint32_t s = 0; s < num_workers; s++) {
        ring_data_avail_ptr[s] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ring_data_avail_id[s]));
        ring_space_avail_noc[s] = get_noc_addr(ring_noc_x[s], ring_noc_y[s], space_avail_sem_l1_offset);
        consumed[s] = 0;
        done[s] = false;
    }

    DPRINT_DISPATCH("[SND] drain loop start (rings={})\n", num_workers);

    while (num_done < num_workers) {
        for (uint32_t s = 0; s < num_workers; s++) {
            if (done[s]) {
                continue;
            }
            // On Blackhole the RISC caches L1 reads (write-through cache), so a remote NoC write into
            // this core's L1 does not invalidate a cached copy. Both *ring_data_avail_ptr[s] and the
            // route_info slot read below are written REMOTELY by the untilizer, so without a fresh L1
            // read each iteration we can spin forever on a stale data_avail==0 (deadlock) or read
            // stale route_info and forward garbage route/dst over the fabric. Mirrors the sibling
            // reader_combine.cpp poll loop.
            // On Wormhole this is unnecessary: the baby RISCV has no L0 data cache over L1 (that
            // cache and its set_l1_data_cache() control exist only on Blackhole), so L1 reads always
            // observe the latest NoC write -- there is nothing to invalidate, and invalidate_l1_cache()
            // (which compiles to a `fence`, itself a no-op on Wormhole) does nothing there.
            invalidate_l1_cache();
            if (*ring_data_avail_ptr[s] >= consumed[s] + 1) {
                uint32_t slot = consumed[s] % writer_cb_size;
                volatile tt_l1_ptr uint32_t* route_info =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ring_route_base[s] + slot * route_info_slot_stride);
                if (route_info[0] == ROUTE_INFO_SENTINEL) {
                    done[s] = true;
                    num_done++;
                    DPRINT_DISPATCH("[SND] ring={} SENTINEL (consumed={})\n", s, consumed[s]);
                } else {
#ifdef SPARSE_MCAST_DISPATCH
                    // Grouped ring slot (1D Ring, any top-k): route_info carries a direction-group of up
                    // to 4 destinations plus the fields the sender needs to rebuild each destination's
                    // metadata locally (the producer leaves the meta ring slot unwritten for groups).
                    // Layout (written by writer_worker_dispatch): [0]=direction, [1]=num_dests,
                    // [2]=token_idx, [3..6]=page[4], [7..10]=dist[4], [11..14]=expert[4], [15..18]=k[4],
                    // [19..22]=weight[4].
                    uint32_t direction = route_info[0];
                    uint32_t num_dests = route_info[1];
                    uint32_t token_idx = route_info[2];
                    uint32_t dest_pages[4];
                    uint32_t dest_dist[4];
                    uint8_t counts[4] = {0, 0, 0, 0};
                    uint8_t num_chips = 0;
                    uint16_t hop_mask = 0;
                    for (uint32_t i = 0; i < num_dests; ++i) {
                        dest_pages[i] = route_info[3 + i];
                        dest_dist[i] = route_info[7 + i];
                        hop_mask |= static_cast<uint16_t>(1u << (dest_dist[i] - 1));  // bit (hops-1) per writing chip
                        // dest_dist is ascending with each chip's pages contiguous (producer packing), so a
                        // new distance opens a writing chip; a repeat extends the current chip's page count.
                        if (i == 0 || dest_dist[i] != dest_dist[i - 1]) {
                            counts[num_chips++] = 1;
                        } else {
                            counts[num_chips - 1]++;
                        }
                    }
                    uint32_t payload_addr = ring_payload_base[s] + slot * aligned_output_page_size;
                    DPRINT_DISPATCH(
                        "[SND] SPARSE_MCAST_FIRED ring={} dir={} num_dests={} num_chips={} hop_mask={}\n",
                        s,
                        direction,
                        num_dests,
                        (uint32_t)num_chips,
                        (uint32_t)hop_mask);

                    // One payload write fanned out to all co-directional destinations; a chip hit by
                    // multiple pages (counts[c] > 1) receives them all from this single multicast.
                    fabric_send_chip_sparse_multicast_noc_scatter_write_1d_in_direction<fabric_max_packet_size>(
                        output_addr_gen,
                        fabric_connections,
                        unicast_packet_header,
                        hop_mask,
                        direction,
                        dest_pages,
                        static_cast<uint8_t>(num_dests),
                        counts,
                        num_chips,
                        payload_addr,
                        (int)aligned_output_page_size,
                        l1_alignment);

                    // Metadata is per-destination: build each one in-place in the meta ring slot from
                    // the grouped route_info, then unicast it along the same direction/hop. One at a
                    // time (single scratch slot) with a flush between so the next build can't clobber an
                    // in-flight source.
                    auto& metadata_sender = fabric_connections[direction];
                    uint32_t meta_addr = ring_meta_base[s] + slot * aligned_metadata_page_size;
                    volatile tt_l1_ptr int32_t* meta = reinterpret_cast<volatile tt_l1_ptr int32_t*>(meta_addr);
                    for (uint32_t i = 0; i < num_dests; ++i) {
                        meta[0] = (int32_t)linearized_mesh_coord;
                        meta[1] = (int32_t)token_idx;
                        meta[2] = (int32_t)route_info[15 + i];  // top-k slot
                        meta[3] = (int32_t)route_info[11 + i];  // routed expert
                        meta[4] = (int32_t)route_info[19 + i];  // routing weight
                        ccl_routing_utils::line_unicast_route_info_t pkt_route_info{};
                        pkt_route_info.distance_in_hops = static_cast<uint16_t>(dest_dist[i]);
                        ccl_routing_utils::fabric_set_line_unicast_route(
                            pkt_hdr_for_route_helper(unicast_packet_header), pkt_route_info);
                        fabric_send_noc_unicast<fabric_max_packet_size>(
                            metadata_addr_gen,
                            metadata_sender,
                            unicast_packet_header,
                            meta_addr,
                            dest_pages[i],
                            (int)aligned_metadata_page_size,
                            l1_alignment);
                        noc_async_writes_flushed();  // meta source reused next iteration
                    }
#else
                    uint32_t route = route_info[0];
                    uint32_t distance = route_info[1];
                    uint32_t page_idx = route_info[2];
                    uint32_t dst_chip_index = route_info[3];
                    uint32_t payload_addr = ring_payload_base[s] + slot * aligned_output_page_size;
                    uint32_t metadata_addr = ring_meta_base[s] + slot * aligned_metadata_page_size;
                    DPRINT_DISPATCH("ring={} send: route={} page={}\n", s, route, page_idx);
#ifdef DEST_CHIP_ID
                    // route_info layout (written by writer_worker_dispatch): [0]=route (1D EDM
                    // index), [1]=distance_hops, [2]=page_idx, [3]=dst_chip_index (2D). Mirrors the
                    // row-major path: under 2D the 1D-style route_info[0] doesn't match the physical
                    // EDM index, so recompute the direction and the (mesh, chip) header from the dest.
                    ccl_routing_utils::line_unicast_route_info_t pkt_route_info{};
                    uint32_t fabric_route;
                    if constexpr (
                        std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader> ||
                        std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::UDMHybridMeshPacketHeader>) {
                        pkt_route_info.dst_chip_id = dest_chip_ids[dst_chip_index];
                        pkt_route_info.dst_mesh_id = dest_mesh_ids[dst_chip_index];
                        // TODO(#46174): drop the private tt_fabric_api.h dependency once
                        // RoutingPlaneConnectionManager exposes a portable (mesh, chip) -> slot lookup.
                        fabric_route = static_cast<uint32_t>(get_next_hop_router_direction(
                            dest_mesh_ids[dst_chip_index], dest_chip_ids[dst_chip_index]));
                    } else {
                        pkt_route_info.distance_in_hops = static_cast<uint16_t>(distance);
                        fabric_route = route;
                    }

                    // Under FABRIC_2D the connections live in the RoutingPlaneConnectionManager (one slot
                    // per direction); select this token's first-hop direction slot. The route is still
                    // set to the actual (possibly multi-hop) destination by fabric_set_line_unicast_route.
#ifdef FABRIC_2D
                    ASSERT(
                        dir_to_slot[fabric_route] != DIR_TO_SLOT_EMPTY);  // first-hop direction must have an open slot
                    auto& payload_sender = fabric_connections.get(dir_to_slot[fabric_route]).sender;
#else
                    auto& payload_sender = fabric_connections[fabric_route];
#endif
                    {
                        // Send payload
                        ccl_routing_utils::fabric_set_line_unicast_route(
                            pkt_hdr_for_route_helper(unicast_packet_header), pkt_route_info);
                        fabric_send_noc_unicast<fabric_max_packet_size>(
                            output_addr_gen,
                            payload_sender,
                            unicast_packet_header,
                            payload_addr,
                            page_idx,
                            (int)aligned_output_page_size,
                            l1_alignment);

                        // Send metadata
                        ccl_routing_utils::fabric_set_line_unicast_route(
                            pkt_hdr_for_route_helper(unicast_packet_header), pkt_route_info);
                        fabric_send_noc_unicast<fabric_max_packet_size>(
                            metadata_addr_gen,
                            payload_sender,
                            unicast_packet_header,
                            metadata_addr,
                            page_idx,
                            (int)aligned_metadata_page_size,
                            l1_alignment);
                        noc_async_writes_flushed();  // Ensure payload+metadata departed L1 before freeing CB slots
                    }
#endif
#endif  // SPARSE_MCAST_DISPATCH
                    noc_semaphore_inc<true>(ring_space_avail_noc[s], 1);
                    consumed[s]++;
                }
            }
        }
    }

#ifdef DEST_CHIP_ID
    // Defensive: drain any pending local NOC writes before fabric atomic-inc traffic,
    // so the exit-sem signal cannot reach peers ahead of the last metadata/payload writes.
    noc_async_write_barrier();

    // Exit semaphore exchange on a dedicated semaphore (exit_semaphore_address). The exit-inc must
    // not be observed before our prior fabric writes (payload + metadata) to that chip have landed,
    // or a peer could see sem-reached-threshold before the data is in DRAM. The two paths enforce
    // this differently:
    //   - FABRIC_2D: the noc_async_write_barrier() above drains local writes, and each 2D unicast
    //     handshake packet is issued with flush=true so the receiver EDM holds the atomic-inc until our
    //     prior fabric writes commit (init uses flush=false: no prior writes to order against).
    //   - FABRIC_1D (#else): send_init_semaphore_to_configured_targets is called with /*flush=*/true
    //     so the receiver EDM holds the atomic-inc until our prior writes commit (the init handshake
    //     uses the default flush=false since it has no prior writes to order against).
    {
        const uint64_t exit_noc_semaphore_addr = get_noc_addr(exit_semaphore_address);
#ifdef FABRIC_2D
        dispatch_2d_unicast_handshake(exit_noc_semaphore_addr, /*flush=*/true);
#else
        send_init_semaphore_to_configured_targets<
            linearized_mesh_coord,
            topology,
            src_chip_id,
            mesh_rows,
            mesh_cols,
            axis,
            num_devices>(
            fabric_connections,
            sem_packet_header,
            dest_chip_ids,
            dest_mesh_ids,
            exit_noc_semaphore_addr,
            /*flush=*/true);
#endif

        volatile tt_l1_ptr uint32_t* exit_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(exit_semaphore_address);
        DPRINT_DISPATCH(
            "[SND] drain DONE; WAIT exit_sem=={} (have={})\n", dispatch_devices - 1, (uint32_t)(*exit_sem_ptr));
        noc_semaphore_wait(exit_sem_ptr, dispatch_devices - 1);
        DPRINT_DISPATCH("[SND] exit handshake done\n");
        noc_semaphore_set(exit_sem_ptr, 0);
    }

    // FABRIC_2D's FLUSH_BLOCKING handshake sends drain the source before packet-header reuse, but do
    // not prove arrival in the EDM inbox; FABRIC_1D's per-target helper sends are non-blocking. Keep a
    // full barrier before either connection path closes so every prior write and atomic has arrived.
    noc_async_full_barrier();

#ifdef FABRIC_2D
    fabric_connections.close();
#else
    close_direction_connections(directions, fabric_connections);
#endif
#endif
}
