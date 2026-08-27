// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "ring_attention_all_gather_metadata.hpp"
#include <cstdint>
#include <utility>

using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

enum CompileTimeArg : uint32_t {
    kMyChipId,
    kReservedPacketHeaderCbId,
    kCbOutputId,
    kPacketSizeInPages,
    kOutputPageSize,
    kNumTargetsForwardDirection,
    kNumTargetsBackwardDirection,
    kFuseOp,
    kTopology,
    kNumInputs,
    kDirection,
    kUnicastRouteArg0,
    kUnicastRouteArg1,
    kHasMetadata,
    kCbMetaId,
    kNumLinks,
    kSplitForwardingEnabled,
    kOutputBankOwnedSchedule,
    kNumDramBanks,
    kRoundRobinBankPackets,
    kNumFixedCompileTimeArgs,
};

constexpr uint32_t my_chip_id = get_compile_time_arg_val(kMyChipId);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(kReservedPacketHeaderCbId);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(kCbOutputId);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(kPacketSizeInPages);
constexpr uint32_t output_page_size = get_compile_time_arg_val(kOutputPageSize);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(kNumTargetsForwardDirection);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(kNumTargetsBackwardDirection);
constexpr bool fuse_op = get_compile_time_arg_val(kFuseOp);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(kTopology));
constexpr uint32_t num_inputs = get_compile_time_arg_val(kNumInputs);
constexpr bool direction = get_compile_time_arg_val(kDirection);  // 1 is forward, 0 is backward
constexpr uint32_t unicast_route_arg0 = get_compile_time_arg_val(kUnicastRouteArg0);
constexpr uint32_t unicast_route_arg1 = get_compile_time_arg_val(kUnicastRouteArg1);
// Trace-safe metadata path: when set, the writer recomputes the gather extent (valid_pages) on-device
// from kv_actual_isl[0] (a 1-element uint32 DRAM tensor) so it stays matched to the reader's on-device
// recompute (else they desync under a placeholder host logical_n). When false neither this nor the
// metadata accessor is emitted.
constexpr bool has_metadata = get_compile_time_arg_val(kHasMetadata);
constexpr uint32_t cb_meta_id = get_compile_time_arg_val(kCbMetaId);
constexpr uint32_t num_links = get_compile_time_arg_val(kNumLinks);
constexpr bool split_forwarding_enabled = get_compile_time_arg_val(kSplitForwardingEnabled);
constexpr bool output_bank_owned_schedule = get_compile_time_arg_val(kOutputBankOwnedSchedule);
constexpr uint32_t num_dram_banks = get_compile_time_arg_val(kNumDramBanks);
constexpr bool round_robin_bank_packets = get_compile_time_arg_val(kRoundRobinBankPackets);

template <typename AddrGenType, typename FabricSender>
FORCE_INLINE void write_bank_owned_slices_round_robin(
    CircularBuffer& cb_output,
    AddrGenType& output_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    FabricSender& fabric_direction_connection,
    uint32_t output_page_base,
    uint32_t valid_pages,
    uint32_t first_bank,
    uint32_t bank_stride) {
    ring_attention_all_gather::BankOwnedPacketSchedule<num_dram_banks> schedule(
        output_page_base, valid_pages, first_bank, bank_stride, packet_size_in_pages);
    uint32_t first_page_offset = 0;
    uint32_t batch = 0;
    while (schedule.next_packet(first_page_offset, batch)) {
        cb_output.wait_front(packet_size_in_pages);
        fabric_write_unidir(
            output_page_base + first_page_offset,
            output_addrgen,
            pkt_hdr,
            fabric_direction_connection,
            cb_output.get_read_ptr(),
            batch * output_page_size);
        cb_output.pop_front(packet_size_in_pages);
    }
}

template <typename AddrGenType, typename FabricSender>
FORCE_INLINE void write_bank_owned_slices_whole_bank(
    CircularBuffer& cb_output,
    AddrGenType& output_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    FabricSender& fabric_direction_connection,
    uint32_t output_page_base,
    uint32_t valid_pages,
    uint32_t first_bank,
    uint32_t bank_stride) {
    for (uint32_t bank = first_bank; bank < num_dram_banks; bank += bank_stride) {
        const auto bank_slice =
            ring_attention_all_gather::get_bank_owned_slice(output_page_base, valid_pages, bank, num_dram_banks);
        for (uint32_t pages_sent = 0; pages_sent < bank_slice.page_count;) {
            const uint32_t batch = std::min(packet_size_in_pages, bank_slice.page_count - pages_sent);
            cb_output.wait_front(packet_size_in_pages);
            fabric_write_unidir(
                output_page_base + bank_slice.first_page_offset + pages_sent * num_dram_banks,
                output_addrgen,
                pkt_hdr,
                fabric_direction_connection,
                cb_output.get_read_ptr(),
                batch * output_page_size);
            cb_output.pop_front(packet_size_in_pages);
            pages_sent += batch;
        }
    }
}

FORCE_INLINE void discard_bank_owned_slices(
    CircularBuffer& cb_output,
    uint32_t output_page_base,
    uint32_t valid_pages,
    uint32_t first_bank,
    uint32_t bank_stride) {
    for (uint32_t bank = first_bank; bank < num_dram_banks; bank += bank_stride) {
        const uint32_t page_count =
            ring_attention_all_gather::get_bank_owned_slice(output_page_base, valid_pages, bank, num_dram_banks)
                .page_count;
        for (uint32_t pages_discarded = 0; pages_discarded < page_count; pages_discarded += packet_size_in_pages) {
            cb_output.wait_front(packet_size_in_pages);
            cb_output.pop_front(packet_size_in_pages);
        }
    }
}

template <typename AddrGenType, typename FabricSender>
FORCE_INLINE void write_bank_owned_slices(
    CircularBuffer& cb_output,
    AddrGenType& output_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    FabricSender& fabric_direction_connection,
    uint32_t output_page_base,
    uint32_t valid_pages,
    uint32_t first_bank,
    uint32_t bank_stride) {
    if constexpr (round_robin_bank_packets) {
        write_bank_owned_slices_round_robin(
            cb_output,
            output_addrgen,
            pkt_hdr,
            fabric_direction_connection,
            output_page_base,
            valid_pages,
            first_bank,
            bank_stride);
    } else {
        write_bank_owned_slices_whole_bank(
            cb_output,
            output_addrgen,
            pkt_hdr,
            fabric_direction_connection,
            output_page_base,
            valid_pages,
            first_bank,
            bank_stride);
    }
}

void kernel_main() {
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<num_inputs, kNumFixedCompileTimeArgs + num_inputs>();
    constexpr uint32_t kMetaArgsOffset = has_metadata
                                             ? std::get<num_inputs - 1>(outputs_args).next_compile_time_args_offset()
                                             : (kNumFixedCompileTimeArgs + num_inputs);
    constexpr auto meta_args = TensorAccessorArgs<kMetaArgsOffset>();
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    uint32_t gather_dim = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    std::array<uint32_t, num_inputs> input_tensor_Wt;
    std::array<uint32_t, num_inputs> input_tensor_Ht;
    std::array<uint32_t, num_inputs> output_tensor_Wt;
    std::array<uint32_t, num_inputs> output_tensor_Ht;
    std::array<uint32_t, num_inputs> input_batch_head_count;
    std::array<uint32_t, num_inputs> input_tile_id_start;
    std::array<uint32_t, num_inputs> input_tile_id_end;
    std::array<uint32_t, num_inputs> input_valid_pages;
    std::array<uint32_t, num_inputs> worker_link;

    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        input_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_head_count[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        // input_batch_base is reader-only. The writer always targets output
        // slot 0, so it reads the arg here only for alignment.
        (void)get_arg_val<uint32_t>(arg_idx++);
        // valid_pages_per_batch_head (slot 6): clamp the gather to the logical_n-valid slab prefix (must
        // match the reader's clamp so cb_output producer/consumer page counts stay aligned). Default
        // (full input) leaves the range unchanged.
        const uint32_t valid_pages = get_arg_val<uint32_t>(arg_idx++);
        worker_link[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_valid_pages[input_idx] = valid_pages;
        const auto link_page_range =
            ring_attention_all_gather::compute_link_page_range(valid_pages, num_links, worker_link[input_idx]);
        input_tile_id_start[input_idx] = link_page_range.start;
        input_tile_id_end[input_idx] = link_page_range.end;
    }

    auto outputs_tuple = make_tensor_accessor_tuple(outputs_args, arg_idx);
    arg_idx += num_inputs;
    auto output_addrgens = make_abstract_tensor_accessor_wrappers(outputs_tuple);

    // Trace-safe metadata path: recompute the gather extent (valid_pages) on-device from kv_actual_isl[0]
    // so it matches the reader's recompute even when the host logical_n is a placeholder. The
    // kv_actual_isl DRAM address and chunk_local_tiles are the next two runtime args (after the
    // output-buffer addrs, before the fabric args). Identical formula to the all-gather reader / host
    // compute_gather_valid_Ht.
    if constexpr (has_metadata) {
        // kv_actual_isl is a 1-element uint32 DRAM tensor (was metadata[1]); read its page 0.
        const uint32_t kv_actual_isl_addr = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t chunk_local_tiles = get_arg_val<uint32_t>(arg_idx++);
        Noc meta_noc;
        CircularBuffer cb_meta(cb_meta_id);
        // Shared read protocol (async_read page 0 -> barrier -> invalidate_l1_cache -> volatile load). The
        // invalidate is required, not cosmetic: this tensor is at a fixed DRAM address the host refreshes in
        // place between trace replays, so a cached L1 line would return the prior chunk's kv_actual_isl and
        // silently clamp the gather to the wrong prefix.
        const uint32_t kv_actual = trace_metadata::read_metadata_scalar_u32(
            meta_noc, meta_args, kv_actual_isl_addr, cb_meta.get_write_ptr());  // kv_actual_isl (tile-aligned)
        // Same formula the reader uses -- both MUST clamp to the same slab prefix or the cb_output
        // producer/consumer page counts drift (see the header's KEEP IN SYNC note).
        const uint32_t gather_valid_Ht =
            ring_attention_all_gather::compute_gather_valid_Ht(kv_actual, chunk_local_tiles, ring_size);
        ring_attention_all_gather::update_link_page_ranges_for_gather_extent(
            gather_valid_Ht,
            num_links,
            input_tensor_Wt,
            input_valid_pages,
            worker_link,
            input_tile_id_start,
            input_tile_id_end);
    }

    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);
    /* Args for overlapped all gather */
    OpSignaler op_signaler_sender;

    if constexpr (fuse_op) {
        arg_idx = arg_for_fab;
        op_signaler_sender = OpSignaler(arg_idx);
    }

    Noc noc_obj;
    CircularBuffer cb_packet_header(reserved_packet_header_cb_id);
    CircularBuffer cb_output(cb_output_id);

    // packet header cb
    cb_packet_header.reserve_back(1);
    auto packet_header_buffer_addr = cb_packet_header.get_write_ptr();
    cb_packet_header.push_back(1);
    cb_packet_header.reserve_back(1);
    auto packet_header_buffer_seminc = cb_packet_header.get_write_ptr();
    cb_packet_header.push_back(1);

    // pre-populate packet headers
    constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info = {
        .dst_mesh_id = static_cast<uint16_t>(unicast_route_arg0),
        .dst_chip_id = static_cast<uint16_t>(unicast_route_arg1)};

    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr, unicast_route_info);

    fabric_connection.open();
    tt::tt_fabric::WorkerToFabricEdmSender* fabric_direction_connection =
        fabric_connection.is_logically_connected() ? (direction == 1 ? &fabric_connection.get_backward_connection()
                                                                     : &fabric_connection.get_forward_connection())
                                                   : nullptr;
    constexpr uint32_t num_targets_in_direction =
        direction == 1 ? num_targets_backward_direction : num_targets_forward_direction;

    uint32_t slice_writes = 0;

    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        /**
         * Write out the local slice to forward and backward devices
         * Note that it is not copied to local output buffer. This is because
         * the fused op (RingJointAttention) reads from the input buffer directly
         * when accessing the local slice. This is a performance optimization
         * to remove startup latency from the fused op.
         */

        const uint32_t input_pages_per_batch_head = input_tensor_Wt[input_idx] * input_tensor_Ht[input_idx];
        const uint32_t output_pages_per_batch_head = output_tensor_Wt[input_idx] * output_tensor_Ht[input_idx];
        if constexpr (output_bank_owned_schedule) {
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; ++bh_idx) {
                const uint32_t output_page_base =
                    bh_idx * output_pages_per_batch_head + my_chip_id * input_pages_per_batch_head;
                if constexpr (num_targets_in_direction) {
                    write_bank_owned_slices(
                        cb_output,
                        output_addrgens[input_idx],
                        pkt_hdr,
                        *fabric_direction_connection,
                        output_page_base,
                        input_valid_pages[input_idx],
                        worker_link[input_idx],
                        num_links);
                } else {
                    discard_bank_owned_slices(
                        cb_output, output_page_base, input_valid_pages[input_idx], worker_link[input_idx], num_links);
                }
            }
        } else {
            uint32_t tile_id_start = my_chip_id * input_tensor_Wt[input_idx];
            uint32_t pages_read_in_row = input_tile_id_start[input_idx] % input_tensor_Wt[input_idx];
            uint32_t row_offset =
                (input_tile_id_start[input_idx] / input_tensor_Wt[input_idx]) * output_tensor_Wt[input_idx];
            uint32_t tiles_read = input_tile_id_start[input_idx];
            uint32_t tiles_to_read = input_tile_id_end[input_idx];
            if (gather_dim == 3) {
                tile_id_start = my_chip_id * input_tensor_Wt[input_idx];
            } else {
                tile_id_start = my_chip_id * input_pages_per_batch_head;
            }

            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
                while (tiles_read < tiles_to_read) {
                    uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                    cb_output.wait_front(packet_size_in_pages);
                    const size_t l1_read_addr_base = cb_output.get_read_ptr();
                    size_t l1_read_addr = l1_read_addr_base;

                    uint32_t tile_id = tile_id_start + row_offset + pages_read_in_row;

                    pages_read_in_row++;
                    if (pages_read_in_row >= input_tensor_Wt[input_idx]) {
                        row_offset += output_tensor_Wt[input_idx];
                        pages_read_in_row = 0;
                    }

                    if (num_pages_to_read == 2) {
                        uint32_t second_tile_id = tile_id_start + row_offset + pages_read_in_row;

                        if constexpr (num_targets_in_direction) {
                            scatter_fabric_write_unidir(
                                tile_id,
                                second_tile_id,
                                output_addrgens[input_idx],
                                pkt_hdr,
                                *fabric_direction_connection,
                                l1_read_addr,
                                output_page_size);
                        }

                        pages_read_in_row++;
                        if (pages_read_in_row >= input_tensor_Wt[input_idx]) {
                            row_offset += output_tensor_Wt[input_idx];
                            pages_read_in_row = 0;
                        }
                    } else {
                        ASSERT(num_pages_to_read == 1);

                        if constexpr (num_targets_in_direction) {
                            fabric_write_unidir(
                                tile_id,
                                output_addrgens[input_idx],
                                pkt_hdr,
                                *fabric_direction_connection,
                                l1_read_addr,
                                output_page_size);
                        }
                    }

                    tiles_read += num_pages_to_read;
                    cb_output.pop_front(packet_size_in_pages);
                }
                tile_id_start += output_pages_per_batch_head;
                tiles_read = input_tile_id_start[input_idx];
                tiles_to_read = input_tile_id_end[input_idx];
                pages_read_in_row = input_tile_id_start[input_idx] % input_tensor_Wt[input_idx];
                row_offset =
                    (input_tile_id_start[input_idx] / input_tensor_Wt[input_idx]) * output_tensor_Wt[input_idx];
            }
        }
    }

    noc_obj.async_write_barrier();
    // increment locally
    if constexpr (fuse_op && direction == 1) {
        /**
         * Synchronize and signal that the local tensor slice is available
         *
         * While the fused op will not wait on this "local write done" increment,
         * the fused op signaler will account for it in future waits.
         */
        op_signaler_sender.synchronize_workers_and_signal_op(my_chip_id);
    }

    // 2. unicast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
    auto* pkt_hdr_sem_inc = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
    pkt_hdr_sem_inc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1

    // Write the unicast packet. num_hops=1 is correct under both topologies: 1D ring-AG always
    // targets the immediate neighbor; 2D ignores num_hops (HybridMesh::to_chip_unicast is a no-op
    // and route_info carries the 2D destination).
    if constexpr (num_targets_in_direction) {
        fabric_direction_connection->wait_for_empty_write_slot();
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc, unicast_route_info);
        fabric_direction_connection->send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }

    uint32_t writes_expected = 0;
    if constexpr (topology == Topology::Linear) {
        if constexpr (direction == 1 && num_targets_backward_direction) {
            writes_expected = num_targets_forward_direction;
        } else if constexpr (direction == 0 && num_targets_forward_direction) {
            writes_expected = num_targets_backward_direction;
        }
    } else if constexpr (topology == Topology::Ring) {
        if constexpr (direction == 1) {
            writes_expected = num_targets_backward_direction - 1;
        } else {
            writes_expected = num_targets_forward_direction - 1;
        }
    }

    // On an even ring the terminal relayed slice (the downstream neighbor's diametric shard) is relayed
    // half per direction. The gate is a host-derived compile-time flag (see top of file).
    if (split_forwarding_enabled && direction == 1) {
        writes_expected++;
    }

    while (slice_writes < writes_expected) {
        // Direction == backward
        // Did I get something from my left to send to my right?
        // In the linear case, I expect num_targets_backward_direction slices from the left, and check if I have a
        // neighbor to the right
        // In the ring case, I expect to write to the right num_forward_target times
        // Direction == forward
        // Did I get something from my right to send to my left?
        // In the linear case, I expect num_targets_forward_direction slices from the right, and check if I have a
        // neighbor to the left
        // In the ring case, I expect to write to the left num_backward_target times

        int slice_chip_id;
        uint32_t actual_slice_chip_id;
        if constexpr (direction == 1) {
            slice_chip_id = my_chip_id + slice_writes + 1;
            actual_slice_chip_id = (slice_chip_id >= (int)ring_size) ? slice_chip_id - ring_size : slice_chip_id;
        } else {
            slice_chip_id = my_chip_id - slice_writes - 1;
            actual_slice_chip_id = (slice_chip_id < 0) ? ring_size + slice_chip_id : slice_chip_id;
        }
        const bool is_split_forwarded_slice = split_forwarding_enabled && (slice_writes == writes_expected - 1);
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            const uint32_t input_pages_per_batch_head = input_tensor_Wt[input_idx] * input_tensor_Ht[input_idx];
            const uint32_t output_pages_per_batch_head = output_tensor_Wt[input_idx] * output_tensor_Ht[input_idx];
            if constexpr (output_bank_owned_schedule) {
                const uint32_t split_factor = is_split_forwarded_slice ? 2 : 1;
                const uint32_t first_bank =
                    worker_link[input_idx] + (is_split_forwarded_slice ? direction * num_links : 0);
                const uint32_t bank_stride = num_links * split_factor;
                for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; ++bh_idx) {
                    const uint32_t output_page_base =
                        bh_idx * output_pages_per_batch_head + actual_slice_chip_id * input_pages_per_batch_head;
                    write_bank_owned_slices(
                        cb_output,
                        output_addrgens[input_idx],
                        pkt_hdr,
                        *fabric_direction_connection,
                        output_page_base,
                        input_valid_pages[input_idx],
                        first_bank,
                        bank_stride);
                }
            } else {
                const uint32_t total_pages = input_tile_id_end[input_idx] - input_tile_id_start[input_idx];
                const uint32_t num_packets = (total_pages + packet_size_in_pages - 1) / packet_size_in_pages;
                const uint32_t first_half_pages = (num_packets / 2) * packet_size_in_pages;
                uint32_t relay_start = input_tile_id_start[input_idx];
                uint32_t relay_end = input_tile_id_end[input_idx];
                if (is_split_forwarded_slice) {
                    if constexpr (direction == 0) {
                        relay_end = relay_start + first_half_pages;
                    } else {
                        relay_start += first_half_pages;
                    }
                }

                uint32_t tiles_read = relay_start;
                uint32_t tiles_to_read = relay_end;
                uint32_t tile_id_start = actual_slice_chip_id * input_tensor_Wt[input_idx];
                const uint32_t slice_Wt = input_tensor_Wt[input_idx];
                const uint32_t stride_Wt = output_tensor_Wt[input_idx];
                uint32_t row_offset = (relay_start / slice_Wt) * stride_Wt;
                uint32_t pages_read_in_row = relay_start % slice_Wt;

                if (gather_dim == 3) {
                    tile_id_start = actual_slice_chip_id * slice_Wt;
                } else {
                    tile_id_start = actual_slice_chip_id * input_pages_per_batch_head;
                }
                for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
                    while (tiles_read < tiles_to_read) {
                        uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                        cb_output.wait_front(packet_size_in_pages);
                        size_t l1_read_addr = cb_output.get_read_ptr();
                        uint32_t first_tile_id = tile_id_start + row_offset + pages_read_in_row;
                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }
                        if (num_pages_to_read == 2) {
                            uint32_t second_tile_id = tile_id_start + row_offset + pages_read_in_row;
                            pages_read_in_row++;
                            if (pages_read_in_row >= slice_Wt) {
                                row_offset += stride_Wt;
                                pages_read_in_row = 0;
                            }

                            scatter_fabric_write_unidir(
                                first_tile_id,
                                second_tile_id,
                                output_addrgens[input_idx],
                                pkt_hdr,
                                *fabric_direction_connection,
                                l1_read_addr,
                                output_page_size);
                        } else {
                            ASSERT(num_pages_to_read == 1);
                            fabric_write_unidir(
                                first_tile_id,
                                output_addrgens[input_idx],
                                pkt_hdr,
                                *fabric_direction_connection,
                                l1_read_addr,
                                output_page_size);
                        }

                        tiles_read += num_pages_to_read;
                        cb_output.pop_front(packet_size_in_pages);
                    }
                    tile_id_start += output_pages_per_batch_head;
                    tiles_read = relay_start;
                    tiles_to_read = relay_end;
                    row_offset = (relay_start / slice_Wt) * stride_Wt;
                    pages_read_in_row = relay_start % slice_Wt;
                }
            }
        }

        // 2. unicast output ready semaphore forward — route was set once on pkt_hdr_sem_inc
        // before the writes loop above; unicast_route_info is constexpr, so no need to
        // re-set it each iteration.
        fabric_direction_connection->wait_for_empty_write_slot();
        fabric_direction_connection->send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));

        slice_writes++;
    }

    // Drain in-flight writes BEFORE closing the EDM connections.
    noc_obj.async_atomic_barrier();
    noc_obj.async_write_barrier();

    fabric_connection.close();
}
