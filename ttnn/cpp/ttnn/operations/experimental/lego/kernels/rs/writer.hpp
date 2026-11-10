// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ring reduce-scatter writer. Port of reduce_scatter_minimal_async/device/kernels/
// ring_reduce_scatter_minimal_async_writer.cpp, tiled staging (contiguous_interm = 0) and a direct fabric connection
// (no worker mux). Same start-of-op barrier handshake, ring state machine, scatter / fused-atomic-inc packets and
// out_ready increments.
//
// CT (A::): my_chip_id ring_size cb_compute_output_id cb_reader_output_id tile_granularity page_size
//   num_tiles_to_write_per_packet output_batch_num_pages input_batch_num_pages input_channel_num_pages
//   output_channel_num_pages input_tensor_Wt slice_C slice_Ht slice_Wt dim
//   cta_route (16 words: fwd unicast, fwd mcast, bwd unicast, bwd mcast route info) cta_interm cta_output
// Common RT: the 8 ReduceScatterCommonArgs words
//   traversals (batched: one ring traversal per batch, as the reader); use_mux, and when use_mux the MuxClient CT
//   values (fabric/mux_client.hpp). The original uses a fabric mux v2
//   client; this block connects through the program-local v1 mux, as all_gather_async's writer does.
// RT: this_x this_y opposite_x opposite_y use_barrier_sem direction chunks_per_sync start_pages_read_in_row
//   start_row_offset start_tiles_read start_tiles_to_read unit_start unit_end, then the fabric connection block
//   (direct) or the 17-word mux client block (use_mux), then [dual_output: dual_addr] [signal_own: slice_counters
//   num_targets x0 y0 ...]
//
// Added hooks: dual_output writes every final (local) output tile also into a second, wider tensor at column offset
// dual_col_offset (the reduce-scatter result placed directly into this device's slot of a following all-gather's
// output: CT dual_Wt dual_col_offset out_Wt cta_dual); signal_own then increments counter word 16 * own_slot on each
// target once those writes have landed (the consumer of that tensor may start on the local slot); when batched, after
// each traversal, to that traversal's targets (RT: counters, then per traversal: num_targets x0 y0 ...).

#pragma once

#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"
#include "ttnn/operations/experimental/lego/kernels/fabric/mux_client.hpp"
#include <type_traits>

namespace lego::rs {

using namespace tt::tt_fabric::linear::experimental;

template <typename A>
struct Writer {
    using address_t = uint32_t;
    using PacketHeaderPtr = decltype(PacketHeaderPool::allocate_header());
    static constexpr uint32_t my_chip_id = A::my_chip_id;
    static constexpr uint32_t ring_size = A::ring_size;
    static constexpr uint32_t tile_granularity = A::tile_granularity;
    static constexpr uint32_t page_size = A::page_size;
    static constexpr uint32_t num_tiles_to_write_per_packet = A::num_tiles_to_write_per_packet;
    static constexpr uint32_t output_batch_num_pages = A::output_batch_num_pages;
    static constexpr uint32_t input_batch_num_pages = A::input_batch_num_pages;
    static constexpr uint32_t input_channel_num_pages = A::input_channel_num_pages;
    static constexpr uint32_t output_channel_num_pages = A::output_channel_num_pages;
    static constexpr uint32_t input_tensor_Wt = A::input_tensor_Wt;
    static constexpr uint32_t slice_C = A::slice_C;
    static constexpr uint32_t slice_Ht = A::slice_Ht;
    static constexpr uint32_t slice_Wt = A::slice_Wt;
    static constexpr uint32_t dim = A::dim;

    static constexpr ccl_routing_utils::line_unicast_route_info_t forward_unicast_route_info =
        ccl_routing_utils::get_line_unicast_route_info_from_args<A::cta_route>();
    static constexpr ccl_routing_utils::line_multicast_route_info_t forward_multicast_route_info =
        ccl_routing_utils::get_line_multicast_route_info_from_args<
            A::cta_route + ccl_routing_utils::num_line_unicast_args>();
    static constexpr ccl_routing_utils::line_unicast_route_info_t backward_unicast_route_info =
        ccl_routing_utils::get_line_unicast_route_info_from_args<
            A::cta_route + ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args>();
    static constexpr ccl_routing_utils::line_multicast_route_info_t backward_multicast_route_info =
        ccl_routing_utils::get_line_multicast_route_info_from_args<
            A::cta_route + 2 * ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args>();

    static FORCE_INLINE uint32_t slice_base_tile_id(uint32_t slice_idx) {
        if constexpr (dim == 3) {
            return slice_idx * slice_Wt;
        } else if constexpr (dim == 2) {
            return slice_idx * slice_Ht * slice_Wt;
        } else {
            return slice_idx * slice_C * slice_Ht * slice_Wt;
        }
    }

    // IntermSink</*Contiguous=*/false> of the original kernel.
    struct IntermSink {
        const uint32_t start_pages_read_in_row;
        const uint32_t start_row_offset;
        const uint32_t start_tiles_read;
        uint32_t interm_slice_base = 0;
        uint32_t output_batch_base = 0;
        uint32_t interm_tile_id_start = 0;
        uint32_t output_tile_id_start = 0;
        uint32_t pages_read_in_row = 0;
        uint32_t row_offset = 0;
        uint32_t output_tiles_read = 0;
        static_assert(num_tiles_to_write_per_packet <= 4, "tiles per packet > 4 is unsupported");
        uint64_t remote_noc_addrs[4] = {0, 0, 0, 0};
        uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
        PacketHeaderPtr pkt_scatter_hdr = nullptr;
        PacketHeaderPtr pkt_unicast_hdr = nullptr;
        PacketHeaderPtr pkt_hdr_fused_unicast = nullptr;
        PacketHeaderPtr pkt_hdr_fused_scatter = nullptr;

        IntermSink(uint32_t pages_read_in_row_start, uint32_t row_offset_start, uint32_t tiles_read_start) :
            start_pages_read_in_row(pages_read_in_row_start),
            start_row_offset(row_offset_start),
            start_tiles_read(tiles_read_start) {}

        void set_packet_header_states(const ccl_routing_utils::line_unicast_route_info_t& unicast_route_info) {
            pkt_scatter_hdr = PacketHeaderPool::allocate_header();
            pkt_unicast_hdr = PacketHeaderPool::allocate_header();
            pkt_hdr_fused_unicast = PacketHeaderPool::allocate_header();
            pkt_hdr_fused_scatter = PacketHeaderPool::allocate_header();
            ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);
            ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr, unicast_route_info);
            ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_fused_unicast, unicast_route_info);
            ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_fused_scatter, unicast_route_info);
            if constexpr (num_tiles_to_write_per_packet >= tt::tt_fabric::NOC_SCATTER_WRITE_MIN_CHUNKS) {
                fabric_unicast_noc_scatter_write_set_state<
                    UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
                    pkt_scatter_hdr,
                    static_cast<uint8_t>(unicast_route_info.distance_in_hops),
                    NocUnicastScatterCommandHeader(remote_noc_addrs, chunk_sizes, num_tiles_to_write_per_packet),
                    page_size * num_tiles_to_write_per_packet);
            }
            fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
                pkt_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, page_size);
            fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
                UnicastFusedAtomicIncUpdateMask::PayloadSize | UnicastFusedAtomicIncUpdateMask::Val |
                UnicastFusedAtomicIncUpdateMask::Flush>(
                pkt_hdr_fused_unicast,
                static_cast<uint8_t>(unicast_route_info.distance_in_hops),
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{0, 0, static_cast<uint32_t>(1)},
                page_size);
            fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state<
                UnicastFusedScatterWriteAtomicIncUpdateMask::PayloadSize |
                UnicastFusedScatterWriteAtomicIncUpdateMask::WriteChunkSizes |
                UnicastFusedScatterWriteAtomicIncUpdateMask::Val | UnicastFusedScatterWriteAtomicIncUpdateMask::Flush>(
                pkt_hdr_fused_scatter,
                static_cast<uint8_t>(unicast_route_info.distance_in_hops),
                tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                    {0, 0}, 0, {static_cast<uint16_t>(page_size)}, static_cast<uint16_t>(1)},
                static_cast<uint16_t>(page_size * 2));
        }
        void begin_iteration(uint32_t b, uint32_t slice_idx) {
            interm_slice_base = b * input_batch_num_pages + slice_base_tile_id(slice_idx);
            output_batch_base = b * output_batch_num_pages;
        }
        void begin_channel(uint32_t c) {
            interm_tile_id_start = interm_slice_base + c * input_channel_num_pages;
            output_tile_id_start = output_batch_base + c * output_channel_num_pages;
            pages_read_in_row = start_pages_read_in_row;
            row_offset = start_row_offset;
            output_tiles_read = start_tiles_read;
        }
        void skip_chunk(uint32_t tiles) {
            for (uint32_t k = 0; k < tiles; ++k) {
                next_interm_tile_id();
                next_output_tile_id();
            }
        }
        template <typename Connection, typename IntermAcc, typename OutputAcc>
        bool write_chunk_remote(
            Connection* connection,
            const Noc& noc,
            const IntermAcc& interm,
            const OutputAcc& output,
            size_t l1_read_addr,
            uint32_t tiles_to_read,
            bool write_to_interm,
            bool fuse_seminc,
            uint64_t sem_noc_addr) {
            bool seminc_fused = false;
            for (uint32_t j = 0; j < tiles_to_read; j += num_tiles_to_write_per_packet) {
                const uint32_t tiles_in_packet = std::min(tiles_to_read - j, num_tiles_to_write_per_packet);
                for (uint32_t k = 0; k < tiles_in_packet; ++k) {
                    const uint32_t interm_tile_id = next_interm_tile_id();
                    const uint32_t output_tile_id = next_output_tile_id();
                    remote_noc_addrs[k] =
                        write_to_interm
                            ? tt::tt_fabric::linear::addrgen_detail::get_noc_address(interm, interm_tile_id, 0)
                            : tt::tt_fabric::linear::addrgen_detail::get_noc_address(output, output_tile_id, 0);
                }
                const bool last_packet = (j + num_tiles_to_write_per_packet >= tiles_to_read);
                seminc_fused |= send_write_packet(
                    connection, l1_read_addr, tiles_in_packet, fuse_seminc && last_packet, sem_noc_addr);
                noc.async_writes_flushed();
                l1_read_addr += page_size * tiles_in_packet;
            }
            return seminc_fused;
        }
        template <typename IntermAcc, typename OutputAcc, typename DualAcc>
        void write_chunk_local(
            const Noc& noc,
            const IntermAcc& interm,
            const OutputAcc& output,
            const DualAcc& dual,
            CircularBuffer& cb_out,
            uint32_t tiles_to_read,
            bool write_to_interm) {
            size_t l1_read_offset = 0;
            for (uint32_t j = 0; j < tiles_to_read; ++j) {
                const uint32_t interm_tile_id = next_interm_tile_id();
                const uint32_t output_tile_id = next_output_tile_id();
                if (write_to_interm) {
                    noc.async_write(
                        cb_out, interm, page_size, {.offset_bytes = l1_read_offset}, {.page_id = interm_tile_id});
                } else {
                    noc.async_write(
                        cb_out, output, page_size, {.offset_bytes = l1_read_offset}, {.page_id = output_tile_id});
                    if constexpr (A::dual_output) {
                        const uint32_t r = output_tile_id / A::out_Wt, c = output_tile_id % A::out_Wt;
                        noc.async_write(
                            cb_out,
                            dual,
                            page_size,
                            {.offset_bytes = l1_read_offset},
                            {.page_id = r * A::dual_Wt + A::dual_col_offset + c});
                    }
                }
                l1_read_offset += page_size;
            }
        }

    private:
        uint32_t next_interm_tile_id() {
            uint32_t tile_id = interm_tile_id_start + row_offset + pages_read_in_row;
            ++pages_read_in_row;
            if (pages_read_in_row == slice_Wt) {
                row_offset += input_tensor_Wt;
                pages_read_in_row -= slice_Wt;
            }
            return tile_id;
        }
        uint32_t next_output_tile_id() { return output_tile_id_start + (output_tiles_read++); }

        template <typename Connection>
        bool send_write_packet(
            Connection* connection, size_t l1_read_addr, uint32_t num_tiles, bool fuse_seminc, uint64_t sem_noc_addr) {
            if (fuse_seminc && num_tiles <= 2) {
                if (num_tiles == 2) {
                    fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state<
                        UnicastFusedScatterWriteAtomicIncUpdateMask::WriteDstAddrs |
                        UnicastFusedScatterWriteAtomicIncUpdateMask::SemaphoreDstAddr>(
                        connection,
                        pkt_hdr_fused_scatter,
                        l1_read_addr,
                        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                            {remote_noc_addrs[0], remote_noc_addrs[1]},
                            sem_noc_addr,
                            {static_cast<uint16_t>(page_size)},
                            static_cast<uint16_t>(1)});
                } else {
                    fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<
                        UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
                        connection,
                        pkt_hdr_fused_unicast,
                        l1_read_addr,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            remote_noc_addrs[0], sem_noc_addr, static_cast<uint32_t>(1)});
                }
                return true;
            }
            if (num_tiles > 1) {
                fabric_unicast_noc_scatter_write_with_state<
                    UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                    UnicastScatterWriteUpdateMask::PayloadSize>(
                    connection,
                    pkt_scatter_hdr,
                    l1_read_addr,
                    NocUnicastScatterCommandHeader(remote_noc_addrs, chunk_sizes, num_tiles),
                    page_size * num_tiles);
            } else {
                fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                    connection, pkt_unicast_hdr, l1_read_addr, NocUnicastCommandHeader{remote_noc_addrs[0]});
            }
            return false;
        }
    };

    uint8_t this_core_x, this_core_y;
    uint32_t opposite_core_x, opposite_core_y;
    bool use_barrier_sem, direction;
    size_t barrier_sem, out_ready_sem;
    uint32_t chunks_per_sync, start_pages_read_in_row, start_row_offset, start_tiles_read, start_tiles_to_read;
    uint32_t unit_start, unit_end;
    address_t interm_tensor_address, output_tensor_address;
    FabricConnectionManager fabric_connection;
    struct NoMux {};
    std::conditional_t<A::use_mux, lego::fabric::MuxClient<A>, NoMux> mux;
    uint32_t hook_arg = 0;

    void init() {
        uint32_t arg_idx = 0;
        interm_tensor_address = crt<A>(ttnn::ccl::ReduceScatterCommonArgs::intermediate);
        output_tensor_address = crt<A>(ttnn::ccl::ReduceScatterCommonArgs::output);
        this_core_x = rt<A>(arg_idx++);
        this_core_y = rt<A>(arg_idx++);
        opposite_core_x = rt<A>(arg_idx++);
        opposite_core_y = rt<A>(arg_idx++);
        use_barrier_sem = rt<A>(arg_idx++);
        barrier_sem = crt<A>(ttnn::ccl::ReduceScatterCommonArgs::barrier);
        direction = rt<A>(arg_idx++);
        out_ready_sem = crt<A>(ttnn::ccl::ReduceScatterCommonArgs::semaphore_0 + direction);
        chunks_per_sync = rt<A>(arg_idx++);
        start_pages_read_in_row = rt<A>(arg_idx++);
        start_row_offset = rt<A>(arg_idx++);
        start_tiles_read = rt<A>(arg_idx++);
        start_tiles_to_read = rt<A>(arg_idx++);
        unit_start = rt<A>(arg_idx++);
        unit_end = rt<A>(arg_idx++);
        if constexpr (A::use_mux) {
            mux.read_args(arg_idx);
            hook_arg = arg_idx + 17;
        } else {
            size_t arg_for_fab = rt_index<A>(arg_idx);
            fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);
            hook_arg = arg_for_fab - A::rt_base;
        }
    }

    void signal_own(uint32_t traversal) {
        if constexpr (A::signal_own) {
            noc_async_write_barrier();
            const uint32_t base = hook_arg + (A::dual_output ? 1 : 0);
            const uint32_t counters = rt<A>(base);
            uint32_t pos = base + 1;
            for (uint32_t tt = 0; tt < traversal; tt++) {
                pos += 1 + 2 * rt<A>(pos);
            }
            const uint32_t n = rt<A>(pos);
            for (uint32_t t = 0; t < n; t++) {
                noc_semaphore_inc(
                    get_noc_addr(rt<A>(pos + 1 + 2 * t), rt<A>(pos + 2 + 2 * t), counters + 16 * A::own_slot), 1);
            }
            noc_async_atomic_barrier();
        }
    }

    void run() {
        if constexpr (A::use_mux) {
            auto* conn = mux.open();
            body(conn);
            mux.close();
        } else {
            if (fabric_connection.is_logically_connected()) {
                fabric_connection.open();
            }
            auto* conn =
                direction ? &fabric_connection.get_forward_connection() : &fabric_connection.get_backward_connection();
            body(conn);
            // Close here, not in teardown(): a later block on this RISC may connect to the same fabric channel.
            if (fabric_connection.is_logically_connected()) {
                fabric_connection.close();
            }
        }
        Noc noc_obj;
        noc_obj.async_write_barrier();
    }

    template <typename Conn>
    void body(Conn* fabric_direction_connection) {
        const auto& unicast_route_info = (direction == 1) ? forward_unicast_route_info : backward_unicast_route_info;
        const auto& multicast_route_info =
            (direction == 1) ? forward_multicast_route_info : backward_multicast_route_info;
        constexpr auto interm_tensor_args = TensorAccessorArgs<A::cta_interm>();
        auto interm_tensor_accessor = TensorAccessor(interm_tensor_args, interm_tensor_address);
        constexpr auto output_tensor_args = TensorAccessorArgs<A::cta_output>();
        auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);
        constexpr auto dual_args = TensorAccessorArgs<A::cta_dual>();
        auto dual_accessor = TensorAccessor(dual_args, A::dual_output ? rt<A>(hook_arg) : 0);
        IntermSink interm_sink(start_pages_read_in_row, start_row_offset, start_tiles_read);

        auto pkt_hdr_seminc = PacketHeaderPool::allocate_header();
        auto pkt_hdr_mcastseminc = PacketHeaderPool::allocate_header();
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);
        interm_sink.set_packet_header_states(unicast_route_info);

        Noc noc_obj;
        CircularBuffer cb_compute_output(A::cb_compute_output_id);
        CircularBuffer cb_reader_output(A::cb_reader_output_id);

        fabric_multicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            pkt_hdr_mcastseminc,
            static_cast<uint8_t>(multicast_route_info.start_distance_in_hops),
            static_cast<uint8_t>(multicast_route_info.range_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, static_cast<uint32_t>(1)});
        fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            pkt_hdr_seminc,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, static_cast<uint32_t>(1)});

        if (use_barrier_sem) {
            uint64_t opposite_barrier_sem_noc_addr =
                safe_get_noc_addr(opposite_core_x, opposite_core_y, barrier_sem, 0);
            fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_direction_connection,
                pkt_hdr_seminc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{opposite_barrier_sem_noc_addr, 0});
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 1);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
        }

        uint64_t this_core_sem_noc_addr = safe_get_noc_addr(this_core_x, this_core_y, out_ready_sem, 0);
        uint64_t opposite_core_sem_noc_addr = safe_get_noc_addr(opposite_core_x, opposite_core_y, out_ready_sem, 0);
        uint64_t even_core_sem_noc_addr = direction ? this_core_sem_noc_addr : opposite_core_sem_noc_addr;
        uint64_t odd_core_sem_noc_addr = !direction ? this_core_sem_noc_addr : opposite_core_sem_noc_addr;

        auto send_seminc = [&](uint64_t sem_noc_addr) {
            fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_direction_connection,
                pkt_hdr_seminc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, 0});
            noc_obj.async_writes_flushed();
        };

        for (uint32_t t = 0; t < A::traversals; ++t) {
            // Units of this traversal: all of the worker's units, or (batched) its units of batch t.
            const uint32_t t_unit_start =
                A::traversals == 1 ? unit_start : (unit_start > t * A::slice_C ? unit_start : t * A::slice_C);
            const uint32_t t_unit_end =
                A::traversals == 1 ? unit_end : (unit_end < (t + 1) * A::slice_C ? unit_end : (t + 1) * A::slice_C);
            constexpr uint32_t ring_size_by_2 = ring_size / 2;
            int slice_idx = my_chip_id + ring_size_by_2;
            uint32_t num_iters = ring_size_by_2 + 1;
            for (uint32_t i = 0; i < num_iters; ++i) {
                bool even_chunks, odd_chunks, reduce_even_chunks, reduce_odd_chunks, write_to_remote, write_to_interm,
                    separate_even_odd_sems;
                if (i == 0) {
                    even_chunks = direction;
                    odd_chunks = !direction;
                    reduce_even_chunks = false;
                    reduce_odd_chunks = false;
                    write_to_remote = true;
                    write_to_interm = true;
                    separate_even_odd_sems = false;
                } else if (i == ring_size_by_2) {
                    even_chunks = direction;
                    odd_chunks = !direction;
                    reduce_even_chunks = even_chunks;
                    reduce_odd_chunks = odd_chunks;
                    write_to_remote = false;
                    write_to_interm = false;
                    separate_even_odd_sems = false;
                } else if (i == 1 || i == ring_size_by_2 - 1) {  // these two coincide for ring_size = 4
                    even_chunks = true;
                    odd_chunks = true;
                    reduce_even_chunks = (i == 1) ? direction : even_chunks;
                    reduce_odd_chunks = (i == 1) ? !direction : odd_chunks;
                    write_to_remote = true;
                    write_to_interm = (i == ring_size_by_2 - 1) ? direction : true;
                    separate_even_odd_sems = (i == ring_size_by_2 - 1);
                } else {
                    even_chunks = true;
                    odd_chunks = true;
                    reduce_even_chunks = even_chunks;
                    reduce_odd_chunks = odd_chunks;
                    write_to_remote = true;
                    write_to_interm = true;
                    separate_even_odd_sems = false;
                }
                if (slice_idx < 0) {
                    slice_idx += ring_size;
                } else if (slice_idx >= (int)ring_size) {
                    slice_idx = (uint32_t)slice_idx - ring_size;
                }
                uint32_t chunk_count = 0, even_chunk_count = 0, odd_chunk_count = 0;
                for (uint32_t u = t_unit_start; u < t_unit_end; ++u) {
                    const uint32_t b = u / slice_C;
                    const uint32_t c = u % slice_C;
                    interm_sink.begin_iteration(b, slice_idx);
                    interm_sink.begin_channel(c);
                    uint32_t tiles_read = start_tiles_read;
                    uint32_t total_tiles_to_read = start_tiles_to_read;
                    while (tiles_read < total_tiles_to_read) {
                        const auto [is_even_chunk, tiles_to_read] =
                            reduce_scatter_common::chunk_ring_parity<tile_granularity>(tiles_read, total_tiles_to_read);
                        if ((is_even_chunk && !even_chunks) || (!is_even_chunk && !odd_chunks) || tiles_to_read == 0) {
                            tiles_read += tiles_to_read;
                            interm_sink.skip_chunk(tiles_to_read);
                        } else {
                            const bool reduce_interm =
                                (is_even_chunk && reduce_even_chunks) || (!is_even_chunk && reduce_odd_chunks);
                            CircularBuffer& cb_out = reduce_interm ? cb_compute_output : cb_reader_output;
                            if (write_to_remote) {
                                uint32_t& sync_counter = separate_even_odd_sems
                                                             ? (is_even_chunk ? even_chunk_count : odd_chunk_count)
                                                             : chunk_count;
                                const uint64_t sem_noc_addr =
                                    separate_even_odd_sems
                                        ? (is_even_chunk ? even_core_sem_noc_addr : odd_core_sem_noc_addr)
                                        : this_core_sem_noc_addr;
                                const bool fuse_seminc = (sync_counter + 1 == chunks_per_sync);
                                cb_out.wait_front(tile_granularity);
                                const bool seminc_fused = interm_sink.write_chunk_remote(
                                    fabric_direction_connection,
                                    noc_obj,
                                    interm_tensor_accessor,
                                    output_tensor_accessor,
                                    cb_out.get_read_ptr(),
                                    tiles_to_read,
                                    write_to_interm,
                                    fuse_seminc,
                                    sem_noc_addr);
                                tiles_read += tiles_to_read;
                                cb_out.pop_front(tile_granularity);
                                if (++sync_counter == chunks_per_sync) {
                                    sync_counter = 0;
                                    if (!seminc_fused) {
                                        send_seminc(sem_noc_addr);
                                    }
                                }
                            } else {
                                cb_out.wait_front(tile_granularity);
                                interm_sink.write_chunk_local(
                                    noc_obj,
                                    interm_tensor_accessor,
                                    output_tensor_accessor,
                                    dual_accessor,
                                    cb_out,
                                    tiles_to_read,
                                    write_to_interm);
                                tiles_read += tiles_to_read;
                                noc_obj.async_write_barrier();
                                cb_out.pop_front(tile_granularity);
                            }
                        }
                    }
                }
                if (write_to_remote) {
                    if (separate_even_odd_sems) {
                        if (even_chunks && even_chunk_count != 0) {
                            send_seminc(even_core_sem_noc_addr);
                        }
                        if (odd_chunks && odd_chunk_count != 0) {
                            send_seminc(odd_core_sem_noc_addr);
                        }
                    } else if (chunk_count != 0) {
                        send_seminc(this_core_sem_noc_addr);
                    }
                }
                slice_idx = direction ? (slice_idx - 1) : (slice_idx + 1);
            }
            signal_own(t);
        }  // traversals
        noc_obj.async_write_barrier();
        noc_obj.async_atomic_barrier();
    }

    void teardown() {}
};

}  // namespace lego::rs
