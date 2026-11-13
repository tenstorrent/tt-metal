// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// sparse_ring_sdpa's cross-device load balancing on a gather writer (the decisions are the balance kernel's:
// sparse_ring_balance.cpp, sparse_ring_balance_plan.hpp). A program semaphore on this core (the flag) is raised by the
// balance core:
//   1  this device's cost page is in DRAM: link 0 multicasts it to the same page on every device in this writer's
//      direction, then an increment of their tables_arrived;
//   2  the partner page is: this writer's share of the swapped-out row blocks (every head's Q pages and mask row) goes
//      to the partner slot's import buffers, each block followed by an increment of the partner's imports_arrived;
//   3  the swapped-in blocks' outputs are in the export buffer: this writer's share goes to the home device's output,
//      each block followed by an increment of the home's returns_arrived.
// A writer's share: the row blocks whose partner is reached through its direction (the shorter way round, ties to the
// minus side; on a line, the partner's side) on its link (block % links). States 1 and 2 run as soon as the flag
// allows, between K/V packets (poll); finish() completes them after the gather and then does 3. Remote writes go
// through the local accessors, as the gather's own: the buffers are mesh buffers at one address on every device, and
// the factory enables this only when every device maps DRAM banks to the same NoC coordinates.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_ring_balance_plan.hpp"

namespace ring_attention_balance {

// Runtime args of the writer's balance block, in order.
enum Arg : uint32_t {
    kTable,
    kPartner,
    kQ,
    kImportQ,
    kMask,
    kImportMask,
    kExportOut,
    kOut,
    kTablesArrived,
    kImportsArrived,
    kReturnsArrived,
    kBalanceX,
    kBalanceY,
    kFlagSemaphore,
    kRank,
    kRingSize,
    kIsRing,
    kTableRange,  // devices the table multicast reaches (0: this writer does not send it)
    kLink,
    kNumLinks,
    kHeads,
    kQBlocks,    // Q blocks per head (row blocks)
    kQRowTiles,  // tile rows of Q / the output per head
    kDht,
    kVdht,
    kBlockTiles,
    kMaskRows,  // mask rows per head (nqb_pad)
    kMaskRowBytes,
    kTablePageBytes,
    kQPageBytes,
    kOutPageBytes,
    kMaxPayload,
    kListOffset,  // the scratch's partner-list region (after the data region)
    kNumArgs,
};

template <
    bool enabled,
    bool minus_side,  // this writer sends toward tensor rank - 1
    typename TableArgs,
    typename PartnerArgs,
    typename QArgs,
    typename MaskArgs,
    typename OutArgs>
class Exchange {
public:
    Exchange(
        uint32_t base, uint32_t scratch, volatile PACKET_HEADER_TYPE* data_hdr, volatile PACKET_HEADER_TYPE* inc_hdr) :
        base_(base), scratch_(scratch), data_hdr_(data_hdr), inc_hdr_(inc_hdr) {}

    // Between K/V packets: whatever the flag already allows (never waits).
    template <typename Sender>
    FORCE_INLINE void poll(Sender* conn) {
        if constexpr (enabled) {
            if (state_ < 2 && Semaphore<>(arg(kFlagSemaphore)).value() > state_) {
                advance(conn);
            }
        }
    }

    // After the gather: the rest of states 1 and 2, then the returns.
    template <typename Sender>
    void finish(Sender* conn) {
        if constexpr (enabled) {
            Semaphore<> flag(arg(kFlagSemaphore));
            while (state_ < 2) {
                flag.wait_min(state_ + 1);
                advance(conn);
            }
            flag.wait_min(3);
            for_each_block([&](uint32_t b, uint32_t pb, uint32_t hops) {
                route(hops);
                send_blocks(conn, OutArgs{}, arg(kExportOut), arg(kOut), b, pb, arg(kVdht), arg(kOutPageBytes));
                increment(conn, arg(kReturnsArrived));
            });
        }
    }

private:
    uint32_t arg(uint32_t i) const { return get_arg_val<uint32_t>(base_ + i); }

    template <typename Sender>
    void advance(Sender* conn) {
        if (state_ == 0) {
            const uint32_t range = arg(kTableRange);
            if (range > 0) {
                send_table(conn, range);
            }
        } else {
            for_each_block([&](uint32_t b, uint32_t pb, uint32_t hops) {
                route(hops);
                send_blocks(conn, QArgs{}, arg(kQ), arg(kImportQ), b, pb, arg(kDht), arg(kQPageBytes));
                send_mask_rows(conn, b, pb);
                increment(conn, arg(kImportsArrived));
            });
        }
        ++state_;
    }

    // Data and increments go k hops along this writer's direction.
    void route(uint32_t hops) {
        ccl_routing_utils::fabric_set_line_unicast_route(
            data_hdr_, {.dst_mesh_id = 0, .dst_chip_id = static_cast<uint16_t>(hops)});
        ccl_routing_utils::fabric_set_line_unicast_route(
            inc_hdr_, {.dst_mesh_id = 0, .dst_chip_id = static_cast<uint16_t>(hops)});
    }

    template <typename Sender>
    void increment(Sender* conn, uint32_t sem_addr) {
        inc_hdr_->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            safe_get_noc_addr(arg(kBalanceX), arg(kBalanceY), sem_addr, 0), 1u});
        conn->wait_for_empty_write_slot();
        conn->send_payload_flush_blocking_from_address(
            reinterpret_cast<uint32_t>(inc_hdr_), sizeof(PACKET_HEADER_TYPE));
    }

    // This device's cost page to the same page of the `range` devices in this direction, in payload-sized pieces.
    template <typename Sender>
    void send_table(Sender* conn, uint32_t range) {
        const auto table = TensorAccessor(TableArgs{}, arg(kTable));
        const uint32_t page = arg(kRank);
        const uint32_t bytes = arg(kTablePageBytes);
        const uint32_t piece = arg(kMaxPayload);
        const tt::tt_fabric::MulticastRoutingCommandHeader mcast{1, static_cast<uint8_t>(range)};
        data_hdr_->to_chip_multicast(mcast);
        for (uint32_t off = 0; off < bytes; off += piece) {
            const uint32_t n = bytes - off < piece ? bytes - off : piece;
            noc_async_read(table.get_noc_addr(page, off), scratch_, n);
            noc_async_read_barrier();
            fabric_write_unidir(page, table, data_hdr_, *conn, scratch_, n, off);
        }
        inc_hdr_->to_chip_multicast(mcast);
        increment(conn, arg(kTablesArrived));
    }

    // fn(local block b, partner block, hops) for this writer's share of the swapped row blocks (the partner page in the
    // scratch's list region, the data region free for fn).
    template <typename Fn>
    void for_each_block(Fn fn) {
        const uint32_t q_blocks = arg(kQBlocks);
        const uint32_t ring_size = arg(kRingSize);
        const uint32_t me = arg(kRank);
        const uint32_t list = scratch_ + arg(kListOffset);
        noc_async_read(
            TensorAccessor(PartnerArgs{}, arg(kPartner)).get_noc_addr(0),
            list,
            (q_blocks + 15) / 16 * 16 * sizeof(uint32_t));
        noc_async_read_barrier();
        invalidate_l1_cache();
        const volatile tt_l1_ptr uint32_t* entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(list);
        for (uint32_t b = 0; b < q_blocks; ++b) {
            const uint32_t p = entries[b];
            if (p == sparse_balance::kHome || b % arg(kNumLinks) != arg(kLink)) {
                continue;
            }
            const uint32_t r = p >> 16;
            const uint32_t minus = (me + ring_size - r) % ring_size;  // hops toward rank - 1 on a ring
            const uint32_t plus = (r + ring_size - me) % ring_size;
            const bool go_minus = arg(kIsRing) != 0 ? minus <= plus : r < me;
            if (go_minus == minus_side) {
                fn(b, p & 0xFFFFu, go_minus ? minus : plus);
            }
        }
    }

    // Every head's tile rows of local block b (block_tiles rows x cols tiles of src) to block pb's of dst (the same
    // layout), up to two pages per packet.
    template <typename Sender, typename Args>
    void send_blocks(
        Sender* conn,
        Args args,
        uint32_t src_addr,
        uint32_t dst_addr,
        uint32_t b,
        uint32_t pb,
        uint32_t cols,
        uint32_t page_bytes) {
        const auto src = TensorAccessor(args, src_addr);
        const auto dst = TensorAccessor(args, dst_addr);
        const uint32_t rows = arg(kBlockTiles);
        const uint32_t row_tiles = arg(kQRowTiles);
        const bool pairs = 2 * page_bytes <= arg(kMaxPayload);
        for (uint32_t h = 0; h < arg(kHeads); ++h) {
            const uint32_t s0 = (h * row_tiles + b * rows) * cols;
            const uint32_t d0 = (h * row_tiles + pb * rows) * cols;
            const uint32_t n = rows * cols;
            for (uint32_t t = 0; t < n; ++t) {
                noc_async_read(src.get_noc_addr(s0 + t), scratch_ + t * page_bytes, page_bytes);
            }
            noc_async_read_barrier();
            for (uint32_t t = 0; t < n;) {
                if (pairs && t + 1 < n) {
                    scatter_fabric_write_unidir(
                        d0 + t, d0 + t + 1, dst, data_hdr_, *conn, scratch_ + t * page_bytes, page_bytes);
                    t += 2;
                } else {
                    fabric_write_unidir(d0 + t, dst, data_hdr_, *conn, scratch_ + t * page_bytes, page_bytes);
                    ++t;
                }
            }
        }
    }

    // Every head's mask row of local block b to block pb's row of the partner's import mask, in payload pieces.
    template <typename Sender>
    void send_mask_rows(Sender* conn, uint32_t b, uint32_t pb) {
        const auto src = TensorAccessor(MaskArgs{}, arg(kMask));
        const auto dst = TensorAccessor(MaskArgs{}, arg(kImportMask));
        const uint32_t bytes = arg(kMaskRowBytes);
        const uint32_t piece = arg(kMaxPayload);
        for (uint32_t h = 0; h < arg(kHeads); ++h) {
            const uint32_t src_row = h * arg(kMaskRows) + b;
            const uint32_t dst_row = h * arg(kMaskRows) + pb;
            for (uint32_t off = 0; off < bytes; off += piece) {
                const uint32_t n = bytes - off < piece ? bytes - off : piece;
                noc_async_read(src.get_noc_addr(src_row, off), scratch_, n);
                noc_async_read_barrier();
                fabric_write_unidir(dst_row, dst, data_hdr_, *conn, scratch_, n, off);
            }
        }
    }

    uint32_t base_;
    uint32_t scratch_;
    volatile PACKET_HEADER_TYPE* data_hdr_;
    volatile PACKET_HEADER_TYPE* inc_hdr_;
    uint32_t state_ = 0;
};

}  // namespace ring_attention_balance
