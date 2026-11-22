// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_dispatch_2d_port_writer: the dispatch port's fabric sender, one hop over its link into the neighbour chip
// Runs on: dispatch port cores (one per direction with a neighbour and per link), RISCV_0 (BRISC), NoC 1
// Responsibilities:
//  - reads no DRAM: takes rows (CB_ROWS) and entries (CB_META) from sparse_dispatch_2d_port_reader
//  - fabric-writes each row (1 or 2 a packet) into the same slot of the neighbour's xbuf (the public fabric API, DRAM
//    address resolved for the fabric router's NoC 1, SYS-1419); stages entries per stream in CB_STAGE
//  - fabric fused write + atomic inc: entry batches into the neighbour ports' manifest rings and ring counters
//  - start fence: atomic inc on the neighbour's opposite port's READY word, waits for its own READY from it
//
// One fabric connection (its link): the fence with the neighbour, then every row one hop into the same slot of the
// neighbour's output, and for each the entries the neighbour's ports need (continue / turn + / turn -), batched per
// stream. A flushed fused write + increment vouches for the rows sent before it on the link.
//
// The kernel's order:
//  1. Start fence: tell the neighbour's opposite port this chip has started the op, wait for the same from it, and take
//     the word back to zero. The neighbour's earlier ops are then done with the L1 and the xbuf (and the manifest
//     rings and counters) this sender is about to write.
//  2. Per group from the reader (cb_wait_front on both CBs): send each row as a fabric write (two rows a packet when
//     rows_per_packet is 2), and after every sent row `follow` what the neighbour does with it, which `stage`s entries
//     for its ports; an END entry is staged on every stream that carries it. A staged batch is sent by `send` when it
//     is full. After the group: wait for the writes to leave the L1 (noc_async_writes_flushed) and pop it.
//  3. Before blocking on the reader for the next group, send every partial staged batch (`flush_all`).
//  4. Stop after the group that held END(GMAX), and send what is still staged.
//
// Why:
//  - Staged entries must not sit while this sender waits for the reader: the neighbour's reader sees an entry only
//    when the counter is raised, and its END generations (which this port's reader waits for) advance only on what it
//    has seen. A sender blocked on its reader, which waits on the neighbour, which waits on entries still staged
//    here, would deadlock; so everything staged is sent before cb_wait_front can block.
//  - The entries of a row are sent after the row (on the same connection), so the fused write + increment vouches for
//    every row sent before it (see the reader).
//  - A staged half is not refilled while its packet is still read from the L1: a stream has two halves of
//    manifest_batch entries, `half` is the one being filled, `staged` its entries and `sent` the entries already put
//    into the neighbour's ring (the next batch's offset there).
//  - END is staged only on the streams that exist (`stream_on` 1) and, on a terminal stream (2), only END(GMAX): a link
//    into a chip with nothing to continue or turn into needs just the signal that the link's rows have landed.
//
// Compile-time args: named (sparse_dispatch_2d_names.hpp): manifest_cb (see the reader), rows_cb, meta_cb, stage_cb,
// row_bytes, group, manifest_batch, rows_per_packet and the mesh constants; then the TensorAccessorArgs of out at
// offset 0.
// Runtime args (sparse_dispatch_2d_layout.hpp sender_arg): out, words (this port's semaphore words; the neighbours' sit
// at the same address), ready_x, ready_y (the neighbour's opposite port, NoC coordinates), dst_mesh_id, dst_chip_id,
// ring_offs, per output stream (0: continue, 1 / 2: turn + / -) stream_on (0 off, 1 on, 2 terminal: END(GMAX) only),
// stream_x, stream_y, stream_ring (the ring there); then the neighbour's coordinates neighbour_row, neighbour_col and
// dir; then the fabric connection's. They are the core's block in the DRAM args table (common/args_table.hpp), read
// through moe_block_args::arg.

#include "sparse_dispatch_2d_common.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"

namespace sparse_dispatch {

// The fabric sends go through the public mesh API: each header is filled once per packet (the rows' by the
// tt_fabric::linear::to_noc_* helpers, which resolve the DRAM address for the fabric router's NoC, SYS-1419), and the
// *_with_state call with no update mask waits for a free slot and sends the payload, then the header. The route is set
// once a header (fabric_set_unicast_route), not per packet.
namespace fabric_mesh = tt::tt_fabric::mesh::experimental;

struct SenderArgs {
    uint32_t out_addr;
    uint32_t man_addr;  // the manifest rings' address (the same on every port core of every chip)
    uint32_t ready_addr;
    uint32_t ready_x;
    uint32_t ready_y;
    uint16_t dst_mesh_id;
    uint16_t dst_chip_id;
    uint32_t ring_offs;
    uint32_t cnt_addr[3];
    uint32_t stream_on[3], stream_x[3], stream_y[3], stream_ring[3];
    uint32_t neighbour_row;  // the neighbour's coordinates
    uint32_t neighbour_col;
    uint32_t dir;
};

FORCE_INLINE SenderArgs sender_args(uint32_t man_addr) {
    namespace sender_arg = lay::sender_arg;
    SenderArgs s{};
    s.out_addr = moe_block_args::arg(sender_arg::kOut);
    const uint32_t words = moe_block_args::arg(sender_arg::kWords);
    s.man_addr = man_addr;
    s.ready_addr = words + READY_SEM * SEM_STRIDE;
    s.ready_x = moe_block_args::arg(sender_arg::kReadyX);
    s.ready_y = moe_block_args::arg(sender_arg::kReadyY);
    s.dst_mesh_id = static_cast<uint16_t>(moe_block_args::arg(sender_arg::kDstMeshId));
    s.dst_chip_id = static_cast<uint16_t>(moe_block_args::arg(sender_arg::kDstChipId));
    s.ring_offs = moe_block_args::arg(sender_arg::kRingOffsets);
    for (uint32_t k = 0; k < 3; ++k) {
        s.cnt_addr[k] = words + k * SEM_STRIDE;
    }
    for (uint32_t i = 0; i < sender_arg::kStreamCount; ++i) {
        const uint32_t at = sender_arg::kStreams + i * sender_arg::kStreamWords;
        s.stream_on[i] = moe_block_args::arg(at + sender_arg::kStreamOn);
        s.stream_x[i] = moe_block_args::arg(at + sender_arg::kStreamX);
        s.stream_y[i] = moe_block_args::arg(at + sender_arg::kStreamY);
        s.stream_ring[i] = moe_block_args::arg(at + sender_arg::kStreamRing);
    }
    s.neighbour_row = moe_block_args::arg(sender_arg::kNeighbourRow);
    s.neighbour_col = moe_block_args::arg(sender_arg::kNeighbourCol);
    s.dir = moe_block_args::arg(sender_arg::kDir);
    return s;
}

// The neighbour chip and the axis this port sends along.
struct Neighbour {
    bool axis_x;
    uint32_t row, col;
};

// The entries staged for the neighbour's ports, one stream each (0: continue, 1 / 2: turn + / -), in two halves of
// manifest_batch entries a stream: `stage` appends to the stream's current half and, when it is full, `send` ships it
// as one fused write + atomic increment into the stream's ring on the neighbour, then switches the stream to its other
// half. Side effects: send blocks on a free fabric slot and flushes the NoC writes; stage may call send.
template <typename CONN, uint32_t manifest_batch>
struct ManifestStager {
    CONN& conn;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* mhdr;  // the header of every manifest packet
    const SenderArgs& args;
    uint32_t stage_l1;               // 3 streams x 2 halves x manifest_batch entries
    uint32_t half[3] = {0, 0, 0};    // the half being filled
    uint32_t staged[3] = {0, 0, 0};  // entries in it
    uint32_t sent[3] = {0, 0, 0};    // entries already put into the neighbour's ring

    FORCE_INLINE uint32_t half_l1(uint32_t stream) const {
        return stage_l1 + (stream * 2 + half[stream]) * manifest_batch * lay::kEntryBytes;
    }
    // ship the stream's staged entries (nothing if none)
    FORCE_INLINE void send(uint32_t stream) {
        if (staged[stream] == 0) {
            return;
        }
        const uint32_t src = half_l1(stream);
        const uint32_t ring = args.stream_ring[stream];
        const uint64_t dst = get_noc_addr(
            args.stream_x[stream],
            args.stream_y[stream],
            args.man_addr + (lay::ring_offset(args.ring_offs, ring) + sent[stream]) * lay::kEntryBytes);
        const uint64_t cnt_noc = get_noc_addr(args.stream_x[stream], args.stream_y[stream], args.cnt_addr[ring]);
        mhdr->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, cnt_noc, staged[stream], true},
            staged[stream] * lay::kEntryBytes);
        fabric_mesh::fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(&conn, mhdr, src);
        noc_async_writes_flushed();
        sent[stream] += staged[stream];
        staged[stream] = 0;
        half[stream] ^= 1;
    }
    FORCE_INLINE void stage(uint32_t stream, uint32_t w0, uint32_t w1) {
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(half_l1(stream)) + lay::kEntryWords * staged[stream];
        p[0] = w0;
        p[1] = w1;
        if (++staged[stream] == manifest_batch) {
            send(stream);
        }
    }
    FORCE_INLINE void flush_all() {
        for (uint32_t stream = 0; stream < 3; ++stream) {
            send(stream);
        }
    }
};

// What the neighbour does with a row this port just sent it: stages the entries for its ports (continue on the same
// axis with one hop less; on a first leg, turn into the other axis where its line needs it).
template <typename STAGER>
FORCE_INLINE void follow(STAGER& stager, const Neighbour& nb, uint32_t w0, uint32_t m) {
    const uint32_t slot = lay::kSlot.get(w0), h = lay::kHops.get(w0), g = lay::kGeneration.get(w0),
                   order = lay::kOrder.get(w0);
    if (h > 1) {
        stager.stage(0, lay::entry_word0(slot, h - 1, g + 1, order), m);
    }
    if (nb.axis_x == (order == 0)) {  // first leg: the neighbour turns into the other axis where its line needs it
        const uint32_t t = slot % S;
        const uint32_t span = turn_span(!nb.axis_x, nb.row, nb.col, m, t);
        const uint32_t fwd = lay::kSpanForward.get(span), bwd = lay::kSpanBackward.get(span);
        if (fwd) {
            stager.stage(1, lay::entry_word0(slot, fwd, g + 1, order), m);
        }
        if (bwd) {
            stager.stage(2, lay::entry_word0(slot, bwd, g + 1, order), m);
        }
    }
}

template <typename CONN>
void send(const SenderArgs& args, CONN& conn) {
    using namespace tt::tt_fabric;
    constexpr uint32_t cb_rows = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kRowsCb);
    constexpr uint32_t cb_meta = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kMetaCb);
    constexpr uint32_t cb_stage = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kStageCb);
    constexpr uint32_t row_bytes = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kRowBytes);
    constexpr uint32_t group = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kGroup);
    constexpr uint32_t manifest_batch = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kManifestBatch);
    constexpr uint32_t ROWS_PER_PKT = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kRowsPerPacket);
    constexpr auto out_args = TensorAccessorArgs<0>();

    const uint32_t ready_addr = args.ready_addr;
    const bool axis_x = args.dir < 2;
    const auto out = TensorAccessor(out_args, args.out_addr, row_bytes);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[group];
    for (uint32_t h = 0; h < group; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        (void)fabric_set_unicast_route(hdrs[h], args.dst_chip_id, args.dst_mesh_id);
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* mhdr = PacketHeaderPool::allocate_header();
    (void)fabric_set_unicast_route(mhdr, args.dst_chip_id, args.dst_mesh_id);
    // fence: tell the neighbour's opposite port this chip has started; wait for the same from it
    hdrs[0]->to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader{get_noc_addr(args.ready_x, args.ready_y, ready_addr), 1, true});
    fabric_mesh::fabric_unicast_noc_unicast_atomic_inc_with_state(&conn, hdrs[0]);
    noc_async_writes_flushed();  // (hdrs[0] is refilled for the first row)
    {
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
        noc_semaphore_wait_min(ready, 1);
        noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
        noc_async_atomic_barrier();
    }

    cb_reserve_back(cb_stage, 1);
    const uint32_t stage_l1 = get_write_ptr(cb_stage);  // 3 streams x 2 halves x manifest_batch entries
    ManifestStager<CONN, manifest_batch> stager{conn, mhdr, args, stage_l1};
    const Neighbour neighbour{axis_x, args.neighbour_row, args.neighbour_col};

    bool done = false;
    while (!done) {
        if (!cb_pages_available_at_front(cb_meta, group)) {
            stager.flush_all();  // about to wait: nothing may sit staged
        }
        cb_wait_front(cb_meta, group);
        cb_wait_front(cb_rows, group);
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_meta));
        const uint32_t rows_l1 = get_read_ptr(cb_rows);
        for (uint32_t i = 0; i < group; ++i) {
            const uint32_t w0 = meta[i * META_WORDS], m = meta[i * META_WORDS + 1];
            if (w0 & END) {
                const bool last = lay::kGeneration.get(w0) == GMAX;
                // (stream_on 2: a terminal stream - END(GMAX) only)
                for (uint32_t stream = 0; stream < 3; ++stream) {
                    if (args.stream_on[stream] == 1 || (args.stream_on[stream] == 2 && last)) {
                        stager.stage(stream, w0, 0);
                    }
                }
                if (last) {
                    done = true;
                }
                continue;
            }
            if (w0 & SKIP) {
                continue;
            }
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[i];
            if constexpr (ROWS_PER_PKT == 2) {
                const uint32_t next_w0 = i + 1 < group ? meta[(i + 1) * META_WORDS] : END;
                if ((next_w0 & (END | SKIP)) == 0) {
                    const uint32_t next_mask = meta[(i + 1) * META_WORDS + 1];
                    linear::to_noc_unicast_scatter_write(
                        row_bytes, hdr, lay::kSlot.get(w0), lay::kSlot.get(next_w0), out);
                    fabric_mesh::fabric_unicast_noc_scatter_write_with_state(&conn, hdr, rows_l1 + i * row_bytes);
                    follow(stager, neighbour, w0, m);
                    follow(stager, neighbour, next_w0, next_mask);
                    ++i;
                    continue;
                }
            }
            linear::to_noc_unicast_write(row_bytes, hdr, lay::kSlot.get(w0), out);
            fabric_mesh::fabric_unicast_noc_unicast_write_with_state(&conn, hdr, rows_l1 + i * row_bytes);
            follow(stager, neighbour, w0, m);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_rows, group);
        cb_pop_front(cb_meta, group);
    }
    stager.flush_all();
}

}  // namespace sparse_dispatch

void kernel_main() {
    constexpr uint32_t cb_man = get_named_compile_time_arg_val(sparse_dispatch_2d_names::kManifestCb);
    moe_block_args::load();  // (this core's values: from the DRAM table, common/args_table.hpp)
    // (a port core of another chip only: one program for the mesh)
    if (moe_block_args::arg(sparse_dispatch::lay::sender_arg::kReadyX) == sparse_dispatch::IDLE_PORT) {
        return;
    }
    const sparse_dispatch::SenderArgs args = sparse_dispatch::sender_args(get_write_ptr(cb_man));
    // (the fabric connection's args: runtime args - kept from the block, from index 0; their eth channel differs by
    // chip, a per-device runtime arg)
    size_t fabric_arg = moe_block_args::kEnabled ? 0 : sparse_dispatch::lay::sender_arg::kFabric;
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(fabric_arg);
    conn.open();
    sparse_dispatch::send(args, conn);
    conn.close();
}
