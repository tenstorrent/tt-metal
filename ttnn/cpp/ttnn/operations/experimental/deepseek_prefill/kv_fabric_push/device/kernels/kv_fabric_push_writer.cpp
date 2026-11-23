// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One launch of the KV push on one chip. The launch's work comes from this core's H2D socket (DEVICE_PULL: pulled
// from pinned host memory a socket page at a time):
//
//   * a header page: the number of runs, then the destination table's bytes;
//   * the destination table: per destination chip its fabric chip id, mesh id, then (NoC x, NoC y, offset) a DRAM bank;
//   * the runs, RUNS a page, 4 words each: source | pages << 8 | destination << 16, first source page, destination
//     buffer (offset from the bank's), first destination page.
//
// A run's pages are consecutive on both sides. A copy source's pages are sent as they are; a tilize source's run is a
// block of ROW_MAJOR bf16 rows, sent as the block's 32x32 tiles (tile-row major). A destination page goes to its
// interleaved bank (page % banks) at the bank's offset plus the buffer's, page / banks pages in.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"

#if !defined(FABRIC_2D)
#error "the push needs a 2D fabric: a remote mesh is reached through its exit chip"
#endif

using namespace tt::tt_fabric::mesh::experimental;

constexpr uint32_t NB = get_compile_time_arg_val(0);       // a destination chip's DRAM banks
constexpr uint32_t STAGE = get_compile_time_arg_val(1);    // pages staged in L1 a NoC read barrier
constexpr uint32_t SLOT = get_compile_time_arg_val(2);     // L1 bytes a staged page
constexpr uint32_t NSRC = get_compile_time_arg_val(3);     // sources
constexpr uint32_t MAX_DST = get_compile_time_arg_val(4);  // L1 bytes for the destination table
constexpr uint32_t SOCKET_CONFIG = get_compile_time_arg_val(5);
constexpr uint32_t PAGE = get_compile_time_arg_val(6);  // socket page bytes
constexpr uint32_t CLASS_CTA = 7;  // then the accessor arguments of each source class, 2 words a class
constexpr uint32_t RUNS = PAGE / 16;
constexpr uint32_t DST_WORDS = 2 + 3 * NB;
constexpr uint32_t TILE = 2048;  // a 32x32 bf16 tile: 4 faces of 16 rows of 32 bytes

struct Source {
    uint32_t addr, page, cls, crta, tilize;
};

FORCE_INLINE void read_page_from_pcie(uint32_t pcie_xy_enc, uint64_t src_pcie, uint32_t dst_l1, uint32_t size) {
    while (size) {
        const uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
        noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
            NOC_INDEX, pcie_xy_enc, src_pcie, dst_l1, chunk);
        src_pcie += chunk;
        dst_l1 += chunk;
        size -= chunk;
    }
}

// The socket's next page into L1 at `l1`, released to the host once read.
FORCE_INLINE void socket_read(SocketReceiverInterface& socket, uint64_t pcie_base, uint32_t l1) {
    socket_wait_for_pages(socket, 1);
    read_page_from_pcie(socket.h2d.pcie_xy_enc, pcie_base + socket.read_ptr - socket.fifo_addr, l1, PAGE);
    noc_async_read_barrier();
    // PCIe routing is sticky in the read command buffer: the DRAM reads between socket pages need it cleared.
    noc_async_read_clear_pcie_state(NOC_INDEX, read_cmd_buf);
    socket_pop_pages(socket, 1);
    socket_notify_sender(socket);
    invalidate_l1_cache();
}

template <typename Sender>
FORCE_INLINE void send_page(
    Sender& sender,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr,
    volatile tt_l1_ptr uint32_t* dst,
    uint32_t l1,
    uint32_t size,
    uint32_t base,
    uint32_t page) {
    const uint32_t bank = page % NB;
    const uint64_t addr =
        get_noc_addr(dst[2 + 3 * bank], dst[3 + 3 * bank], dst[4 + 3 * bank] + base + (page / NB) * size);
    fabric_unicast_noc_unicast_write<Sender, false>(
        &sender, hdr, dst[0], dst[1], l1, size, tt::tt_fabric::NocUnicastCommandHeader{addr});
    noc_async_writes_flushed();  // the header and the staged page left L1 before either is rewritten
}

template <typename Accessor, typename Sender>
void copy_run(
    const Accessor& src,
    uint32_t page_bytes,
    uint32_t sp,
    uint32_t count,
    Sender& sender,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr,
    volatile tt_l1_ptr uint32_t* dst,
    uint32_t base,
    uint32_t dp,
    uint32_t buf) {
    for (uint32_t q0 = 0; q0 < count; q0 += STAGE) {
        const uint32_t m = (count - q0) < STAGE ? (count - q0) : STAGE;
        for (uint32_t q = 0; q < m; ++q) {
            noc_async_read(src.get_noc_addr(sp + q0 + q), buf + q * SLOT, page_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t q = 0; q < m; ++q) {
            send_page(sender, hdr, dst, buf + q * SLOT, page_bytes, base, dp + q0 + q);
        }
    }
}

// `rows` bf16 rows of `row_bytes` (pages sp on) as their tiles: tile (r / 32) * tile_cols + c holds rows r - r % 32
// on, columns 32 c on. The host keeps 2 * rows * row_bytes within the staging buffer.
template <typename Accessor, typename Sender>
void tilize_run(
    const Accessor& src,
    uint32_t row_bytes,
    uint32_t sp,
    uint32_t rows,
    Sender& sender,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr,
    volatile tt_l1_ptr uint32_t* dst,
    uint32_t base,
    uint32_t dp,
    uint32_t buf) {
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_read(src.get_noc_addr(sp + r), buf + r * row_bytes, row_bytes);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t tile_cols = row_bytes / 64;
    const uint32_t tiles = buf + rows * row_bytes;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < tile_cols; ++c) {
            for (uint32_t h = 0; h < 2; ++h) {  // the row's two 16-value halves of tile column c: faces h and h + 2
                const uint32_t* from = reinterpret_cast<const uint32_t*>(buf + r * row_bytes + c * 64 + h * 32);
                volatile tt_l1_ptr uint32_t* to = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    tiles + ((r / 32) * tile_cols + c) * TILE + (((r % 32) / 16) * 2 + h) * 512 + (r % 16) * 32);
                for (uint32_t w = 0; w < 8; ++w) {
                    to[w] = from[w];
                }
            }
        }
    }
    const uint32_t n = rows / 32 * tile_cols;
    for (uint32_t t = 0; t < n; ++t) {
        send_page(sender, hdr, dst, tiles + t * TILE, TILE, base, dp + t);
    }
}

template <uint32_t CTA, typename Sender>
void run(
    const Source& s,
    uint32_t sp,
    uint32_t count,
    Sender& sender,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr,
    volatile tt_l1_ptr uint32_t* dst,
    uint32_t base,
    uint32_t dp,
    uint32_t buf) {
    const auto src = TensorAccessor(TensorAccessorArgs<CTA>(s.crta), s.addr, s.page);
    if (s.tilize) {
        tilize_run(src, s.page, sp, count, sender, hdr, dst, base, dp, buf);
    } else {
        copy_run(src, s.page, sp, count, sender, hdr, dst, base, dp, buf);
    }
}

void kernel_main() {
    // Runtime args: (address, page bytes, class, accessor arguments' offset, tilize) a source, then the connection.
    size_t a = 5 * NSRC;
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t tbuf = buf + STAGE * SLOT;  // a socket page: the header, then each page of runs
    const uint32_t dbuf = tbuf + PAGE;         // the destination table
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbuf);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dbuf);

    // socket_notify_sender posts bytes_acked to the host with a PCIe write on the write command buffer.
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    SocketReceiverInterface socket = create_receiver_socket_interface(SOCKET_CONFIG);
    set_receiver_socket_page_size(socket, PAGE);
    const uint64_t pcie_base =
        (static_cast<uint64_t>(socket.h2d.data_addr_hi) << 32) | static_cast<uint64_t>(socket.h2d.data_addr_lo);

    socket_read(socket, pcie_base, tbuf);
    const uint32_t n = t[0], dst_bytes = t[1];
    ASSERT(dst_bytes <= MAX_DST && dst_bytes % PAGE == 0);
    for (uint32_t o = 0; o < dst_bytes; o += PAGE) {
        socket_read(socket, pcie_base, dbuf + o);
    }

    tt::tt_fabric::RoutingPlaneConnectionManager conn;
    open_connections(conn, 1, a);
    auto& slot = conn.get(0);
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    uint32_t cur = 0xffffffffu;

    for (uint32_t i0 = 0; i0 < n; i0 += RUNS) {
        socket_read(socket, pcie_base, tbuf);
        const uint32_t nr = (n - i0) < RUNS ? (n - i0) : RUNS;
        for (uint32_t r = 0; r < nr; ++r) {
            const uint32_t w = t[4 * r], s = w & 0xff, count = (w >> 8) & 0xff, j = w >> 16;
            const uint32_t sp = t[4 * r + 1], base = t[4 * r + 2], dp = t[4 * r + 3];
            volatile tt_l1_ptr uint32_t* dst = d + j * DST_WORDS;
            if (j != cur) {
                fabric_set_unicast_route(hdr, dst[0], dst[1]);
                cur = j;
            }
            const Source src{
                get_arg_val<uint32_t>(5 * s),
                get_arg_val<uint32_t>(5 * s + 1),
                get_arg_val<uint32_t>(5 * s + 2),
                get_arg_val<uint32_t>(5 * s + 3),
                get_arg_val<uint32_t>(5 * s + 4)};
            switch (src.cls) {
                case 0: run<CLASS_CTA>(src, sp, count, slot.sender, hdr, dst, base, dp, buf); break;
                case 1: run<CLASS_CTA + 2>(src, sp, count, slot.sender, hdr, dst, base, dp, buf); break;
                case 2: run<CLASS_CTA + 4>(src, sp, count, slot.sender, hdr, dst, base, dp, buf); break;
                default: run<CLASS_CTA + 6>(src, sp, count, slot.sender, hdr, dst, base, dp, buf); break;
            }
        }
    }
    close_connections(conn);
    update_socket_config(socket);
    noc_async_write_barrier();
}
