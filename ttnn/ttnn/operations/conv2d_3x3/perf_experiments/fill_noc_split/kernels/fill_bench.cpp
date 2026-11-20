// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// fill_noc_split bench: the conv2d_3x3 ROW_MAJOR interleaved slab fill in isolation. Per M-block a core reads the
// three kernel-row bands of its slab, one DRAM page (one input pixel) per NoC read, into L1, and waits for them to
// land before it starts the next M-block (in the op the slab is refilled only after compute popped every entry).
// The same source runs on NCRISC (reader, NoC0) and, for the split variants, on BRISC (writer, NoC1). The host plans
// every M-block's band pieces per RISC (runtime args: page0, count, slab pixel), so both RISCs place every pixel at
// the slab pixel the op's slab_bands puts it.
//
// Split handshake (the op's HEIGHT RM SLAB_SPLIT_NOCS protocol): the reader raises slab_free when it starts block m
// (the writer may now write block m into the slab), the writer raises slab_fwd_done once its part of block m landed,
// and the reader waits for that before block m counts as filled.
//
// Address modes (raw NoC command-buffer writes for 1-3: there is no helper for a stateful one-packet read whose
// coordinate changes per read; the TensorAccessor lookup per page is what mode 0 measures):
//   0  TensorAccessor::get_noc_addr per page + noc_async_read (the op's read_pages loop, TaggedReads throttle).
//   1  incremental: the 8 bank bases from the accessor once; per read only RET_ADDR_LO, TARG_ADDR_LO, the bank
//      coordinate and CMD_CTRL (the one-packet length is set once).
//   2  bank-major: same as 1, but each band is read bank by bank (pages p, p + 8, ...), so a run of reads keeps its
//      coordinate and only the two addresses change (stateful read).
//   3  diagnostic: mode 1's issue loop with the source in this core's own L1 (no DRAM): the issue / NIU ceiling.
// window: 0 = the op's throttle (every 64 reads, wait for 128 free ids on the trid counter); N = at most N reads in
// flight on this RISC.
// Store emulation (writer, store_tiles > 0): at the start of every block after the first the writer stores
// store_tiles output tiles (block m - 1's output, written while block m fills in the op): store_mode 0 before its own
// fill part, 1 after it, 2 spread over it (a share of the tiles after every 64 reads, the rest after the part).
// Dynamic split (dyn_chunk > 0): both RISCs get the block's whole pieces, cut into chunks of dyn_chunk pixels; the
// reader claims chunks from the front, the writer from the back, each publishing its claim count (tagged with the
// block) in its own L1 word. The reader publishes before it checks the writer's count and stops at the writer's first
// claimed chunk; the writer checks the reader's count before it publishes and stops DYN_GAP chunks short of it. So no
// chunk is left unread (the reader runs until the writer's claims), and a chunk is read twice only if the reader
// advanced DYN_GAP chunks between the writer's check and its publish (harmless: same bytes, same slab pixels).
// Lock-step (lockstep): every core's reader, once its block is complete, bumps a counter on the master core (core 0),
// which releases every core's next block when all are in (the op's weight multicast couples the cores' blocks the same
// way: cb_weights is two rounds deep and every receiver acks every round, so no core runs a block ahead).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/semaphore.h"

namespace {
constexpr uint32_t TRID = 1;
constexpr uint32_t ARGS_PER_PIECE = 3;
constexpr uint32_t PIECES_PER_BLOCK = 3;
constexpr uint32_t BASE_ARGS = 11;
constexpr uint32_t DYN_GAP = 2;
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_slab = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t addr_mode = get_compile_time_arg_val(2);
    constexpr uint32_t window = get_compile_time_arg_val(3);
    constexpr uint32_t role = get_compile_time_arg_val(4);  // 0 reader, 1 writer
    constexpr bool split = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t fwd_sem = get_compile_time_arg_val(6);
    constexpr uint32_t free_sem = get_compile_time_arg_val(7);
    constexpr uint32_t store_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t store_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t store_mode = get_compile_time_arg_val(10);
    constexpr bool lockstep = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t bar_sem = get_compile_time_arg_val(12);
    constexpr uint32_t go_sem = get_compile_time_arg_val(13);
    // arg 14: the variant's index (keeps every variant its own cached program)
    constexpr uint32_t dyn_chunk = get_compile_time_arg_val(15);  // dynamic split: chunk pixels (0 = static pieces)
    constexpr uint32_t claim_r_sem = get_compile_time_arg_val(16);
    constexpr uint32_t claim_w_sem = get_compile_time_arg_val(17);
    constexpr auto in_args = TensorAccessorArgs<18>();
    constexpr auto dump_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto store_args = TensorAccessorArgs<dump_args.next_compile_time_args_offset()>();

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t dump_addr = get_arg_val<uint32_t>(1);
    const uint32_t store_addr = get_arg_val<uint32_t>(2);
    const uint32_t dump_page0 = get_arg_val<uint32_t>(3);
    const uint32_t slab_pages = get_arg_val<uint32_t>(4);
    const uint32_t num_blocks = get_arg_val<uint32_t>(5);
    const uint32_t store_page0 = get_arg_val<uint32_t>(6);
    const uint32_t bank_rot = get_arg_val<uint32_t>(7);
    const uint32_t master_xy = get_arg_val<uint32_t>(8);  // (x << 16) | y of the master core, this NoC
    const uint32_t num_peers = get_arg_val<uint32_t>(9);  // master: cores it releases (their xy follow the blocks)
    const uint32_t peers_arg = get_arg_val<uint32_t>(10);
    if (num_blocks == 0) {
        return;
    }

    const auto input = TensorAccessor(in_args, in_addr, page_bytes);
    const uint32_t slab_l1 = get_write_ptr(cb_slab);
    const uint8_t noc = noc_index;

    // Bank bases (page b is bank b's first page) for the raw modes.
    uint32_t bank_lo[8];
    uint32_t bank_coord[8];
    for (uint32_t b = 0; b < 8; ++b) {
        uint64_t a = input.get_noc_addr(b);
        if constexpr (addr_mode == 3) {
            a = get_noc_addr(slab_l1);
        }
        bank_lo[b] = static_cast<uint32_t>(a);
        bank_coord[b] = static_cast<uint32_t>(a >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
    }
    noc_async_read_set_trid(TRID, noc);
    if constexpr (addr_mode != 0) {
        noc_async_read_one_packet_set_state(input.get_noc_addr(0), page_bytes, 0, noc);
    }
    uint32_t since = 0;
    auto throttle = [&]() {
        if constexpr (window == 0) {
            if (++since >= 64) {
                since = 0;
                while (noc_available_transactions(noc, TRID) < (NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2) {
                }
            }
        } else {
            while (NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(TRID)) >= window) {
            }
        }
    };
    auto raw_read = [&](uint32_t coord, uint32_t lo, uint32_t l1, bool new_coord) {
        throttle();
        while (!noc_cmd_buf_ready(noc, read_cmd_buf)) {
        }
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, l1);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, lo);
        if (new_coord) {
            NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, coord);
        }
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[noc] += 1;
    };

    // store_mode 2: after every 64 reads, a share of the pending output tiles.
    uint32_t reads_since_store = 0;
    uint32_t tiles_per_chunk = 0;
    [[maybe_unused]] const auto store_acc = TensorAccessor(store_args, store_addr, store_tile_bytes);
    uint32_t stored = store_tiles;
    auto store_some = [&](uint32_t n) {
        if constexpr (store_tiles > 0) {
            const uint32_t end = stored + n < store_tiles ? stored + n : store_tiles;
            for (; stored < end; ++stored) {
                noc_async_write(slab_l1, store_acc.get_noc_addr(store_page0 + stored), store_tile_bytes, noc);
            }
        }
    };
    auto after_read = [&]() {
        if constexpr (store_mode == 2 && store_tiles > 0) {
            if (++reads_since_store == 64) {
                reads_since_store = 0;
                store_some(tiles_per_chunk);
            }
        }
    };

    // Pages [page, page + count) into slab pixels from slab_px.
    auto read_piece = [&](uint32_t page, uint32_t count, uint32_t slab_px) {
        uint32_t l1 = slab_l1 + slab_px * page_bytes;
        if constexpr (addr_mode == 0) {
            for (uint32_t i = 0; i < count; ++i) {
                throttle();
                noc_async_read(input.get_noc_addr(page + i), l1, page_bytes, noc);
                l1 += page_bytes;
                after_read();
            }
        } else if constexpr (addr_mode == 1 || addr_mode == 3) {
            uint32_t b = page & 7;
            uint32_t off = (page >> 3) * page_bytes;
            if constexpr (addr_mode == 3) {
                off = 0;
            }
            for (uint32_t i = 0; i < count; ++i) {
                if constexpr (addr_mode == 3) {
                    raw_read(bank_coord[0], bank_lo[0] + (off & 0xFFFF), l1, i == 0);
                    off += page_bytes;
                } else {
                    raw_read(bank_coord[b], bank_lo[b] + off, l1, true);
                    after_read();
                    if (++b == 8) {
                        b = 0;
                        off += page_bytes;
                    }
                }
                l1 += page_bytes;
            }
        } else {
            const uint32_t end = page + count;
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t b = (bank_rot + j) & 7;
                uint32_t p = page + ((b - page) & 7);
                if (p >= end) {
                    continue;
                }
                uint32_t lo = bank_lo[b] + (p >> 3) * page_bytes;
                uint32_t dst = l1 + (p - page) * page_bytes;
                bool first = true;
                for (; p < end; p += 8) {
                    raw_read(bank_coord[b], lo, dst, first);
                    first = false;
                    lo += page_bytes;
                    dst += 8 * page_bytes;
                }
            }
        }
    };

    auto store_block = [&]() {
        if constexpr (store_tiles > 0) {
            store_some(store_tiles);
            noc_async_writes_flushed(noc);
        }
    };

    {
        DeviceZoneScopedN("fill_loop");
        for (uint32_t m = 0; m < num_blocks; ++m) {
            if constexpr (role == 0) {
                if constexpr (split || store_tiles > 0) {
                    Semaphore<>(free_sem).up(1);
                }
            } else {
                Semaphore<>(free_sem).wait_min(m + 1);
                if (m > 0) {
                    stored = 0;
                    if constexpr (store_mode == 0) {
                        store_block();
                    } else if constexpr (store_mode == 2) {
                        uint32_t part = 0;
                        for (uint32_t k = 0; k < PIECES_PER_BLOCK; ++k) {
                            part += get_arg_val<uint32_t>(BASE_ARGS + (m * PIECES_PER_BLOCK + k) * ARGS_PER_PIECE + 1);
                        }
                        if constexpr (dyn_chunk > 0) {
                            part /= 2;  // the writer's expected share
                        }
                        const uint32_t chunks = part / 64 + 1;
                        tiles_per_chunk = (store_tiles + chunks - 1) / chunks;
                        reads_since_store = 0;
                    }
                }
            }
            {
                DeviceZoneScopedN("own");
                if constexpr (dyn_chunk > 0) {
                    uint32_t pg[PIECES_PER_BLOCK], cnt[PIECES_PER_BLOCK], px[PIECES_PER_BLOCK], nch[PIECES_PER_BLOCK];
                    uint32_t num = 0;
                    for (uint32_t k = 0; k < PIECES_PER_BLOCK; ++k) {
                        const uint32_t a = BASE_ARGS + (m * PIECES_PER_BLOCK + k) * ARGS_PER_PIECE;
                        pg[k] = get_arg_val<uint32_t>(a);
                        cnt[k] = get_arg_val<uint32_t>(a + 1);
                        px[k] = get_arg_val<uint32_t>(a + 2);
                        nch[k] = (cnt[k] + dyn_chunk - 1) / dyn_chunk;
                        num += nch[k];
                    }
                    auto chunk = [&](uint32_t c) {
                        uint32_t k = 0;
                        while (c >= nch[k]) {
                            c -= nch[k];
                            ++k;
                        }
                        const uint32_t off = c * dyn_chunk;
                        const uint32_t n = cnt[k] - off < dyn_chunk ? cnt[k] - off : dyn_chunk;
                        read_piece(pg[k] + off, n, px[k] + off);
                    };
                    volatile tt_l1_ptr uint32_t* rpub =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(claim_r_sem));
                    volatile tt_l1_ptr uint32_t* wpub =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(claim_w_sem));
                    const uint32_t tag = (m + 1) << 16;
                    if constexpr (role == 0) {
                        for (uint32_t r = 0;; ++r) {
                            *rpub = tag | (r + 1);
                            invalidate_l1_cache();
                            const uint32_t wv = *wpub;
                            const uint32_t w = (wv & 0xFFFF0000u) == tag ? (wv & 0xFFFFu) : 0;
                            if (r + w >= num) {
                                break;
                            }
                            chunk(r);
                        }
                    } else {
                        for (uint32_t w = 0; w < num; ++w) {
                            invalidate_l1_cache();
                            const uint32_t rv = *rpub;
                            const uint32_t r = (rv & 0xFFFF0000u) == tag ? (rv & 0xFFFFu) : 0;
                            const uint32_t c = num - 1 - w;
                            if (c < r + DYN_GAP) {
                                break;
                            }
                            *wpub = tag | (w + 1);
                            chunk(c);
                        }
                    }
                } else {
                    for (uint32_t k = 0; k < PIECES_PER_BLOCK; ++k) {
                        const uint32_t a = BASE_ARGS + (m * PIECES_PER_BLOCK + k) * ARGS_PER_PIECE;
                        const uint32_t count = get_arg_val<uint32_t>(a + 1);
                        if (count > 0) {
                            read_piece(get_arg_val<uint32_t>(a), count, get_arg_val<uint32_t>(a + 2));
                        }
                    }
                }
                while (!ncrisc_noc_read_with_transaction_id_flushed(noc, TRID)) {
                }
            }
            if constexpr (role == 0) {
                if constexpr (split) {
                    Semaphore<>(fwd_sem).wait_min(m + 1);
                }
                if constexpr (lockstep) {
                    noc_semaphore_inc(
                        get_noc_addr(master_xy >> 16, master_xy & 0xFFFF, get_semaphore(bar_sem)), 1, noc);
                    if (num_peers > 0) {
                        noc_semaphore_wait_min(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(bar_sem)),
                            num_peers * (m + 1));
                        for (uint32_t j = 0; j < num_peers; ++j) {
                            const uint32_t xy = get_arg_val<uint32_t>(peers_arg + j);
                            noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(go_sem)), 1, noc);
                        }
                    }
                    noc_semaphore_wait_min(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(go_sem)), m + 1);
                }
            } else {
                if constexpr (split) {
                    Semaphore<>(fwd_sem).up(1);
                }
                if constexpr (store_mode != 0) {
                    store_block();  // the rest (mode 2) or all (mode 1) of the pending block's output
                }
            }
        }
    }
    noc_async_read_barrier(noc);
    noc_async_atomic_barrier(noc);
    if constexpr (role == 0) {
        // Untimed: the last block's slab to this core's dump pages, for the host's bit-exact check.
        const auto dump = TensorAccessor(dump_args, dump_addr, page_bytes);
        for (uint32_t i = 0; i < slab_pages; ++i) {
            noc_async_write(slab_l1 + i * page_bytes, dump.get_noc_addr(dump_page0 + i), page_bytes, noc);
        }
        noc_async_write_barrier(noc);
    } else {
        noc_async_write_barrier(noc);
    }
}
