// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm reader (regimes R1 and R3, op_design.md -> Dataflow Strategy).
//
// γ is NOT read here: the writer supplies cb_gamma_tiles (perf lamp L4), so this RISC-V only moves x
// and can prefetch block k+1 while block k is scaled (x_depth = 2).
//
// Every core owns a W-slice [w0, w0 + V) of its tile-rows (R1: the whole row, w0 = 0, V = Wt).
// Prologue (once per kernel):
//   - cb_reduce_scaler <- 1.0 (calculate_and_prepare_reduce_scaler<SUM, REDUCE_ROW>); with the sharded W-pad
//     mask, the [full, partial] pair (calculate_and_prepare_partial_reduce_scalers)
//   - interleaved TILE: zero the pad tiles [V, Wp) of every tile-row slot of cb_x_tiles once (ring positions
//     are fixed because every tile-row is pushed as exactly Wp tiles and capacity % Wp == 0)
//   - sharded: cb_x_tiles IS the input shard (zero-copy) — push all of this core's rows once, no NoC reads
// Per block of R_b tile-rows:
//   load_block (interleaved): per tile-row, per C-tile chunk, push exactly C pages (nominal quanta):
//     TILE -> cb_x_tiles   (valid slice tiles read, W-pad columns of tile Wt-1 zeroed in L1)
//     RM   -> cb_x_rm_sticks (interleaved R1: full chunks via read_sticks_for_tilize; the tail chunk is custom:
//             its stride / page count must stay C-wide and its pad bytes must be zeroed)
//   load_block (ROW_MAJOR shard, R3): the core's own shard sticks (local L1) re-laid into cb_x_rm_sticks at
//     the C*32-element tilize stride, slice pad bytes zeroed — the shard is element-granular (slices of 8 /
//     24 / ... elements, 205-stick shards), so tilize cannot read it in place
//   combine_block (R3, G > 1): the cross-core Σx² combine of the row-group (op_design.md -> Deferred-regime
//     Tensix-to-Tensix contracts, regime R3; catalog tensix_all_reduce: root reduce + one multicast back):
//     contributor: cb_partial_col (R_b tiles) -> root's cb_gather slots [index*R_b, +R_b) (one write), barrier,
//                  increment the root's gather semaphore; then receive the inv_rms multicast into cb_inv_rms
//     root:        own partial -> its gather slots (local copy), wait for G-1 arrivals, push cb_gather to
//                  compute; wait cb_rms_root and multicast it into every member's cb_inv_rms (itself included)
//     Slot safety: a contributor only writes block k+1's partial after receiving block k's inv_rms, which the
//     root only sends after its compute has consumed block k's gather. Every core of a group walks the same
//     block sequence, so the gather / inv_rms write pointers stay aligned (only a group's last block is short).
//   combine_block (R3, reduce-scatter owners > 1): the same exchange with the block's rows split over owners —
//     see scatter_block.
//
// Raw NoC / CB calls: no kernel_lib helper reads TILE pages, zeroes partial columns, reads a C-wide-stride
// tail chunk, or gathers tiles to a root (op_design.md -> "Helpers considered and rejected"); the inv_rms
// broadcast uses the kernel_lib multicast channel (McastArgs).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

// Measurement-only payload ablation (host RMS_NORM_ABLATE; 0 in production): the NoC data movement is skipped,
// every CB / semaphore / barrier handshake is kept.
#ifndef RMS_ABLATE_READS
#define RMS_ABLATE_READS 0
#endif
#ifndef RMS_ABLATE_COMBINE
#define RMS_ABLATE_COMBINE 0
#endif

// Stage zones (permanent; MaybeDeviceZoneScope, kernel_lib/perf_instrumentation.hpp) — waits are zoned apart from
// payload so a starved stage is not read as an expensive one:
//   rd_prologue                          scaler fill + pad-tile zeroing
//   rd_load (per block)                  the whole interleaved block load; inside, per read batch:
//     rd_reserve / rd_issue / rd_barrier   CB space (back-pressure from compute) / issue loop / landing wait
//   R3 combine, per block:
//     rd_wait_partial                    compute's collapsed partial (cb_partial_col)
//     rd_send                            partial writes + write barrier + semaphore increments
//     rd_gather_wait                     (root / owner) the other members' partials
//     rd_wait_root                       (root / owner) compute's finalized inv_rms (cb_rms_root)
//     rd_allgather                       (root / owner) inv_rms multicast / unicast all-gather + barrier
//     rd_wait_inv                        the group's inv_rms rows landing in cb_inv_rms
namespace {

constexpr uint32_t cb_x_rm_sticks = 0;
constexpr uint32_t cb_x_tiles = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_inv_rms = 4;
constexpr uint32_t cb_partial_col = 9;
constexpr uint32_t cb_gather = 10;
constexpr uint32_t cb_rms_root = 11;
constexpr uint32_t cb_ag_token = 15;  // combine_pipeline AG_WRITER: writer -> reader "wave all-gathered" tokens

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t L1_ALIGN = 16;  // NOC_L1_READ_ALIGNMENT_BYTES for the zero loopback

// Bank stagger (perf lamp L9) — compile-time switch RMS_BANK_STAGGER (host define, 0/1). On: a walk over
// [0, n) starts at this core's bank offset s (< NUM_DRAM_BANKS) and wraps — same transactions, same L1
// slots, only the issue order rotates, so cores in lockstep do not all hit one DRAM bank (every tile-row
// starts on the same bank when Wt % NUM_DRAM_BANKS == 0). Off: the plain in-order loop, no extra work.
constexpr bool bank_stagger_on = RMS_BANK_STAGGER != 0;
FORCE_INLINE uint32_t stagger_first(uint32_t s, uint32_t n) {
    if constexpr (bank_stagger_on) {
        return s < n ? s : s % n;
    }
    return 0;
}
FORCE_INLINE uint32_t stagger_next(uint32_t j, uint32_t n) {
    if constexpr (bank_stagger_on) {
        return (j + 1 == n) ? 0 : j + 1;
    }
    return j + 1;
}

// Element-sized store of zero over [addr, addr + n_bytes) — only used for < 16 B unaligned heads
// and for partial face rows (<= 15 elements).
template <uint32_t es>
FORCE_INLINE void store_zeros(uint32_t addr, uint32_t n_elems) {
    if constexpr (es == 4) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        for (uint32_t i = 0; i < n_elems; ++i) {
            p[i] = 0;
        }
    } else {
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
        for (uint32_t i = 0; i < n_elems; ++i) {
            p[i] = 0;
        }
    }
}

// Zero columns [w_valid, 32) of a TILE-layout tile at L1 address tile_addr (face layout: 4 faces
// of 16x16, faces 0/1 = rows 0-15 cols 0-15 / 16-31, faces 2/3 = rows 16-31). Whole pad faces are
// zeroed by the NoC zero loopback; partial face rows by element stores.
template <uint32_t es>
FORCE_INLINE void zero_tile_w_padding(
    const Noc& noc, const CircularBuffer& cb, uint32_t tile_offset, uint32_t w_valid) {
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * es;
    constexpr uint32_t face_row_bytes = FACE_DIM * es;
    const uint32_t tile_addr = cb.get_write_ptr() + tile_offset;
    if (w_valid <= FACE_DIM) {
        // Faces 1 and 3 are entirely padding.
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset + 1 * face_bytes});
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset + 3 * face_bytes});
        if (w_valid < FACE_DIM) {
            const uint32_t n = FACE_DIM - w_valid;
            for (uint32_t f = 0; f < 4; f += 2) {
                for (uint32_t r = 0; r < FACE_DIM; ++r) {
                    store_zeros<es>(tile_addr + f * face_bytes + r * face_row_bytes + w_valid * es, n);
                }
            }
        }
        noc.write_zeros_l1_barrier();
    } else {
        const uint32_t c0 = w_valid - FACE_DIM;
        const uint32_t n = FACE_DIM - c0;
        for (uint32_t f = 1; f < 4; f += 2) {
            for (uint32_t r = 0; r < FACE_DIM; ++r) {
                store_zeros<es>(tile_addr + f * face_bytes + r * face_row_bytes + c0 * es, n);
            }
        }
    }
}

// Zero bytes [valid_bytes, chunk_stick_bytes) of staging stick s: element stores up to the next 16 B
// boundary, the NoC zero loopback for the rest. The caller issues write_zeros_l1_barrier.
template <uint32_t es, uint32_t chunk_stick_bytes>
FORCE_INLINE void zero_stick_tail(const Noc& noc, const CircularBuffer& cb, uint32_t s, uint32_t valid_bytes) {
    const uint32_t stick = s * chunk_stick_bytes;
    const uint32_t head_end = (valid_bytes + L1_ALIGN - 1) / L1_ALIGN * L1_ALIGN;
    if (head_end > valid_bytes) {
        store_zeros<es>(cb.get_write_ptr() + stick + valid_bytes, (head_end - valid_bytes) / es);
    }
    if (head_end < chunk_stick_bytes) {
        noc.async_write_zeros(cb, chunk_stick_bytes - head_end, {.offset_bytes = stick + head_end});
    }
}

// R3 combine of one block (see the file header): the contributor's side ...
template <uint32_t G, uint32_t T_interm, typename Pipe>
FORCE_INLINE void contribute_block(
    Pipe& receiver,
    Semaphore<>& gather_sem,
    const Noc& noc,
    uint32_t rows_in_block,
    uint32_t my_index,
    uint32_t root_x,
    uint32_t root_y) {
    CircularBuffer partial(cb_partial_col);
    CircularBuffer inv_rms(cb_inv_rms);
    // This core never pushes cb_gather, so its write pointer is the base every core shares — the
    // address of the root's gather slots.
    const uint32_t gather_base = CircularBuffer(cb_gather).get_write_ptr();
    {
        MaybeDeviceZoneScope("rd_wait_partial");
        partial.wait_front(rows_in_block);
    }
    {
        MaybeDeviceZoneScope("rd_send");
        // Contributor-major gather (compute reduces it with REDUCE_COL): this core's R_b partials are contiguous
        // slots [my_index * R_b, +R_b) of the root's cb_gather — one write.
        if constexpr (!RMS_ABLATE_COMBINE) {
            noc_async_write(
                partial.get_read_ptr(),
                get_noc_addr(root_x, root_y, gather_base + my_index * rows_in_block * T_interm),
                rows_in_block * T_interm);
        }
        noc_async_write_barrier();  // the partial has landed before the root can count it
        gather_sem.up(noc, root_x, root_y, 1);
    }
    partial.pop_front(rows_in_block);
    {
        MaybeDeviceZoneScope("rd_wait_inv");
        inv_rms.reserve_back(rows_in_block);  // the previous block's inv_rms is released before we ack
        receiver.receive();
    }
    inv_rms.push_back(rows_in_block);
}

// ... and the root's side.
template <uint32_t G, uint32_t T_interm, typename Pipe>
FORCE_INLINE void root_block(
    Pipe& sender, Semaphore<>& gather_sem, uint32_t block_idx, uint32_t rows_in_block, uint32_t my_index) {
    CircularBuffer partial(cb_partial_col);
    CircularBuffer gather(cb_gather);
    CircularBuffer rms_root(cb_rms_root);
    CircularBuffer inv_rms(cb_inv_rms);
    gather.reserve_back(G * rows_in_block);
    const uint32_t gather_base = gather.get_write_ptr();
    {
        MaybeDeviceZoneScope("rd_wait_partial");
        partial.wait_front(rows_in_block);
    }
    {
        MaybeDeviceZoneScope("rd_send");
        noc_async_read(
            get_noc_addr(partial.get_read_ptr()),
            gather_base + my_index * rows_in_block * T_interm,
            rows_in_block * T_interm);
        noc_async_read_barrier();
    }
    partial.pop_front(rows_in_block);
    {
        MaybeDeviceZoneScope("rd_gather_wait");
        gather_sem.wait_min((block_idx + 1) * (G - 1));  // every contributor's partial of this block landed
    }
    gather.push_back(G * rows_in_block);
    {
        MaybeDeviceZoneScope("rd_wait_root");
        rms_root.wait_front(rows_in_block);
        inv_rms.reserve_back(rows_in_block);
    }
    {
        MaybeDeviceZoneScope("rd_allgather");
        // Lands in every member's cb_inv_rms at the same address, this core's included.
        sender.send(rms_root.get_read_ptr(), inv_rms.get_write_ptr(), rows_in_block * T_interm);
    }
    inv_rms.push_back(rows_in_block);
    rms_root.pop_front(rows_in_block);
}

// R3 reduce-scatter combine of one block (OWNERS > 1, catalog tensix_all_reduce: reduce-scatter + all-gather).
// The block's R_b tile-rows are split into ragged runs of q = ceil(R_b / OWNERS) over the first owners of the
// row-group; owner k sums + finalizes rows [k*q, k*q + n_k). Every member (owner or not):
//   1. sends its partials of owner k's rows — contiguous in cb_partial_col — in ONE write to slots
//      [my_index*q, +n_k) of k's cb_gather (contributor-major; its own rows by a loopback write), write barrier,
//      then increments each OTHER owner's SEM_GATHER;
//   2. (owner) waits (G-1) arrivals, pushes the uniform G*q gather quantum (so every owner's write pointer
//      returns to the shared base), waits its n_k finalized rows (cb_rms_root) and writes them to row offset k*q
//      of EVERY member's cb_inv_rms (itself included), write barrier, increments each other member's SEM_INV.
//      Unicast, not multicast: G owners multicasting into one row-group concurrently measured 1.0-2.5x slower
//      and erratic across cores (8192x1024 BLOCK: inv_rms wait 7.7 k -> 7.9-19.3 k cycles);
//   3. reserves cb_inv_rms, waits the other owners' SEM_INV increments, pushes the block's R_b rows.
// Slot safety (why no ready handshake is needed): a member's partial of block b+1 exists only after its compute
// popped cb_inv_rms of block b, and owner k sends block b+1 rows only after every member's partial b+1 reached
// it, so every member's cb_inv_rms region is free. Likewise contributor j writes k's gather for b+1 only after j
// received k's rows of block b, which k's compute produced after unpacking all of k's gather b. Every member
// walks the same block sequence and full blocks push whole capacities, so cb_gather / cb_inv_rms sit at their
// bases at every block start — the addresses every member targets.
template <uint32_t G, uint32_t OWNERS, uint32_t T_interm, uint32_t BLOCK_ROWS>
FORCE_INLINE void scatter_block(
    const Noc& noc,
    Semaphore<>& gather_sem,
    Semaphore<>& inv_sem,
    uint32_t gather_base,
    uint32_t inv_base,
    uint32_t rms_root_base,
    uint32_t member_arg_base,
    uint32_t my_index,
    uint32_t rows_in_block,
    uint32_t& gather_expected,
    uint32_t& inv_expected) {
    CircularBuffer partial(cb_partial_col);
    CircularBuffer gather(cb_gather);
    CircularBuffer rms_root(cb_rms_root);
    CircularBuffer inv_rms(cb_inv_rms);
    // Row-group member k's virtual coordinates (owner k = member k).
    auto member_x = [member_arg_base](uint32_t k) { return get_arg_val<uint32_t>(member_arg_base + k) >> 16; };
    auto member_y = [member_arg_base](uint32_t k) { return get_arg_val<uint32_t>(member_arg_base + k) & 0xFFFF; };
    const uint32_t q = (rows_in_block + OWNERS - 1) / OWNERS;
    const uint32_t num_owners = (rows_in_block + q - 1) / q;
    const bool is_owner = my_index < num_owners;
    const uint32_t my_first = my_index * q;
    const uint32_t my_rows = !is_owner ? 0 : (rows_in_block - my_first < q ? rows_in_block - my_first : q);

    // 1. partials -> owners
    {
        MaybeDeviceZoneScope("rd_wait_partial");
        if (is_owner) {
            gather.reserve_back(q * G);  // compute released this owner's previous gather block
        }
        partial.wait_front(rows_in_block);
    }
    {
        MaybeDeviceZoneScope("rd_send");
        const uint32_t src = partial.get_read_ptr();
        const uint32_t slot = gather_base + my_index * q * T_interm;
        for (uint32_t k = 0; k < num_owners; ++k) {
            const uint32_t first = k * q;
            const uint32_t n = rows_in_block - first < q ? rows_in_block - first : q;
            if constexpr (!RMS_ABLATE_COMBINE) {
                noc_async_write(src + first * T_interm, get_noc_addr(member_x(k), member_y(k), slot), n * T_interm);
            }
        }
        noc_async_write_barrier();  // the partials have landed before any owner can count them
        for (uint32_t k = 0; k < num_owners; ++k) {
            if (k != my_index) {
                gather_sem.up(noc, member_x(k), member_y(k), 1);
            }
        }
    }
    partial.pop_front(rows_in_block);

    // 2. owner: sum + finalize (compute), then all-gather the finalized rows to the whole group
    if (is_owner) {
        gather_expected += G - 1;
        {
            MaybeDeviceZoneScope("rd_gather_wait");
            gather_sem.wait_min(gather_expected);
        }
        gather.push_back(q * G);
        {
            MaybeDeviceZoneScope("rd_wait_root");
            rms_root.wait_front(my_rows);
        }
        MaybeDeviceZoneScope("rd_allgather");
        // cb_rms_root holds q_full = ceil(BLOCK_ROWS / OWNERS) pages, but the LAST owner of a ragged split pushes
        // fewer rows per full block, so its read pointer drifts around the ring and this block's rows may wrap
        // past the ring end: send them as (up to) two contiguous runs. (It is popped one page at a time below:
        // a CB pointer wraps only when it lands exactly on the ring end, so the pop quantum must divide the
        // capacity.)
        constexpr uint32_t rms_root_end_page = (BLOCK_ROWS + OWNERS - 1) / OWNERS;
        const uint32_t rows_src = rms_root.get_read_ptr();
        const uint32_t to_end = (rms_root_base + rms_root_end_page * T_interm - rows_src) / T_interm;
        const uint32_t run0 = my_rows < to_end ? my_rows : to_end;
        const uint32_t dst = inv_base + my_first * T_interm;
        for (uint32_t m = 0; m < G; ++m) {
            const uint64_t member = get_noc_addr(member_x(m), member_y(m), dst);
            if constexpr (!RMS_ABLATE_COMBINE) {
                noc_async_write(rows_src, member, run0 * T_interm);
                if (run0 < my_rows) {
                    noc_async_write(rms_root_base, member + run0 * T_interm, (my_rows - run0) * T_interm);
                }
            }
        }
        noc_async_write_barrier();
        for (uint32_t m = 0; m < G; ++m) {
            if (m != my_index) {
                inv_sem.up(noc, member_x(m), member_y(m), 1);
            }
        }
        for (uint32_t r = 0; r < my_rows; ++r) {
            rms_root.pop_front(1);
        }
    }

    // 3. every owner's rows of this block are in cb_inv_rms
    {
        MaybeDeviceZoneScope("rd_wait_inv");
        inv_rms.reserve_back(rows_in_block);
        inv_expected += num_owners - (is_owner ? 1 : 0);
        inv_sem.wait_min(inv_expected);
    }
    inv_rms.push_back(rows_in_block);
}

// combine_pipeline: the reduce-scatter combine of one block in WAVES (compute side: rms_norm_compute.cpp, WAVE_ROWS).
// Wave k = tile-rows [k*Rw, +n_k) of the block; owner i of the wave sums rows [i*q_w, +q_w) of it. Per wave the
// exchange is the one-shot scatter_block's, on its own semaphore pair (gather sem GS+k, inv sem IS+k): with one
// counter, "count >= (G-1)" could not tell wave k's arrivals from a faster member's wave k+1. The four steps run
// as a POLL LOOP (one RISC-V, non-blocking checks in priority order), so whichever is ready proceeds:
//   A. send wave ks's partials (compute pushed them) to the wave's owners, write barrier, gather-sem increments
//   B. (holder) wave kp's gather complete -> push the G*q_w quantum (a holder that owns no row of a short wave
//      pushes it unfilled: every holder walks the same push sequence, so cb_gather is at base at block start)
//   C. (owner of wave ka) compute finalized its rows -> unicast them into every member's cb_inv_rms at row
//      k*Rw + i*q_w, write barrier, inv-sem increments
//   D. wave ki's rows from every owner landed -> push n_ki pages of cb_inv_rms (compute waits cumulatively)
// Slot safety is the one-shot argument per block (no address is reused within a block: every wave has its own
// cb_gather region [k*G*q_w, +G*q_w) and cb_inv_rms rows; the next block's writes need this block's whole
// exchange done, see scatter_block). Deadlock freedom: A depends only on this core's compute, B on other cores'
// A, C on this core's compute (combine of wave ka, which needs B), D on other cores' C; compute consumes in
// wave order and every RISC-V step makes progress whenever its dependency is met (no blocking wait in the loop).
template <
    uint32_t G,
    uint32_t T_interm,
    uint32_t BLOCK_ROWS,
    uint32_t RW_FULL,
    uint32_t QW_FULL,
    uint32_t WAVES_FULL,
    bool AG_WRITER>
FORCE_INLINE void pipe_block(
    const Noc& noc,
    uint32_t gather_sem_base,
    uint32_t inv_sem_base,
    uint32_t gather_base,
    uint32_t inv_base,
    uint32_t rms_root_base,
    uint32_t member_arg_base,
    uint32_t my_index,
    uint32_t rows_in_block,
    uint32_t* gather_expected,
    uint32_t* inv_expected) {
    CircularBuffer partial(cb_partial_col);
    CircularBuffer gather(cb_gather);
    CircularBuffer rms_root(cb_rms_root);
    CircularBuffer inv_rms(cb_inv_rms);
    auto member_x = [member_arg_base](uint32_t k) { return get_arg_val<uint32_t>(member_arg_base + k) >> 16; };
    auto member_y = [member_arg_base](uint32_t k) { return get_arg_val<uint32_t>(member_arg_base + k) & 0xFFFF; };
    constexpr uint32_t H = (RW_FULL + QW_FULL - 1) / QW_FULL;  // gather-slot holders
    constexpr uint32_t rms_root_pages = WAVES_FULL * QW_FULL;
    const bool holder = my_index < H;
    // Full blocks: the host's wave geometry; a short last block re-splits over the same holders (mirrors compute).
    uint32_t QW = QW_FULL;
    if (rows_in_block != BLOCK_ROWS) {
        const uint32_t q = ((rows_in_block + H - 1) / H + WAVES_FULL - 1) / WAVES_FULL;
        QW = q < QW_FULL ? q : QW_FULL;
    }
    const uint32_t RW = QW * H;
    const uint32_t waves = (rows_in_block + RW - 1) / RW;
    auto wave_rows = [&](uint32_t k) { return (rows_in_block - k * RW) < RW ? rows_in_block - k * RW : RW; };
    auto wave_owners = [&](uint32_t k) { return (wave_rows(k) + QW - 1) / QW; };
    auto owned_rows = [&](uint32_t k) -> uint32_t {
        const uint32_t n = wave_rows(k);
        const uint32_t first = my_index * QW;
        return first >= n ? 0 : ((n - first) < QW ? n - first : QW);
    };
    auto sem_value = [](uint32_t id) { return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    uint32_t ks = 0, kp = 0, ka = 0, ki = 0;
    // A. partials of wave ks -> the wave's owners
    auto try_send = [&]() -> bool {
        if (ks >= waves || !partial.pages_available_at_front(wave_rows(ks))) {
            return false;
        }
        MaybeDeviceZoneScope("rd_send");
        const uint32_t n = wave_rows(ks);
        const uint32_t nown = wave_owners(ks);
        const uint32_t src = partial.get_read_ptr();
        const uint32_t slot = gather_base + (ks * G + my_index) * QW * T_interm;
        for (uint32_t o = 0; o < nown; ++o) {
            const uint32_t first = o * QW;
            const uint32_t m = (n - first) < QW ? n - first : QW;
            if constexpr (!RMS_ABLATE_COMBINE) {
                noc_async_write(src + first * T_interm, get_noc_addr(member_x(o), member_y(o), slot), m * T_interm);
            }
        }
        noc_async_write_barrier();
        const uint32_t sem_addr = get_semaphore(gather_sem_base + ks);
        for (uint32_t o = 0; o < nown; ++o) {
            if (o != my_index) {
                noc_semaphore_inc(get_noc_addr(member_x(o), member_y(o), sem_addr), 1);
            }
        }
        partial.pop_front(n);
        ++ks;
        return true;
    };
    // B. (holder) wave kp's gather complete -> compute (cheap: a semaphore read + a push)
    auto try_push_gather = [&]() -> bool {
        if (kp >= ks) {
            return false;
        }
        if (!holder) {
            kp = ks;
            return true;
        }
        if (owned_rows(kp) > 0) {
            if (sem_value(gather_sem_base + kp) < gather_expected[kp] + (G - 1)) {
                return false;
            }
            gather_expected[kp] += G - 1;
        }
        gather.reserve_back(G * QW);
        gather.push_back(G * QW);
        ++kp;
        return true;
    };
    // C. (owner of wave ka) all-gather the finalized rows
    auto try_allgather = [&]() -> bool {
        if (ka >= kp) {
            return false;
        }
        if constexpr (AG_WRITER) {
            // The writer RISC-V all-gathers (rms_norm_writer.cpp allgather_block) and pushes one token per owned
            // wave once its writes have landed; this RISC-V issues the arrival increments (NCRISC atomics).
            const uint32_t mine_w = owned_rows(ka);
            if (mine_w == 0) {
                ++ka;
                return true;
            }
            CircularBuffer token(cb_ag_token);
            if (!token.pages_available_at_front(1)) {
                return false;
            }
            token.pop_front(1);
            const uint32_t sem_addr_w = get_semaphore(inv_sem_base + ka);
            for (uint32_t m = 0; m < G; ++m) {
                if (m != my_index) {
                    noc_semaphore_inc(get_noc_addr(member_x(m), member_y(m), sem_addr_w), 1);
                }
            }
            ++ka;
            return true;
        }
        const uint32_t mine = owned_rows(ka);
        if (mine == 0) {
            ++ka;
            return true;
        }
        if (!rms_root.pages_available_at_front(mine)) {
            return false;
        }
        MaybeDeviceZoneScope("rd_allgather");
        const uint32_t rows_src = rms_root.get_read_ptr();
        const uint32_t to_end = (rms_root_base + rms_root_pages * T_interm - rows_src) / T_interm;
        const uint32_t run0 = mine < to_end ? mine : to_end;
        const uint32_t dst = inv_base + (ka * RW + my_index * QW) * T_interm;
        for (uint32_t m = 0; m < G; ++m) {
            const uint64_t member = get_noc_addr(member_x(m), member_y(m), dst);
            if constexpr (!RMS_ABLATE_COMBINE) {
                noc_async_write(rows_src, member, run0 * T_interm);
                if (run0 < mine) {
                    noc_async_write(rms_root_base, member + run0 * T_interm, (mine - run0) * T_interm);
                }
            }
        }
        noc_async_write_barrier();
        const uint32_t sem_addr = get_semaphore(inv_sem_base + ka);
        for (uint32_t m = 0; m < G; ++m) {
            if (m != my_index) {
                noc_semaphore_inc(get_noc_addr(member_x(m), member_y(m), sem_addr), 1);
            }
        }
        for (uint32_t r = 0; r < mine; ++r) {
            rms_root.pop_front(1);
        }
        ++ka;
        return true;
    };
    // D. wave ki's rows from every owner have landed (this core's own rows: ki < ka, its C step is done)
    auto try_push_inv = [&]() -> bool {
        if (ki >= ka) {
            return false;
        }
        const uint32_t senders = wave_owners(ki) - (owned_rows(ki) > 0 ? 1 : 0);
        if (sem_value(inv_sem_base + ki) < inv_expected[ki] + senders) {
            return false;
        }
        inv_expected[ki] += senders;
        const uint32_t n = wave_rows(ki);
        inv_rms.reserve_back(n);
        inv_rms.push_back(n);
        ++ki;
        return true;
    };
    // Priority: the cheap hand-offs (D, B) drain first so a long NoC step never sits on a ready push; then one send
    // (others' combines wait on it), then one all-gather.
    while (ki < waves) {
        while (try_push_inv() || try_push_gather()) {
        }
        if (try_send()) {
            continue;
        }
        try_allgather();
    }
}

// Walk the blocks; with the R3 combine, each block's combine step runs in this core's role. Templated on
// the channel type so the role pipes are only instantiated when the multicast channel is present.
template <
    bool COMBINE,
    uint32_t G,
    uint32_t OWNERS,
    uint32_t BLOCK_ROWS,
    uint32_t T_interm,
    uint32_t GATHER_SEM_ID,
    uint32_t INV_SEM_ID,
    uint32_t WAVE_ROWS,
    uint32_t WAVE_Q,
    uint32_t WAVES_FULL,
    bool AG_WRITER,
    typename MC,
    typename ForEachBlock>
FORCE_INLINE void run_blocks(
    const MC& mcast,
    const Noc& noc,
    bool is_root,
    uint32_t my_index,
    uint32_t root_x,
    uint32_t root_y,
    uint32_t member_arg_base,
    ForEachBlock&& for_each_block) {
    if constexpr (COMBINE && OWNERS > 1 && WAVE_ROWS > 0) {
        static_assert(WAVES_FULL >= 1 && WAVES_FULL <= 8, "combine_pipeline: one semaphore pair per wave (16 max)");
        const uint32_t gather_base = CircularBuffer(cb_gather).get_write_ptr();
        const uint32_t inv_base = CircularBuffer(cb_inv_rms).get_write_ptr();
        const uint32_t rms_root_base = CircularBuffer(cb_rms_root).get_read_ptr();
        uint32_t gather_expected[WAVES_FULL] = {};
        uint32_t inv_expected[WAVES_FULL] = {};
        for_each_block([&](uint32_t, uint32_t rows_in_block) {
            pipe_block<G, T_interm, BLOCK_ROWS, WAVE_ROWS, WAVE_Q, WAVES_FULL, AG_WRITER>(
                noc,
                GATHER_SEM_ID,
                INV_SEM_ID,
                gather_base,
                inv_base,
                rms_root_base,
                member_arg_base,
                my_index,
                rows_in_block,
                gather_expected,
                inv_expected);
        });
    } else if constexpr (COMBINE && OWNERS > 1) {
        Semaphore<> gather_sem(GATHER_SEM_ID);
        Semaphore<> inv_sem(INV_SEM_ID);
        // Captured before any push: the CB bases, identical on every member (same allocation everywhere).
        const uint32_t gather_base = CircularBuffer(cb_gather).get_write_ptr();
        const uint32_t inv_base = CircularBuffer(cb_inv_rms).get_write_ptr();
        const uint32_t rms_root_base = CircularBuffer(cb_rms_root).get_read_ptr();
        uint32_t gather_expected = 0;
        uint32_t inv_expected = 0;
        for_each_block([&](uint32_t, uint32_t rows_in_block) {
            scatter_block<G, OWNERS, T_interm, BLOCK_ROWS>(
                noc,
                gather_sem,
                inv_sem,
                gather_base,
                inv_base,
                rms_root_base,
                member_arg_base,
                my_index,
                rows_in_block,
                gather_expected,
                inv_expected);
        });
    } else if constexpr (COMBINE) {
        Semaphore<> gather_sem(GATHER_SEM_ID);
        if (is_root) {
            auto sender = mcast.sender(noc);
            for_each_block([&](uint32_t block_idx, uint32_t rows_in_block) {
                root_block<G, T_interm>(sender, gather_sem, block_idx, rows_in_block, my_index);
            });
        } else {
            auto receiver = mcast.receiver(noc);
            for_each_block([&](uint32_t, uint32_t rows_in_block) {
                contribute_block<G, T_interm>(receiver, gather_sem, noc, rows_in_block, my_index, root_x, root_y);
            });
        }
    } else {
        for_each_block([](uint32_t, uint32_t) {});
    }
}

}  // namespace

void kernel_main() {
    // ---- compile-time args (shared geometry prefix, see rms_norm_program_descriptor.py) ----
    constexpr bool is_rm = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t block_rows = get_compile_time_arg_val(2);
    constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t W = get_compile_time_arg_val(6);
    constexpr uint32_t num_sticks = get_compile_time_arg_val(7);
    constexpr uint32_t x_depth = get_compile_time_arg_val(8);
    // TILE: C-tile chunks reserved/read per NoC read barrier (pushed together); >= 1, <= num_w_chunks.
    constexpr uint32_t read_batch_chunks = get_compile_time_arg_val(9);
    // CT args 10..13 are the writer's γ-supply args (shared prefix).
    constexpr bool sharded = get_compile_time_arg_val(14) != 0;
    constexpr bool combine = get_compile_time_arg_val(15) != 0;
    constexpr uint32_t x_row_stride = get_compile_time_arg_val(16);  // WS
    constexpr uint32_t group_size = get_compile_time_arg_val(17);    // G
    constexpr bool masked_cfg = get_compile_time_arg_val(18) != 0;
    constexpr uint32_t gather_sem_id = get_compile_time_arg_val(19);
    constexpr bool rm_sharded = get_compile_time_arg_val(20) != 0;
    constexpr uint32_t shard_row_bytes = get_compile_time_arg_val(21);  // ROW_MAJOR shard stick bytes
    constexpr auto input_args = TensorAccessorArgs<22>();
    // Reader-only CT args after the accessor: R3 combine owners (1 = root + multicast, > 1 = reduce-scatter +
    // unicast all-gather), the all-gather semaphore, and the runtime-arg index of the packed row-group member
    // coordinates.
    constexpr uint32_t reader_ct_base = input_args.next_compile_time_args_offset();
    constexpr uint32_t combine_owners = get_compile_time_arg_val(reader_ct_base);
    constexpr uint32_t inv_sem_id = get_compile_time_arg_val(reader_ct_base + 1);
    constexpr uint32_t member_arg_base = get_compile_time_arg_val(reader_ct_base + 2);
    // combine_pipeline: wave rows (0 = the one-shot scatter_block), per-owner rows per wave, waves per full block.
    constexpr uint32_t wave_rows = get_compile_time_arg_val(reader_ct_base + 3);
    constexpr uint32_t wave_q = get_compile_time_arg_val(reader_ct_base + 4);
    constexpr uint32_t waves_full = get_compile_time_arg_val(reader_ct_base + 5);
    constexpr bool ag_writer = get_compile_time_arg_val(reader_ct_base + 6) != 0;
    constexpr auto mcast = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();

    constexpr uint32_t C = w_chunk_tiles;
    constexpr uint32_t Wp = x_row_stride;  // interleaved: width_padded_tiles of the slice (num_w_chunks * C)
    constexpr uint32_t T_in = get_tile_size(cb_x_tiles);
    constexpr uint32_t T_interm = get_tile_size(cb_inv_rms);
    constexpr uint32_t es = T_in / (TILE_DIM * TILE_DIM);
    constexpr uint32_t w_valid_in_last_tile = W - TILE_DIM * (Wt - 1);
    // es is a per-datum byte count only for Float32 / Float16_b. A Bfp8_b tile (1088 B) is TILE-only and
    // tile-aligned in W (op EXCLUSION {bfloat8_b, w_non_aligned}): W-pad zeroing inside it would be
    // exponent-section work. Pad *tiles* are zeroed as raw bytes — a valid Bfp8_b zero tile.
    static_assert(
        T_in % (TILE_DIM * TILE_DIM) == 0 || (!is_rm && w_valid_in_last_tile == TILE_DIM),
        "Bfp8_b input: TILE layout with W % 32 == 0 only");
    static_assert(!is_rm || !sharded, "ROW_MAJOR shards are re-laid into the tilize staging, never zero-copy");
    static_assert(!is_rm || rm_sharded || !combine, "ROW_MAJOR interleaved input: regime R1 only");
    // RM: valid stick bytes in the last chunk of a row.
    constexpr uint32_t chunk_stick_bytes = C * TILE_DIM * es;
    constexpr uint32_t tail_stick_bytes = (W - (num_w_chunks - 1) * C * TILE_DIM) * es;
    constexpr bool tail_chunk_is_full = tail_stick_bytes == chunk_stick_bytes;

    // ---- runtime args ----
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t rows_this_core = get_arg_val<uint32_t>(2);
    // This core's DRAM bank offset (only read when RMS_BANK_STAGGER is on).
    const uint32_t bank_stagger = bank_stagger_on ? get_arg_val<uint32_t>(3) % NUM_DRAM_BANKS : 0;
    const uint32_t slice_w0 = get_arg_val<uint32_t>(4);     // first tile of this core's W-slice
    const uint32_t slice_valid = get_arg_val<uint32_t>(5);  // V: valid tiles of the slice
    const bool is_root = get_arg_val<uint32_t>(6) != 0;
    const uint32_t my_index = get_arg_val<uint32_t>(7);
    const uint32_t root_x = get_arg_val<uint32_t>(8);
    const uint32_t root_y = get_arg_val<uint32_t>(9);
    const uint32_t slice_elems = get_arg_val<uint32_t>(11);  // valid elements of the slice
    const uint32_t stick_count = get_arg_val<uint32_t>(12);  // ROW_MAJOR shard: valid sticks on this core
    // The slice holds the row's last tile, whose W-pad columns this reader zeroes (interleaved TILE).
    const bool holds_last_tile = slice_w0 + slice_valid == Wt;

    const auto input_accessor = TensorAccessor(input_args, input_addr);
    Noc noc;

    // ---- prologue ----
    {
        MaybeDeviceZoneScope("rd_prologue");
        if constexpr (masked_cfg) {
            dataflow_kernel_lib::calculate_and_prepare_partial_reduce_scalers<
                cb_reduce_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                w_valid_in_last_tile>();
        } else {
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                cb_reduce_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW>();
        }

        if constexpr (sharded) {
            // The shard is already resident: hand all of this core's rows to compute (no data movement).
            CircularBuffer x_cb(cb_x_tiles);
            x_cb.reserve_back(rows_this_core * x_row_stride);
            x_cb.push_back(rows_this_core * x_row_stride);
        } else if constexpr (!is_rm) {
            if (slice_valid < Wp) {
                // Zero the pad tiles of every tile-row slot of the x ring once; they are never overwritten.
                constexpr uint32_t ring_rows = x_depth * block_rows;
                CircularBuffer x_cb(cb_x_tiles);
                x_cb.reserve_back(ring_rows * Wp);
                for (uint32_t row = 0; row < ring_rows; ++row) {
                    noc.async_write_zeros(
                        x_cb, (Wp - slice_valid) * T_in, {.offset_bytes = (row * Wp + slice_valid) * T_in});
                }
                noc.write_zeros_l1_barrier();
            }
        }
    }  // rd_prologue

    // load_block (interleaved): R_b tile-rows of this core's slice.
    auto load_block = [&](uint32_t block_first_row, uint32_t rows_in_block) {
        for (uint32_t r = 0; r < rows_in_block; ++r) {
            const uint32_t tile_row = start_tile_row + block_first_row + r;
            if constexpr (!is_rm) {
                CircularBuffer x_cb(cb_x_tiles);
                const uint32_t row_page = tile_row * Wt + slice_w0;
                // Batches of read_batch_chunks chunks per barrier. A row occupies a contiguous Wp-tile
                // window of the ring (capacity % Wp == 0), so a batch window never wraps.
                for (uint32_t c0 = 0; c0 < num_w_chunks; c0 += read_batch_chunks) {
                    const uint32_t batch_chunks =
                        (num_w_chunks - c0) < read_batch_chunks ? (num_w_chunks - c0) : read_batch_chunks;
                    const uint32_t batch_tiles = batch_chunks * C;
                    const uint32_t first = c0 * C;
                    // Valid slice tiles in this batch (0 for a batch made only of pad tiles).
                    const uint32_t left = slice_valid > first ? slice_valid - first : 0;
                    const uint32_t valid = left < batch_tiles ? left : batch_tiles;
                    {
                        MaybeDeviceZoneScope("rd_reserve");
                        x_cb.reserve_back(batch_tiles);
                    }
                    const uint32_t l1 = x_cb.get_write_ptr();
                    {
                        MaybeDeviceZoneScope("rd_issue");
                        uint32_t j = stagger_first(bank_stagger, valid);
                        for (uint32_t n = 0; n < valid; ++n) {
                            if constexpr (!RMS_ABLATE_READS) {
                                noc_async_read_tile(row_page + first + j, input_accessor, l1 + j * T_in);
                            }
                            j = stagger_next(j, valid);
                        }
                    }
                    {
                        MaybeDeviceZoneScope("rd_barrier");
                        noc_async_read_barrier();
                    }
                    if constexpr (w_valid_in_last_tile < TILE_DIM) {
                        if (holds_last_tile && valid > 0 && first + valid == slice_valid) {
                            zero_tile_w_padding<es>(noc, x_cb, (valid - 1) * T_in, w_valid_in_last_tile);
                        }
                    }
                    x_cb.push_back(batch_tiles);
                }
            } else if constexpr (rm_sharded) {
                // ROW_MAJOR shard: the core's own sticks (local L1, row stride shard_row_bytes) are re-laid
                // into the tilize staging at stride C*32*es, chunk by chunk; the slice's pad bytes (a slice
                // need not be tile-aligned) are zeroed so they add nothing to Σx². No TensorAccessor, no DRAM.
                const uint32_t first_stick = tile_row * TILE_DIM;  // local to the shard
                const uint32_t left = stick_count - first_stick;
                const uint32_t sticks = left < TILE_DIM ? left : TILE_DIM;
                const uint32_t slice_bytes = slice_elems * es;
                CircularBuffer rm_cb(cb_x_rm_sticks);
                for (uint32_t c = 0; c < num_w_chunks; ++c) {
                    const uint32_t first_byte = c * chunk_stick_bytes;
                    const uint32_t rest = slice_bytes > first_byte ? slice_bytes - first_byte : 0;
                    const uint32_t valid_bytes = rest < chunk_stick_bytes ? rest : chunk_stick_bytes;
                    rm_cb.reserve_back(C);
                    const uint32_t l1 = rm_cb.get_write_ptr();
                    const uint32_t src = input_addr + first_stick * shard_row_bytes + first_byte;
                    if (valid_bytes > 0 && !RMS_ABLATE_READS) {
                        for (uint32_t s = 0; s < sticks; ++s) {
                            noc_async_read(
                                get_noc_addr(src + s * shard_row_bytes), l1 + s * chunk_stick_bytes, valid_bytes);
                        }
                    }
                    if (valid_bytes < chunk_stick_bytes) {
                        for (uint32_t s = 0; s < sticks; ++s) {
                            zero_stick_tail<es, chunk_stick_bytes>(noc, rm_cb, s, valid_bytes);
                        }
                        noc.write_zeros_l1_barrier();
                    }
                    noc_async_read_barrier();
                    rm_cb.push_back(C);
                }
            } else {
                const uint32_t first_stick = tile_row * TILE_DIM;
                const uint32_t left = num_sticks - first_stick;
                const uint32_t sticks = left < TILE_DIM ? left : TILE_DIM;
                constexpr uint32_t full_chunks = tail_chunk_is_full ? num_w_chunks : num_w_chunks - 1;
                for (uint32_t c = 0; c < full_chunks; ++c) {
                    dataflow_kernel_lib::read_sticks_for_tilize<cb_x_rm_sticks>(
                        input_accessor, sticks, chunk_stick_bytes, first_stick, c * chunk_stick_bytes);
                }
                if constexpr (!tail_chunk_is_full) {
                    // Tail chunk: stride C*32*es, C pages, pad bytes [tail_stick_bytes, chunk_stick_bytes) zeroed.
                    constexpr uint32_t tail_offset = (num_w_chunks - 1) * chunk_stick_bytes;
                    constexpr uint32_t pad_head_end = (tail_stick_bytes + L1_ALIGN - 1) / L1_ALIGN * L1_ALIGN;
                    constexpr uint32_t pad_head_elems = (pad_head_end - tail_stick_bytes) / es;
                    CircularBuffer rm_cb(cb_x_rm_sticks);
                    rm_cb.reserve_back(C);
                    const uint32_t l1 = rm_cb.get_write_ptr();
                    for (uint32_t s = 0; s < sticks; ++s) {
                        noc_async_read(
                            input_accessor.get_noc_addr(first_stick + s, tail_offset),
                            l1 + s * chunk_stick_bytes,
                            tail_stick_bytes);
                    }
                    for (uint32_t s = 0; s < sticks; ++s) {
                        if constexpr (pad_head_elems > 0) {
                            store_zeros<es>(l1 + s * chunk_stick_bytes + tail_stick_bytes, pad_head_elems);
                        }
                        if constexpr (pad_head_end < chunk_stick_bytes) {
                            noc.async_write_zeros(
                                rm_cb,
                                chunk_stick_bytes - pad_head_end,
                                {.offset_bytes = s * chunk_stick_bytes + pad_head_end});
                        }
                    }
                    noc.write_zeros_l1_barrier();
                    noc_async_read_barrier();
                    rm_cb.push_back(C);
                }
            }
        }
    };

    // ---- main schedule ----
    const uint32_t num_blocks = (rows_this_core + block_rows - 1) / block_rows;
    auto for_each_block = [&](auto&& combine_step) {
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            const uint32_t block_first_row = block_idx * block_rows;
            const uint32_t remaining = rows_this_core - block_first_row;
            const uint32_t rows_in_block = remaining < block_rows ? remaining : block_rows;  // R_b
            if constexpr (!sharded) {
                MaybeDeviceZoneScope("rd_load");
                load_block(block_first_row, rows_in_block);
            }
            // The combine of block k completes before block k+1 is loaded: the load then overlaps
            // normalize(k) (x_depth = 2), and the partial goes out as soon as compute produces it.
            combine_step(block_idx, rows_in_block);
        }
    };
    run_blocks<
        combine,
        group_size,
        combine_owners,
        block_rows,
        T_interm,
        gather_sem_id,
        inv_sem_id,
        wave_rows,
        wave_q,
        waves_full,
        ag_writer>(mcast, noc, is_root, my_index, root_x, root_y, member_arg_base, for_each_block);
}
