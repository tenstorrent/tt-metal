// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post reader (NCRISC / NoC0) — plain-2d and stats-2d-tree regimes.
//
// Per core: rows [row_start, row_start + num_rows) × c-tiles [c_start, c_start + num_c).
//   load_coefficients(row): read the post / comb source tiles (one page each) into the
//       reader-private scratch cb_coef_src, then expand every coefficient q into column 0
//       of its own tile in cb_coef_cols (order: post_0..n-1, comb_{i,j} at n + i*n + j).
//       The coefficient page(s) are barriered (and multicast) alone; the row's first F/X block is
//       then issued and the expansion runs while its reads are in flight (Perf 1).
//   load_block(row, b): block_c F tiles + n*block_c X tiles (layout [i][c]), reads issued
//       back-to-back, one barrier per block, nominal block_c counts pushed (ragged tail
//       reads only its valid tiles).
//   fused (stats): load the resident W slice once (block-major [b][j][c], compact ragged tail);
//       as a combine consumer (leader / root) GRANT each row's gather slot set at the row start
//       (after reserving it locally) and COLLECT it (arrival semaphore -> push to compute)
//       after the mix of row + LAG (root: + 2*LAG). See mhc_post_combine.hpp.
//   reuse multicast (kernel_lib McastArgs, host ttnn.Mcast; any channel may be ABSENT or roleless here):
//     w_mc: the column group's injector reads each first-row block's W slice and multicasts it to the
//       group (no handshake, Counter signal: the resident slice is written once); receivers skip the
//       W reads and wait round b.
//     coef_mc / coef_tail_mc: the row segment's sender reads the coefficient source page(s) and
//       multicasts them to the segment's cb_coef_src (handshake: the scratch is reused per row);
//       receivers skip the reads. A core with no role on any channel reads for itself.

#include <stdint.h>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/ttnn/operations/mhc_post/kernels/mhc_post_combine.hpp"

// dram_straggler variant switches: v8/v9 split the W read over the column group; v9/v10 keep one
// block in flight (v0's depth) with the trid F/X-vs-W split.
#define DS_SPLIT_W (DS_VARIANT == 8 || DS_VARIANT == 9)
#define DS_AHEAD (DS_VARIANT != 9 && DS_VARIANT != 10)

namespace {

constexpr uint32_t cb_coef_src = get_compile_time_arg_val(0);
constexpr uint32_t cb_coef_cols = get_compile_time_arg_val(1);
constexpr uint32_t cb_f_tiles = get_compile_time_arg_val(2);
constexpr uint32_t cb_x_tiles = get_compile_time_arg_val(3);
constexpr uint32_t n = get_compile_time_arg_val(4);
constexpr uint32_t block_c = get_compile_time_arg_val(5);
constexpr uint32_t tensor_c_tiles = get_compile_time_arg_val(6);
constexpr uint32_t n_coef_cols = get_compile_time_arg_val(7);
constexpr uint32_t post_width_tiles = get_compile_time_arg_val(8);
constexpr uint32_t comb_width_tiles = get_compile_time_arg_val(9);
constexpr uint32_t n_src = get_compile_time_arg_val(10);      // distinct coefficient source pages per row
constexpr uint32_t comb_slot = get_compile_time_arg_val(11);  // cb_coef_src slot of the comb window
constexpr uint32_t pre_slot = get_compile_time_arg_val(12);   // cb_coef_src slot of the P window (collapse)
constexpr uint32_t pre_width_tiles = get_compile_time_arg_val(13);
constexpr uint32_t FB = 14;  // fused block (descriptor fused_ct)
constexpr bool fused = get_compile_time_arg_val(FB + 0) != 0;
constexpr uint32_t cb_w_slice = get_compile_time_arg_val(FB + 2);
constexpr uint32_t cb_gather = get_compile_time_arg_val(FB + 8);
constexpr uint32_t cb_root_gather = get_compile_time_arg_val(FB + 9);
constexpr uint32_t Q = get_compile_time_arg_val(FB + 13);
constexpr uint32_t COMBINE_FANIN = get_compile_time_arg_val(FB + 14);
constexpr uint32_t COMBINE_DEPTH = get_compile_time_arg_val(FB + 15);
constexpr uint32_t COMBINE_LAG = get_compile_time_arg_val(FB + 16);
constexpr uint32_t cb_comb_flags = get_compile_time_arg_val(FB + 11);  // per-slot arrival counters
constexpr uint32_t SEM_LEAD_CREDIT = get_compile_time_arg_val(FB + 19);
constexpr uint32_t SEM_ROOT_CREDIT = get_compile_time_arg_val(FB + 20);
static_assert(!fused || COMBINE_LAG + 1 <= COMBINE_DEPTH, "grants run LAG rows ahead of collects");

constexpr uint32_t proj_passes = get_compile_time_arg_val(FB + 22);
constexpr uint32_t cb_proj_masks = get_compile_time_arg_val(FB + 23);
constexpr bool collapse = get_compile_time_arg_val(FB + 25) != 0;
constexpr uint32_t FUSED_CT_LEN = 30;  // len(fused_ct) in the descriptor
constexpr auto f_args = TensorAccessorArgs<FB + FUSED_CT_LEN>();
constexpr auto x_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
constexpr auto post_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();
constexpr auto pre_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto w_mc = dataflow_kernel_lib::
    McastArgs<get_named_compile_time_arg_val("w_mc_ct_offset"), get_named_compile_time_arg_val("w_mc_rt_offset")>();
constexpr auto coef_mc = dataflow_kernel_lib::McastArgs<
    get_named_compile_time_arg_val("coef_mc_ct_offset"),
    get_named_compile_time_arg_val("coef_mc_rt_offset")>();
constexpr auto coef_tail_mc = dataflow_kernel_lib::McastArgs<
    get_named_compile_time_arg_val("coef_tail_mc_ct_offset"),
    get_named_compile_time_arg_val("coef_tail_mc_rt_offset")>();
constexpr uint32_t PRE_COEF_BASE = n + n * n;  // P_i at cb_coef_cols tile n + n^2 + i
static_assert(!collapse || n_src >= 1, "collapse reads a P window");

constexpr uint32_t FACE_WORDS = 256;  // 16x16 fp32 face
constexpr uint32_t FACE_DIM = 16;
static_assert(proj_passes == 1 || proj_passes == FACE_DIM, "exact projection: one masked column pair per face lane");

// Word offset of element (row t, col c) inside a 32x32 fp32 tile (faces 0..3 row-major).
FORCE_INLINE uint32_t tile_word(uint32_t t, uint32_t c) {
    return ((t / FACE_DIM) * 2 + (c / FACE_DIM)) * FACE_WORDS + (t % FACE_DIM) * FACE_DIM + (c % FACE_DIM);
}

// Copy `count` coefficients from source column `src_col` onward into column 0 of consecutive
// destination tiles starting at `dst_base` (32 tokens each). Only column 0 is meaningful.
// Perf 1: plain __restrict__ pointers, 8 loads issued ahead of their 8 stores (4 batches of 8
// tokens: face rows 0..7 / 8..15 of faces 0 and 2), bracketed by compiler barriers — the CB
// reserve / push handshakes order the data. The previous volatile per-word copy serialized every
// load before its store: 13.1 us per row at the focus shape, on the critical path (compute cannot
// mix before the coefficients land); this form takes ~1-2 us. Only the batch is unrolled (the op
// sits near the kernel-config-buffer text limit in profiler builds).
FORCE_INLINE void expand_coefficients(uint32_t src_addr, uint32_t src_col, uint32_t dst_base, uint32_t count) {
    asm volatile("" ::: "memory");
    const uint32_t* __restrict__ src = reinterpret_cast<const uint32_t*>(src_addr);
    uint32_t* __restrict__ dst = reinterpret_cast<uint32_t*>(dst_base);
#pragma GCC unroll 1
    for (uint32_t q = 0; q < count; ++q) {
        const uint32_t c = src_col + q;
        const uint32_t* __restrict__ s = src + (c / FACE_DIM) * FACE_WORDS + (c % FACE_DIM);
        uint32_t* __restrict__ d = dst + q * 1024;
#pragma GCC unroll 1
        for (uint32_t b = 0; b < 4; ++b) {
            // batch b: face (b / 2) * 2 (0 or 2), face rows (b % 2) * 8 .. +7
            const uint32_t off = (b / 2) * 2 * FACE_WORDS + (b % 2) * 8 * FACE_DIM;
            const uint32_t* __restrict__ sb = s + off;
            uint32_t* __restrict__ db = d + off;
            uint32_t v[8];
#pragma GCC unroll 8
            for (uint32_t k = 0; k < 8; ++k) {
                v[k] = sb[k * FACE_DIM];
            }
#pragma GCC unroll 8
            for (uint32_t k = 0; k < 8; ++k) {
                db[k * FACE_DIM] = v[k];
            }
        }
    }
    asm volatile("" ::: "memory");
}

}  // namespace

void kernel_main() {
    const uint32_t f_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_arg_val<uint32_t>(2);
    const uint32_t comb_addr = get_arg_val<uint32_t>(3);
    const uint32_t row_start = get_arg_val<uint32_t>(4);
    const uint32_t num_rows = get_arg_val<uint32_t>(5);
    const uint32_t c_start = get_arg_val<uint32_t>(6);
    const uint32_t num_c = get_arg_val<uint32_t>(7);
    const uint32_t post_col = get_arg_val<uint32_t>(8);
    const uint32_t comb_col = get_arg_val<uint32_t>(9);
    // arg 10: g_r (unused here)
    const uint32_t g_c = get_arg_val<uint32_t>(11);
    const uint32_t w_addr = get_arg_val<uint32_t>(12);
    const uint32_t pre_addr = get_arg_val<uint32_t>(13);

    const uint32_t f_tile_bytes = get_tile_size(cb_f_tiles);
    const uint32_t x_tile_bytes = get_tile_size(cb_x_tiles);
    const uint32_t coef_tile_bytes = get_tile_size(cb_coef_src);

    const auto f_acc = TensorAccessor(f_args, f_addr, f_tile_bytes);
    const auto x_acc = TensorAccessor(x_args, x_addr, x_tile_bytes);
    const auto post_acc = TensorAccessor(post_args, post_addr, coef_tile_bytes);
    const auto comb_acc = TensorAccessor(comb_args, comb_addr, coef_tile_bytes);
    const auto pre_acc = TensorAccessor(pre_args, pre_addr, coef_tile_bytes);

    // Reader-private scratch: slot 0 = post source tile, comb / P windows at their (host-deduplicated)
    // slots — one packed mixes page holding every window is a single slot. Never pushed.
    const uint32_t src_post_l1 = get_write_ptr(cb_coef_src);
    const uint32_t src_comb_l1 = src_post_l1 + comb_slot * coef_tile_bytes;
    const uint32_t src_pre_l1 = src_post_l1 + pre_slot * coef_tile_bytes;

    const uint32_t num_blocks = (num_c + block_c - 1) / block_c;

    // --- reuse-multicast roles (runtime; a core with none reads for itself) ---
    Noc noc;
    auto w_tx = w_mc.optional_sender(noc);
    auto w_rx = w_mc.optional_receiver(noc);
    auto coef_tx = coef_mc.optional_sender(noc);
    auto coef_rx = coef_mc.optional_receiver(noc);
    auto coef_tail_tx = coef_tail_mc.optional_sender(noc);
    auto coef_tail_rx = coef_tail_mc.optional_receiver(noc);
    const bool w_reads = !w_rx.has_value();
    const bool coef_reads = !coef_rx.has_value() && !coef_tail_rx.has_value();
    const uint32_t coef_bytes = n_src * coef_tile_bytes;
    // Hand the row's coefficient page(s) across the segment (after the sender's read barrier).
    auto share_coefficients = [&]() {
        if (coef_tx.has_value()) {
            coef_tx->send(src_post_l1, src_post_l1, coef_bytes);
        } else if (coef_rx.has_value()) {
            coef_rx->receive();
        }
        if (coef_tail_tx.has_value()) {
            coef_tail_tx->send(src_post_l1, src_post_l1, coef_bytes);
        } else if (coef_tail_rx.has_value()) {
            coef_tail_rx->receive();
        }
    };

    // --- fused: resident W slice, block-major [b][j][c] (compact K = n * extent per block) ---
    const auto geo = mhc_post_combine::Geometry::of(g_c, Q, COMBINE_FANIN);
    uint32_t lead_granted = 0, lead_pushed = 0, root_granted = 0, root_pushed = 0;
    // The slice is reserved once here and filled per block on the first row (issue_block), so the
    // W stream overlaps the first row's mix instead of serializing ahead of it; compute waits the
    // cumulative count before each first-row projection.
    const uint32_t w_tile_bytes = fused ? get_tile_size(cb_w_slice) : x_tile_bytes;
    const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);
    uint32_t w_l1 = 0;
    if constexpr (fused) {
        cb_reserve_back(cb_w_slice, n * num_c);
        w_l1 = get_write_ptr(cb_w_slice);
        if constexpr (proj_passes > 1) {
            // Exact-projection masks: tile j = 1.0 (bf16) at columns j and j + 16, 0 elsewhere.
            // Zeroed by the NoC; only the 64 pattern lanes per tile are written by the RISC.
            const uint32_t mask_bytes = get_tile_size(cb_proj_masks);
            cb_reserve_back(cb_proj_masks, proj_passes);
            Noc noc;
            CircularBuffer masks(cb_proj_masks);
            noc.async_write_zeros(masks, proj_passes * mask_bytes);
            noc.write_zeros_l1_barrier();
            volatile tt_l1_ptr uint16_t* m =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_proj_masks));
            constexpr uint16_t BF16_ONE = 0x3F80;
            for (uint32_t j = 0; j < proj_passes; ++j) {
                volatile tt_l1_ptr uint16_t* t = m + j * (mask_bytes / 2);
                for (uint32_t face = 0; face < 4; ++face) {
                    for (uint32_t row = 0; row < FACE_DIM; ++row) {
                        t[face * FACE_WORDS + row * FACE_DIM + j] = BF16_ONE;
                    }
                }
            }
            cb_push_back(cb_proj_masks, proj_passes);
        }
    }
    // Zero my arrival counters before my first grant (no sender touches them before it).
    uint32_t flags_base = 0;
    if constexpr (fused) {
        flags_base = get_write_ptr(cb_comb_flags);
        volatile tt_l1_ptr uint32_t* fw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags_base);
        for (uint32_t i = 0; i < 4 * COMBINE_DEPTH * (geo.slots + geo.groups); ++i) {
            fw[i] = 0;
        }
    }
    constexpr uint32_t RT_MEMBERS = 14;  // (x, y) of my group's members, then of every group leader
    const uint32_t rt_leaders = RT_MEMBERS + 2 * geo.size;

    // GRANT row r's slot sets: reserve them locally (compute has popped their previous
    // occupants), then credit every sender.
    auto grant = [&](uint32_t r) {
        if (geo.leader) {
            cb_reserve_back(cb_gather, geo.slots * (lead_granted - lead_pushed + 1));
            ++lead_granted;
            const uint32_t sem = get_semaphore(SEM_LEAD_CREDIT);
            for (uint32_t m = 0; m < geo.size; ++m) {
                const uint32_t px = get_arg_val<uint32_t>(RT_MEMBERS + 2 * m);
                const uint32_t py = get_arg_val<uint32_t>(RT_MEMBERS + 2 * m + 1);
                noc_semaphore_inc(get_noc_addr(px, py, sem), 1);
            }
        }
        if (geo.roots(r)) {
            cb_reserve_back(cb_root_gather, geo.groups * (root_granted - root_pushed + 1));
            ++root_granted;
            const uint32_t sem = get_semaphore(SEM_ROOT_CREDIT + geo.group);
            for (uint32_t g = 0; g < geo.groups; ++g) {
                const uint32_t px = get_arg_val<uint32_t>(rt_leaders + 2 * g);
                const uint32_t py = get_arg_val<uint32_t>(rt_leaders + 2 * g + 1);
                noc_semaphore_inc(get_noc_addr(px, py, sem), 1);
            }
        }
    };
    // COLLECT: hand the row's slots to compute one at a time, in slot order, as each sender's
    // per-slot arrival counter (cb_comb_flags) ticks, so compute's adds overlap the arrivals.
    auto flag_ptr = [&](uint32_t off) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags_base + off); };
    auto collect_post_row = [&](uint32_t r) {
        if (geo.leader && r >= COMBINE_LAG && r - COMBINE_LAG < num_rows) {
            const uint32_t rl = r - COMBINE_LAG;
            const uint32_t set = rl % COMBINE_DEPTH;
            for (uint32_t s = 0; s < geo.size; ++s) {
                noc_semaphore_wait_min(flag_ptr(geo.gather_flag(set, s)), rl / COMBINE_DEPTH + 1);
                cb_push_back(cb_gather, 1);
            }
            if (geo.slots > geo.size) {
                cb_push_back(cb_gather, geo.slots - geo.size);  // padding slots of a short group
            }
            ++lead_pushed;
        }
        if (r >= 2 * COMBINE_LAG && r - 2 * COMBINE_LAG < num_rows && geo.roots(r - 2 * COMBINE_LAG)) {
            const uint32_t k = geo.root_ordinal(r - 2 * COMBINE_LAG);
            const uint32_t set = k % COMBINE_DEPTH;
            for (uint32_t g = 0; g < geo.groups; ++g) {
                noc_semaphore_wait_min(flag_ptr(geo.root_flag(COMBINE_DEPTH, set, g)), k / COMBINE_DEPTH + 1);
                cb_push_back(cb_root_gather, 1);
            }
            ++root_pushed;
        }
    };

    // Issue the reads of one F/X block (nominal reserve; only valid tiles read) and, on the first
    // row, the block's compact W slice [j][c]. No barrier.
    auto issue_block = [&](uint32_t row, uint32_t b, bool first_row) {
        const uint32_t c0 = c_start + b * block_c;
        const uint32_t extent = (num_c - b * block_c) < block_c ? (num_c - b * block_c) : block_c;
        {
            MaybeDeviceZoneScope("reader_in_reserve");  // back-pressure from compute
            cb_reserve_back(cb_f_tiles, block_c);
            cb_reserve_back(cb_x_tiles, n * block_c);
        }
        MaybeDeviceZoneScope("reader_issue");
        uint32_t f_l1 = get_write_ptr(cb_f_tiles);
        const uint32_t x_l1 = get_write_ptr(cb_x_tiles);
        const uint32_t f_page0 = row * tensor_c_tiles + c0;
        for (uint32_t c = 0; c < extent; ++c) {
            noc_async_read_tile(f_page0 + c, f_acc, f_l1);
            f_l1 += f_tile_bytes;
        }
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t x_page0 = (row * n + i) * tensor_c_tiles + c0;
            uint32_t x_dst = x_l1 + i * block_c * x_tile_bytes;
            for (uint32_t c = 0; c < extent; ++c) {
                noc_async_read_tile(x_page0 + c, x_acc, x_dst);
                x_dst += x_tile_bytes;
            }
        }
        if constexpr (fused) {
            if (first_row && w_reads) {
                for (uint32_t j = 0; j < n; ++j) {
                    for (uint32_t c = 0; c < extent; ++c) {
                        noc_async_read_tile(j * tensor_c_tiles + c0 + c, w_acc, w_l1);
                        w_l1 += w_tile_bytes;
                    }
                }
            }
        }
    };
    uint32_t w_sent_l1 = fused ? w_l1 : 0;  // start of the next block's W slice (multicast source/landing)
    auto push_block = [&](uint32_t b, bool first_row) {
        cb_push_back(cb_f_tiles, block_c);
        cb_push_back(cb_x_tiles, n * block_c);
        if constexpr (fused) {
            if (first_row) {
                const uint32_t extent = (num_c - b * block_c) < block_c ? (num_c - b * block_c) : block_c;
                const uint32_t w_block_bytes = n * extent * w_tile_bytes;
                MaybeDeviceZoneScope("reader_w_mcast");
                if (w_tx.has_value()) {
                    w_tx->send(w_sent_l1, w_sent_l1, w_block_bytes);
                } else if (w_rx.has_value()) {
                    w_rx->receive(b);
                }
                w_sent_l1 += w_block_bytes;
                cb_push_back(cb_w_slice, n * extent);
            }
        }
    };

#if DS_VARIANT == 0
    const uint32_t total = fused ? num_rows + 2 * COMBINE_LAG : num_rows;
    for (uint32_t r = 0; r < total; ++r) {
        if (r >= num_rows) {
            if constexpr (fused) {
                MaybeDeviceZoneScope("reader_collect");
                collect_post_row(r);  // drain the trailing combines
            }
            continue;
        }
        const uint32_t row = row_start + r;

        // load_coefficients: source reads (sender / roleless core) and their own barrier, the segment
        // multicast, then the row's first block is issued and the coefficients are expanded while
        // its reads are in flight (Perf 1: coefficient-first ordering).
        if (coef_reads) {
            MaybeDeviceZoneScope("reader_coef_issue");
            noc_async_read_tile(row * post_width_tiles + post_col / 32, post_acc, src_post_l1);
            if constexpr (comb_slot != 0) {
                noc_async_read_tile(row * comb_width_tiles + comb_col / 32, comb_acc, src_comb_l1);
            }
            if constexpr (collapse && pre_slot != 0 && pre_slot != comb_slot) {
                noc_async_read_tile(row * pre_width_tiles, pre_acc, src_pre_l1);  // P at columns 0..n-1
            }
            noc_async_read_barrier();  // coefficient page(s) only
        }
        {
            MaybeDeviceZoneScope("reader_coef_share");
            share_coefficients();  // sender: after its read barrier; receiver: waits the multicast
        }
        issue_block(row, 0, r == 0);
        if constexpr (fused) {
            grant(r);
        }
        {
            cb_reserve_back(cb_coef_cols, n_coef_cols);
        }
        {
            MaybeDeviceZoneScope("reader_coef_expand");
            const uint32_t coef_l1 = get_write_ptr(cb_coef_cols);
            expand_coefficients(src_post_l1, post_col % 32, coef_l1, n);
            expand_coefficients(src_comb_l1, comb_col % 32, coef_l1 + n * coef_tile_bytes, n * n);
            if constexpr (collapse) {
                expand_coefficients(src_pre_l1, 0, coef_l1 + PRE_COEF_BASE * coef_tile_bytes, n);
            }
            cb_push_back(cb_coef_cols, n_coef_cols);
        }
        {
            MaybeDeviceZoneScope("reader_barrier");
            noc_async_read_barrier();  // block 0
        }
        push_block(0, r == 0);

        // load_block for the rest of the row.
        for (uint32_t b = 1; b < num_blocks; ++b) {
            issue_block(row, b, r == 0);
            {
                MaybeDeviceZoneScope("reader_barrier");
                noc_async_read_barrier();
            }
            push_block(b, r == 0);
        }
        if constexpr (fused) {
            MaybeDeviceZoneScope("reader_collect");
            collect_post_row(r);
        }
    }
#else
    // ---- dram_straggler v1+: transaction-id read pipeline ----
    // F/X of flattened block k carry trid 1 + (k & 1), its W slice 6 + (b & 7), coefficient pages
    // TRID_COEF. Block k + 1 (next row's block 0 included) is issued before block k is barriered,
    // so IO_DEPTH = 2 blocks stay on the wire; F/X are pushed as soon as THEIR reads land (the W
    // slice no longer gates them). Landing slots are computed (the CB write pointer lags the
    // in-flight block): the CBs hold exactly 2 nominal blocks and every push is nominal, so block k
    // lands at base + (k % 2) * block bytes.
    constexpr uint32_t TRID_COEF = 5;
    const uint32_t f_base = get_write_ptr(cb_f_tiles);
    const uint32_t x_base = get_write_ptr(cb_x_tiles);
    const uint32_t f_blk_bytes = block_c * f_tile_bytes;
    const uint32_t x_blk_bytes = n * block_c * x_tile_bytes;
    const uint32_t K = num_rows * num_blocks;
    uint32_t k_issued = 0, k_pushed = 0;
    auto issue_next = [&]() {
        if (k_issued >= K) {
            return;
        }
        const uint32_t k = k_issued;
        const uint32_t row = row_start + k / num_blocks;
        const uint32_t b = k % num_blocks;
        const bool first_row = k < num_blocks;
        const uint32_t c0 = c_start + b * block_c;
        const uint32_t extent = (num_c - b * block_c) < block_c ? (num_c - b * block_c) : block_c;
        const uint32_t ahead = k_issued - k_pushed + 1;
        {
            MaybeDeviceZoneScope("reader_in_reserve");
            cb_reserve_back(cb_f_tiles, ahead * block_c);
            cb_reserve_back(cb_x_tiles, ahead * n * block_c);
        }
        MaybeDeviceZoneScope("reader_issue");
        uint32_t f_l1 = f_base + (k & 1) * f_blk_bytes;
        const uint32_t x_l1 = x_base + (k & 1) * x_blk_bytes;
        noc_async_read_set_trid(1 + (k & 1));
        const uint32_t f_page0 = row * tensor_c_tiles + c0;
        for (uint32_t c = 0; c < extent; ++c) {
            noc_async_read_tile(f_page0 + c, f_acc, f_l1);
            f_l1 += f_tile_bytes;
        }
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t x_page0 = (row * n + i) * tensor_c_tiles + c0;
            uint32_t x_dst = x_l1 + i * block_c * x_tile_bytes;
            for (uint32_t c = 0; c < extent; ++c) {
                noc_async_read_tile(x_page0 + c, x_acc, x_dst);
                x_dst += x_tile_bytes;
            }
        }
        if constexpr (fused) {
            if (first_row && w_reads) {
                noc_async_read_set_trid(3 + (k & 1));
                for (uint32_t j = 0; j < n; ++j) {
                    for (uint32_t c = 0; c < extent; ++c) {
                        noc_async_read_tile(j * tensor_c_tiles + c0 + c, w_acc, w_l1);
                        w_l1 += w_tile_bytes;
                    }
                }
            }
        }
        ++k_issued;
    };
    auto retire = [&](uint32_t k) {
        const uint32_t b = k % num_blocks;
        const bool first_row = k < num_blocks;
        {
            MaybeDeviceZoneScope("reader_barrier");
            noc_async_read_barrier_with_trid(1 + (k & 1));
        }
        cb_push_back(cb_f_tiles, block_c);
        cb_push_back(cb_x_tiles, n * block_c);
        ++k_pushed;
        if constexpr (fused) {
            if (first_row) {
                const uint32_t extent = (num_c - b * block_c) < block_c ? (num_c - b * block_c) : block_c;
                const uint32_t w_block_bytes = n * extent * w_tile_bytes;
                MaybeDeviceZoneScope("reader_w_mcast");
                if (w_reads) {
                    noc_async_read_barrier_with_trid(3 + (k & 1));
                }
                if (w_tx.has_value()) {
                    w_tx->send(w_sent_l1, w_sent_l1, w_block_bytes);
                } else if (w_rx.has_value()) {
                    w_rx->receive(b);
                }
                w_sent_l1 += w_block_bytes;
                cb_push_back(cb_w_slice, n * extent);
            }
        }
    };

    const uint32_t total = fused ? num_rows + 2 * COMBINE_LAG : num_rows;
    for (uint32_t r = 0; r < total; ++r) {
        if (r >= num_rows) {
            if constexpr (fused) {
                MaybeDeviceZoneScope("reader_collect");
                collect_post_row(r);
            }
            continue;
        }
        const uint32_t row = row_start + r;
        if (coef_reads) {
            MaybeDeviceZoneScope("reader_coef_issue");
            noc_async_read_set_trid(TRID_COEF);
            noc_async_read_tile(row * post_width_tiles + post_col / 32, post_acc, src_post_l1);
            if constexpr (comb_slot != 0) {
                noc_async_read_tile(row * comb_width_tiles + comb_col / 32, comb_acc, src_comb_l1);
            }
            if constexpr (collapse && pre_slot != 0 && pre_slot != comb_slot) {
                noc_async_read_tile(row * pre_width_tiles, pre_acc, src_pre_l1);
            }
            noc_async_read_barrier_with_trid(TRID_COEF);
        }
        {
            MaybeDeviceZoneScope("reader_coef_share");
            share_coefficients();
        }
        if (r == 0) {
            issue_next();
#if DS_VARIANT == 1
            issue_next();  // v1: second block before the expansion (its issue can stall the RISC)
#endif
        }
        if constexpr (fused) {
            grant(r);
        }
        cb_reserve_back(cb_coef_cols, n_coef_cols);
        {
            MaybeDeviceZoneScope("reader_coef_expand");
            const uint32_t coef_l1 = get_write_ptr(cb_coef_cols);
            expand_coefficients(src_post_l1, post_col % 32, coef_l1, n);
            expand_coefficients(src_comb_l1, comb_col % 32, coef_l1 + n * coef_tile_bytes, n * n);
            if constexpr (collapse) {
                expand_coefficients(src_pre_l1, 0, coef_l1 + PRE_COEF_BASE * coef_tile_bytes, n);
            }
            cb_push_back(cb_coef_cols, n_coef_cols);
        }
#if DS_VARIANT >= 2 && DS_AHEAD
        if (r == 0) {
            issue_next();  // v2: second block after the expansion
        }
#endif
        for (uint32_t b = 0; b < num_blocks; ++b) {
            retire(r * num_blocks + b);
            if (b + 1 == num_blocks) {
                if constexpr (fused) {
                    MaybeDeviceZoneScope("reader_collect");
                    collect_post_row(r);
                }
            }
            issue_next();
        }
    }
    noc_async_read_set_trid(0);
#endif
    if constexpr (fused) {
        noc_async_atomic_barrier();
    }
}
