// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm — worker reader (RISCV_1, NoC0).
//   load_block: the worker's resident x block (h_act x w_act tiles, one barrier per tile row, one push of the block),
//               gamma slivers (face0/face1 row 0 of each tile: 2 x 64 B), the SUM scaler.
//   leader only — combine input: zero cb_gather, wait until both ports of the link report a complete board, read
//               every source's 64 B half-row runs into the gather tiles (row s = g * num_w_chunks + wc).
//   every worker — inv_rms: the leader's writer drops the fp32 row vectors into cb_inv_row's bytes; this kernel is
//               the CB's single producer and pushes them once sem_inv_ready says they have landed.
//   sharded x  — x is the core's own shard, backing cb_x_block (zero-copy): a round's "load" is a push of its pages,
//               never a NoC read. A padded last WIDTH shard has its pad columns zero-filled once (sum-of-squares safe).
//   pre-combine (leader) — the row-group's num_w_chunks partials land in cb_pregather (writers' NoC writes); this
//               kernel zero-fills the unused rows and pushes them once sem_pre_ready counts every chunk.
//   gamma mcast (R4) — one worker per W-chunk column reads its chunk's gamma slivers into cb_gamma and multicasts
//               the cb_gamma pages (mcast_pipe) to the column's other workers (same CB address everywhere).
//   streamed gamma — the gamma slivers are re-read per tile row of each round in GAMMA_BATCH batches (wide shards
//               where a resident gamma + stall shadow does not fit).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

constexpr uint32_t cb_x_block = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_gather = 7;
constexpr uint32_t cb_inv_row = 8;
constexpr uint32_t sem_board_ready_id = 1;
constexpr uint32_t sem_inv_ready_id = 2;
constexpr uint32_t sem_board_ready_odd_id = 4;
constexpr uint32_t cb_pregather = 12;
constexpr uint32_t sem_pre_ready_id = 5;
constexpr uint32_t sem_pre_ready_odd_id = 6;
constexpr uint32_t sem_gather_done_id = 7;  // on both ports of the link: this leader has read the whole board
constexpr uint32_t ports_per_link = 2;
constexpr uint32_t fp32_tile_bytes = 4096;
constexpr uint32_t face_bytes_fp32 = 1024;
constexpr uint32_t half_row_bytes = 64;
constexpr uint32_t sliver_bytes = 64;

// gamma mcast faces: templates on the McastArgs type, so an absent channel (gamma mcast off) discards the bodies
template <typename MC>
FORCE_INLINE void gamma_mcast_send(const MC& mc, uint32_t base, uint32_t bytes) {
    if constexpr (MC::active) {
        Noc noc;
        auto pipe = mc.sender(noc);
        pipe.send(base, base, bytes);
    }
}
template <typename MC>
FORCE_INLINE void gamma_mcast_receive(const MC& mc) {
    if constexpr (MC::active) {
        Noc noc;
        auto pipe = mc.receiver(noc);
        pipe.receive();
        // the call's single receive: re-arm the counter for the next call
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MC::data_ready)), 0);
    }
}

void kernel_main() {
    constexpr uint32_t block_h = get_compile_time_arg_val(0);
    constexpr uint32_t block_w = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(2);
    constexpr bool has_gamma = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t x_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t g_page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t g_face_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t n_gather_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(8);  // partial vectors per tile row on the board
    constexpr uint32_t group_size = get_compile_time_arg_val(9);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t num_rounds = get_compile_time_arg_val(11);
    constexpr uint32_t round_stride = get_compile_time_arg_val(12);  // tile rows between this core's rounds
    constexpr bool x_sharded = get_compile_time_arg_val(13) != 0;
    constexpr bool pre_combine = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t n_pre_tiles = get_compile_time_arg_val(15);
    constexpr bool gamma_stream = get_compile_time_arg_val(16) != 0;
    constexpr uint32_t num_chunks = get_compile_time_arg_val(17);  // W-chunks of the row-group (pre-combine sources)
    constexpr uint32_t gamma_batch = get_compile_time_arg_val(18);
    // DRAM stagger: tile row t walks its columns from (t % w_act) onwards, so the cores of one round do not all open
    // on the same DRAM banks (every core's first page sits at the same column offset of its chunk)
    constexpr bool dram_stagger = get_compile_time_arg_val(19) != 0;
    // W_local % 32 (0 = tile-aligned): valid columns of the device's last tile column; its tile padding is zeroed
    // here so it never enters the sum of squares (the normalize output's pad columns are don't-care)
    constexpr uint32_t x_last_cols = get_compile_time_arg_val(20);
    constexpr uint32_t x_elem_bytes = get_compile_time_arg_val(21);  // 4 fp32, 2 bf16, 1 bfp8_b (mantissa bytes)
    // ROW_MAJOR gamma: page c is the 32-element stick of tile column c (= the tile's row 0)
    constexpr bool gamma_rm = get_compile_time_arg_val(22) != 0;
    // gamma mcast: the column's sender reads the chunk's slivers, the other workers receive its cb_gamma pages
    constexpr bool gamma_mcast = get_compile_time_arg_val(23) != 0;
    constexpr auto x_args = TensorAccessorArgs<24>();
    constexpr auto gmc = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("gmc_ct_offset"), get_named_compile_time_arg_val("gmc_rt_offset")>();
    [[maybe_unused]] constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t round_bytes = group_size * slot_bytes;  // board region of one round
    constexpr uint32_t x_block_pages = block_h * block_w;      // cb_x_block capacity
    constexpr uint32_t g_tile_bytes = 4 * g_face_bytes;        // cb_gamma page (a tile; g_page_bytes is the DRAM page)

    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t g_addr = get_arg_val<uint32_t>(a++);
    const uint32_t row0 = get_arg_val<uint32_t>(a++);     // round 0's first tile row
    const uint32_t row_end = get_arg_val<uint32_t>(a++);  // past this core's last tile row
    const uint32_t col0 = get_arg_val<uint32_t>(a++);
    const uint32_t w_act = get_arg_val<uint32_t>(a++);    // tile columns compute walks (a padded shard's full width)
    const uint32_t w_valid = get_arg_val<uint32_t>(a++);  // tile columns that hold tensor data (<= w_act)
    const bool is_leader = get_arg_val<uint32_t>(a++) != 0;
    const uint32_t board_addr = get_arg_val<uint32_t>(a++);
    const uint32_t rt_base = get_arg_val<uint32_t>(a++);
    const uint32_t port_xy[2] = {get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1)};  // fwd, bwd of the link
    a += 2;
    const uint32_t src_ports_idx = a;  // group_size entries: (y << 16) | x of the port holding slot g

    // tile rows of round r (0 = this row-slot has no row-block in round r; only the last round can be short)
    auto rows_of = [&](uint32_t r) -> uint32_t {
        const uint32_t first = row0 + r * round_stride;
        if (first >= row_end) {
            return 0;
        }
        return (row_end - first) < block_h ? (row_end - first) : block_h;
    };

    // ---- scaler (constant SUM scaler 1.0; the 1/W is applied in the combine post-op) -------------------------------
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(1.0f);

    // ---- W tile-padding mask: zero columns [x_last_cols, 32) of the tile at byte offset `off` of cb_x_block's write
    // window. Whole pad faces go through the DM engine; only the partial face rows (<= 32 rows x < 16 elements) are
    // RISC stores. bfp8_b: a zero mantissa byte is a zero value whatever the shared exponent (64 B exponent header).
    constexpr uint32_t x_data_off = x_elem_bytes == 1 ? 64 : 0;
    constexpr uint32_t x_face_row_bytes = 16 * x_elem_bytes;
    constexpr uint32_t x_face_bytes = 16 * x_face_row_bytes;
    constexpr uint32_t mask_lo = x_last_cols % 16;             // first pad column inside the partial faces
    constexpr uint32_t mask_pface = x_last_cols < 16 ? 0 : 1;  // partial faces: {0, 2} or {1, 3}
    // whole pad faces: {1, 3} when the valid columns end in the left faces; {1, 3} too when they end exactly at 16
    constexpr bool mask_full_right = x_last_cols <= 16;
    auto mask_x_tile = [&](Noc& noc, CircularBuffer& x_cb, uint32_t off) {
        if constexpr (x_last_cols != 0) {
            if constexpr (mask_full_right) {
                for (uint32_t f = 1; f < 4; f += 2) {
                    noc.async_write_zeros(x_cb, x_face_bytes, {.offset_bytes = off + x_data_off + f * x_face_bytes});
                }
            }
            if constexpr (x_last_cols != 16) {
                const uint32_t base = x_cb.get_write_ptr() + off + x_data_off;
                for (uint32_t f = mask_pface; f < 4; f += 2) {
                    for (uint32_t row = 0; row < 16; ++row) {
                        uint32_t addr = base + f * x_face_bytes + row * x_face_row_bytes + mask_lo * x_elem_bytes;
                        const uint32_t end = base + f * x_face_bytes + (row + 1) * x_face_row_bytes;
                        for (; addr < end && (addr & 3); ++addr) {
                            *reinterpret_cast<volatile tt_l1_ptr uint8_t*>(addr) = 0;
                        }
                        for (; addr < end; addr += 4) {  // face rows end 16 B aligned
                            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = 0;
                        }
                    }
                }
            }
        }
    };
    // this worker's chunk holds the device's last tile column iff it ends at tensor_w_tiles
    const bool has_last_col = x_last_cols != 0 && col0 + w_valid == tensor_w_tiles;

    // ---- load_block: x (resident for one round). The whole capacity is pushed every round (pages past h*w_act carry
    // no data and are popped unread by compute) so each round's block starts at the FIFO base: compute reads it as one
    // contiguous block, and a ragged chunk (w_act < block_w) would otherwise leave the pointer mid-FIFO.
    // Sharded: the shard is already in this core's L1 behind cb_x_block, so a round's load is a push of its h*w_act
    // pages (rounds walk the shard's rows contiguously from the FIFO base). The pad columns of a padded last WIDTH
    // shard are zero-filled once up front: compute walks the whole shard width, and zeros add nothing to the sum.
    if constexpr (x_sharded) {
        if (w_valid < w_act || has_last_col) {
            Noc noc;
            CircularBuffer x_cb(cb_x_block);
            for (uint32_t rr = 0; rr < row_end - row0; ++rr) {
                if (w_valid < w_act) {
                    noc.async_write_zeros(
                        x_cb,
                        (w_act - w_valid) * x_page_bytes,
                        {.offset_bytes = (rr * w_act + w_valid) * x_page_bytes});
                }
                if (has_last_col) {
                    mask_x_tile(noc, x_cb, (rr * w_act + w_valid - 1) * x_page_bytes);
                }
            }
            noc.write_zeros_l1_barrier();
        }
    }
    const auto x_acc = TensorAccessor(x_args, x_addr, x_page_bytes);
    auto load_x = [&](uint32_t r) {
        const uint32_t h = rows_of(r);
        if (h == 0) {
            return;
        }
        if constexpr (x_sharded) {
            cb_reserve_back(cb_x_block, h * w_act);
            cb_push_back(cb_x_block, h * w_act);
            return;
        }
        MaybeDeviceZoneScope("R_xload");
        const uint32_t first = row0 + r * round_stride;
        cb_reserve_back(cb_x_block, x_block_pages);
        const uint32_t wbase = get_write_ptr(cb_x_block);
        for (uint32_t rr = 0; rr < h; ++rr) {
            const uint32_t page0 = (first + rr) * tensor_w_tiles + col0;
            const uint32_t row_ptr = wbase + rr * w_act * x_page_bytes;
            const uint32_t rot = dram_stagger ? (first + rr) % w_act : 0;
            for (uint32_t i = 0, c = rot; i < w_act; ++i, c = (c + 1 == w_act) ? 0 : c + 1) {
                noc_async_read_page(page0 + c, x_acc, row_ptr + c * x_page_bytes);
            }
            noc_async_read_barrier();  // one barrier per batch (block_w tiles)
        }
        if (has_last_col) {
            Noc noc;
            CircularBuffer x_cb(cb_x_block);
            for (uint32_t rr = 0; rr < h; ++rr) {
                mask_x_tile(noc, x_cb, (rr * w_act + w_valid - 1) * x_page_bytes);
            }
            noc.write_zeros_l1_barrier();
        }
        cb_push_back(cb_x_block, x_block_pages);
    };

    // ---- gamma slivers (row 0 of face 0 and face 1 of each tile) of tile columns [c0, c0 + n) into one push; columns
    // past w_valid (a padded shard's pad) get zero pages
    // fill: cb_gamma's write window (already reserved) gets the n pages; push_gamma reserves, fills and pushes
    // issue: the DRAM reads only (no barrier); finish: barrier, RM stick relocation, zero pad pages
    auto issue_gamma = [&](uint32_t c0, uint32_t n) {
        if constexpr (has_gamma) {
            const auto g_acc = TensorAccessor(g_args, g_addr, g_page_bytes);
            uint32_t wptr = get_write_ptr(cb_gamma);
            for (uint32_t c = c0; c < c0 + n && c < w_valid; ++c) {
                if constexpr (gamma_rm) {
                    // the whole stick (64 B aligned DRAM read) lands as face0 rows 0-1; its second half is moved
                    // to face1 row 0 below, once it has landed
                    noc_async_read(g_acc.get_noc_addr(col0 + c, 0), wptr, g_page_bytes);
                } else {
                    noc_async_read(g_acc.get_noc_addr(col0 + c, 0), wptr, sliver_bytes);
                    noc_async_read(g_acc.get_noc_addr(col0 + c, g_face_bytes), wptr + g_face_bytes, sliver_bytes);
                }
                wptr += g_tile_bytes;
            }
        }
    };
    auto finish_gamma = [&](uint32_t c0, uint32_t n) {
        if constexpr (has_gamma) {
            const uint32_t wbase = get_write_ptr(cb_gamma);
            const uint32_t n_real = c0 >= w_valid ? 0 : ((w_valid - c0) < n ? (w_valid - c0) : n);
            noc_async_read_barrier();
            if constexpr (gamma_rm) {
                constexpr uint32_t half_stick = g_face_bytes / 16;  // 16 elements = one face row
                for (uint32_t i = 0; i < n_real; ++i) {
                    const uint32_t t = wbase + i * g_tile_bytes;
                    noc_async_read(get_noc_addr(t + half_stick), t + g_face_bytes, half_stick);
                }
                noc_async_read_barrier();
            }
            if (n_real < n) {
                Noc noc;
                CircularBuffer g_cb(cb_gamma);
                noc.async_write_zeros(g_cb, (n - n_real) * g_tile_bytes, {.offset_bytes = n_real * g_tile_bytes});
                noc.write_zeros_l1_barrier();
            }
        }
    };
    auto push_gamma = [&](uint32_t c0, uint32_t n) {
        if constexpr (has_gamma) {
            cb_reserve_back(cb_gamma, n);
            issue_gamma(c0, n);
            finish_gamma(c0, n);
            cb_push_back(cb_gamma, n);
        }
    };
    // gamma mcast: the chunk's resident gamma block arrives at cb_gamma's base on every worker of the column. One
    // block per call into an empty CB, so the window is free on every receiver from the launch (no handshake).
    // Counter signal: a receiver may construct its pipe after the sender's increment has landed (a Flag pipe's
    // constructor would clear it); the counter is re-armed right after the call's single receive.
    // The sender multicasts right after its own load_x(0); every worker pushes after load_x(0).
    // The sender's sliver reads are issued ahead of its x(0) reads (they share load_x(0)'s barrier)
    auto issue_send_gamma = [&]() {
        if constexpr (has_gamma && gamma_mcast) {
            if (gmc.can_send()) {
                cb_reserve_back(cb_gamma, w_act);
                issue_gamma(0, w_act);
            }
        }
    };
    auto send_gamma = [&]() {
        if constexpr (has_gamma && gamma_mcast) {
            if (gmc.can_send()) {
                MaybeDeviceZoneScope("R_gsend");
                finish_gamma(0, w_act);
                gamma_mcast_send(gmc, get_write_ptr(cb_gamma), w_act * g_tile_bytes);
            }
        }
    };
    auto recv_gamma = [&]() {
        if constexpr (has_gamma && gamma_mcast) {
            cb_reserve_back(cb_gamma, w_act);
            if (!gmc.can_send()) {
                MaybeDeviceZoneScope("R_grecv");
                gamma_mcast_receive(gmc);
            }
            cb_push_back(cb_gamma, w_act);
        }
    };
    // resident: constant over rounds, read once
    auto load_gamma = [&]() {
        if constexpr (gamma_mcast) {
            recv_gamma();
        } else {
            push_gamma(0, w_act);
        }
    };
    // streamed: once per tile row of round r, in the order the fused normalize walks the block
    auto stream_gamma = [&](uint32_t r) {
        const uint32_t h = rows_of(r);
        for (uint32_t rr = 0; rr < h; ++rr) {
            for (uint32_t c = 0; c < w_act; c += gamma_batch) {
                push_gamma(c, (w_act - c) < gamma_batch ? (w_act - c) : gamma_batch);
            }
        }
    };

    // board_ready has two sources (the link's two ports), so it is counted per round parity: a port signals round
    // r+2 only after this leader has gathered round r, so even and odd rounds never mix within one counter.
    volatile tt_l1_ptr uint32_t* board_ready_par[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_board_ready_id)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_board_ready_odd_id))};
    volatile tt_l1_ptr uint32_t* inv_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_inv_ready_id));

    // ---- leader: gather round r's board region into cb_gather (counts are cumulative over rounds, reset at the end)
    auto gather = [&](uint32_t r) {
        const uint32_t h = rows_of(r);
        if (h == 0) {
            // an empty round is still counted: both ports signal every round
            noc_semaphore_wait_min(board_ready_par[r & 1], ports_per_link * (r / 2 + 1));
            return;
        }
        const uint32_t n_gather_pages = n_gather_tiles * h;
        cb_reserve_back(cb_gather, n_gather_pages);
        const uint32_t gather_base = get_write_ptr(cb_gather);
        {
            // rows past G * num_w_chunks enter the REDUCE_COL sum: zero the whole window first (DM engine, not RISC)
            Noc noc;
            CircularBuffer gather_cb(cb_gather);
            noc.async_write_zeros(gather_cb, n_gather_pages * fp32_tile_bytes, {.offset_bytes = 0});
            noc.write_zeros_l1_barrier();
        }
        // both ports of the link report round r's board region complete
        {
            MaybeDeviceZoneScope("R_gwait");
            noc_semaphore_wait_min(board_ready_par[r & 1], ports_per_link * (r / 2 + 1));
        }
        MaybeDeviceZoneScope("R_gread");
        const uint32_t round_base = board_addr + r * round_bytes;
        for (uint32_t g = 0; g < group_size; ++g) {
            const uint32_t xy = get_arg_val<uint32_t>(src_ports_idx + g);
            const uint32_t px = xy & 0xFFFF, py = xy >> 16;
            const uint32_t slot_base = round_base + g * slot_bytes;
            for (uint32_t rt = 0; rt < h; ++rt) {
                for (uint32_t hh = 0; hh < 2; ++hh) {
                    // run of num_w_chunks half-rows (one per chunk) for (g, rt, hh); rows s = g*C + wc
                    const uint32_t src = slot_base + ((rt_base + rt) * 2 + hh) * num_w_chunks * half_row_bytes;
                    uint32_t wc = 0;
                    while (wc < num_w_chunks) {
                        const uint32_t s = g * num_w_chunks + wc;
                        const uint32_t band_left = 16 - (s % 16);  // contiguous rows until the 16-row band ends
                        const uint32_t n = (num_w_chunks - wc) < band_left ? (num_w_chunks - wc) : band_left;
                        const uint32_t t = s / 32;
                        const uint32_t face = ((s % 32) < 16 ? 0 : 2) + hh;
                        const uint32_t dst = gather_base + (t * h + rt) * fp32_tile_bytes + face * face_bytes_fp32 +
                                             (s % 16) * half_row_bytes;
                        noc_async_read(get_noc_addr(px, py, src + wc * half_row_bytes), dst, n * half_row_bytes);
                        wc += n;
                    }
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_gather, n_gather_pages);
    };

    // ---- pre-combine (leader): round r's num_chunks partial vectors (row s = chunk) land in half r % 2 of
    // cb_pregather from the row-group's writers. Rows past num_chunks enter the REDUCE_COL sum and are never written
    // by a writer, so zeroing them here races with nothing; the count is kept per round parity (several sources, a
    // writer is at most one round ahead).
    volatile tt_l1_ptr uint32_t* pre_ready_par[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_pre_ready_id)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_pre_ready_odd_id))};
    auto pregather = [&](uint32_t r) {
        if constexpr (pre_combine) {
            const uint32_t h = rows_of(r);
            if (h == 0) {
                return;
            }
            const uint32_t n_pages = n_pre_tiles * h;
            cb_reserve_back(cb_pregather, n_pages);
            constexpr uint32_t s0 = num_chunks % 32;  // first unused row of the last tile (0: none unused)
            if constexpr (s0 != 0) {
                Noc noc;
                CircularBuffer pre_cb(cb_pregather);
                constexpr uint32_t t = n_pre_tiles - 1;
                for (uint32_t rt = 0; rt < h; ++rt) {
                    const uint32_t tile_off = (t * h + rt) * fp32_tile_bytes;
                    for (uint32_t face = 0; face < 4; ++face) {
                        const uint32_t lo = face < 2 ? s0 : (s0 > 16 ? s0 - 16 : 0);  // first unused row in the face
                        if (lo < 16) {
                            noc.async_write_zeros(
                                pre_cb,
                                (16 - lo) * half_row_bytes,
                                {.offset_bytes = tile_off + face * face_bytes_fp32 + lo * half_row_bytes});
                        }
                    }
                }
                noc.write_zeros_l1_barrier();
            }
            noc_semaphore_wait_min(pre_ready_par[r & 1], num_chunks * (r / 2 + 1));
            cb_push_back(cb_pregather, n_pages);
        }
    };

    // ---- inv_rms row vectors of round r from the row-group leader (cumulative sem_inv_ready)
    auto push_inv = [&](uint32_t r) {
        const uint32_t h = rows_of(r);
        if (h == 0) {
            return;
        }
        MaybeDeviceZoneScope("R_invwait");
        cb_reserve_back(cb_inv_row, h);
        noc_semaphore_wait_min(inv_ready, r + 1);
        cb_push_back(cb_inv_row, h);
    };

    // Round order mirrors compute. With the stall shadow (resident gamma), x(r) is consumed by scale_by_gamma before
    // round r's stats arrive, so x(r+1) is loaded (and its sumsq/exchange started) before round r's combine; otherwise
    // x(r) is held until normalize(r), so rounds run back to back.
    constexpr bool stall_shadow = has_gamma && !gamma_stream;
    if constexpr (stall_shadow) {
        issue_send_gamma();
        load_x(0);  // round 0's partials start the exchange chain; gamma is needed only by x*gamma
        send_gamma();
        load_gamma();
        for (uint32_t r = 0; r < num_rounds; ++r) {
            if (r + 1 < num_rounds) {
                load_x(r + 1);
            }
            if (is_leader) {
                pregather(r);
                gather(r);
            }
            push_inv(r);
        }
    } else {
        for (uint32_t r = 0; r < num_rounds; ++r) {
            load_x(r);
            if (is_leader) {
                pregather(r);
                gather(r);
            }
            push_inv(r);
            if constexpr (gamma_stream) {
                stream_gamma(r);
            }
        }
    }
    // every increment of the counters has landed (the last waits above covered all of them): re-arm for next call
    if (is_leader) {
        // the board is fully read (every gather ended in a read barrier): the ports may hand the credit back
        for (uint32_t q = 0; q < 2; ++q) {
            noc_semaphore_inc(
                get_noc_addr(port_xy[q] & 0xFFFF, port_xy[q] >> 16, get_semaphore(sem_gather_done_id)), 1);
        }
        noc_semaphore_set(board_ready_par[0], 0);
        noc_semaphore_set(board_ready_par[1], 0);
        if constexpr (pre_combine) {
            noc_semaphore_set(pre_ready_par[0], 0);
            noc_semaphore_set(pre_ready_par[1], 0);
        }
    }
    noc_semaphore_set(inv_ready, 0);
}
