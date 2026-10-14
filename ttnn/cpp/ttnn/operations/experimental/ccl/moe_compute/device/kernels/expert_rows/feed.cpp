// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows feed for the combine (data movement only), on the feeder cores (one per ring position). Feeder r copies
// its W2 output column slice [c0, c0 + nc) tiles of every routed row from the row buffer into moe_compute's double
// buffer on the combine cores, one local expert at a time, with the handshake of the ring program's dm1 (dm1.cpp):
// the slice may span several width shards (columns); before expert e the feeder waits until the NTP combine cores
// of each of those columns released the half e % 2 (one increment each per expert), after expert e it increments
// each of those cores' semaphore once. Expert e's n_e rows go to the NTP height shards in order (earlier shards take
// the remainder), row dt of shard h at half * BLOCK + dt * SEG; the slice's bytes land at its offset within each
// width shard.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_s = get_named_compile_time_arg_val("cb_scratch");
    constexpr uint32_t E = get_named_compile_time_arg_val("local_experts");
    constexpr uint32_t NTP = get_named_compile_time_arg_val("height_shards");
    constexpr uint32_t DP = get_named_compile_time_arg_val("width_shards");
    constexpr uint32_t SHARD_TILES = get_named_compile_time_arg_val("shard_width_tiles");
    constexpr uint32_t TWB = get_named_compile_time_arg_val("tile_width_bytes");
    constexpr uint32_t BLOCK = get_named_compile_time_arg_val("half_bytes");
    constexpr uint32_t ROW_BYTES = get_named_compile_time_arg_val("row_bytes");
    constexpr uint32_t BATCH = get_named_compile_time_arg_val("batch_rows");
    constexpr uint32_t SLICE = get_named_compile_time_arg_val("slice_stride");
    constexpr uint32_t P_RTC = get_named_compile_time_arg_val("table_counts");
    constexpr uint32_t P_RTO = get_named_compile_time_arg_val("table_offsets");
    constexpr uint32_t P_RTR = get_named_compile_time_arg_val("table_rows");
    constexpr uint32_t P_RTBS = get_named_compile_time_arg_val("table_bytes");
    constexpr uint32_t STAGE = get_named_compile_time_arg_val("s_stage");
    constexpr uint32_t SEM = get_named_compile_time_arg_val("combine_sync_semaphore_id");
    constexpr uint32_t SEG = SHARD_TILES * TWB;
    constexpr auto rows_args = TensorAccessorArgs<0>();
    constexpr auto tab_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();

    uint32_t a = 0;
    const uint32_t rows_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t dense = get_arg_val<uint32_t>(a++);  // double buffer: same L1 address on every combine core
    const uint32_t c0 = get_arg_val<uint32_t>(a++);
    const uint32_t nc = get_arg_val<uint32_t>(a++);
    const uint32_t cxy = a;  // combine cores (x, y), shard order h * DP + w

    cb_reserve_back(cb_s, 1);
    const uint32_t s = get_write_ptr(cb_s);
    const auto tab = TensorAccessor(tab_args, tab_addr, P_RTBS);
    noc_async_read(tab.get_noc_addr(0), s, P_RTR);
    noc_async_read_barrier();
    const tt_l1_ptr uint16_t* cnt = reinterpret_cast<const tt_l1_ptr uint16_t*>(s + P_RTC);
    const tt_l1_ptr uint16_t* ofs = reinterpret_cast<const tt_l1_ptr uint16_t*>(s + P_RTO);

    // the slice in at most three width-shard pieces: source offset, width shard, offset in it, bytes
    uint32_t pw[3], po[3], pb[3], ps[3], np = 0;
    for (uint32_t t = c0, done = 0; done < nc && np < 3; ++np) {
        const uint32_t in = t % SHARD_TILES;
        const uint32_t n = SHARD_TILES - in < nc - done ? SHARD_TILES - in : nc - done;
        ps[np] = done * TWB;
        pw[np] = t / SHARD_TILES;
        po[np] = in * TWB;
        pb[np] = n * TWB;
        t += n;
        done += n;
    }

    const auto rw = TensorAccessor(rows_args, rows_addr, ROW_BYTES);
    const uint32_t sem_addr = get_semaphore(SEM);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    uint32_t released = 0;
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t n = cnt[e], first = ofs[e];
        const uint32_t half = (e & 1) * BLOCK;
        auto issue_reads = [&](uint32_t j) {
            const uint32_t i0 = j * BATCH;
            const uint32_t nb = n - i0 < BATCH ? n - i0 : BATCH;
            const uint32_t buf = s + STAGE + (j & 1) * BATCH * SLICE;
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_read(rw.get_noc_addr(first + i0 + i, c0 * TWB), buf + i * SLICE, nc * TWB);
            }
        };
        const uint32_t batches = (n + BATCH - 1) / BATCH;
        if (batches) {
            issue_reads(0);
        }
        noc_semaphore_wait_min(sem, released);  // the combine released this half
        const uint32_t q = n / NTP, rem = n % NTP;
        uint32_t h = 0, row = 0, cap = q + (rem ? 1 : 0);
        for (uint32_t j = 0; j < batches; ++j) {
            noc_async_read_barrier();
            noc_async_writes_flushed();  // batch j - 1 left the other staging buffer
            if (j + 1 < batches) {
                issue_reads(j + 1);
            }
            const uint32_t i0 = j * BATCH;
            const uint32_t nb = n - i0 < BATCH ? n - i0 : BATCH;
            const uint32_t buf = s + STAGE + (j & 1) * BATCH * SLICE;
            for (uint32_t i = 0; i < nb; ++i) {
                for (uint32_t p = 0; p < np; ++p) {
                    const uint32_t idx = h * DP + pw[p];
                    const uint32_t x = get_arg_val<uint32_t>(cxy + 2 * idx);
                    const uint32_t y = get_arg_val<uint32_t>(cxy + 2 * idx + 1);
                    noc_async_write(
                        buf + i * SLICE + ps[p], get_noc_addr(x, y, dense + half + row * SEG + po[p]), pb[p]);
                }
                if (++row == cap) {
                    ++h;
                    row = 0;
                    cap = q + (h < rem ? 1 : 0);
                }
            }
        }
        noc_async_write_barrier();  // the rows are in the combine cores' L1 before their semaphore moves
        for (uint32_t p = 0; p < np; ++p) {
            for (uint32_t y = 0; y < NTP; ++y) {
                const uint32_t idx = y * DP + pw[p];
                noc_semaphore_inc(
                    get_noc_addr(
                        get_arg_val<uint32_t>(cxy + 2 * idx), get_arg_val<uint32_t>(cxy + 2 * idx + 1), sem_addr),
                    1);
            }
        }
        released += NTP * np;  // every combine core signalled releases this feeder once per expert
    }
    noc_semaphore_wait_min(sem, released);  // the combine's last release, then reset for the next call
    noc_semaphore_set(sem, 0);
    noc_async_atomic_barrier();
}
