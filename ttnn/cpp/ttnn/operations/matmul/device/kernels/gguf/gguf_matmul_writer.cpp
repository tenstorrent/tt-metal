// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Input and output mover of the GGUF matmul. Every compute core holds the input row block (K tiles in c_1, at the
// same L1 address on every core of the grid). The collector reads it chunk by chunk, multicasts each chunk to the
// grid and then sets the chunk count in the grid's semaphore with a multicast ordered behind the data; the others
// push each chunk once the count reaches it, so the matmul starts on chunk 0 while later chunks are in flight. Then,
// per segment (gguf_matmul_program_factory.cpp Role): whole and finish write the bf16 output tile (c_16); a tail sends
// its fp32 partial (c_17) into its slot of the finishing core's c_18 and counts it on that core's semaphore; a head
// moves its partial into this core's c_20; a finish first waits for its partials and hands them to the compute kernel.
// Compile args: K tiles, tiles per chunk, input semaphore, partial semaphore, the input's and the output's
// TensorAccessorArgs. Runtime args: collector, input address, output address, multicast rectangle (NoC x / y start,
// x / y end, in this writer's NoC order), cores in it, segments, then per segment: tail: slot, role, finishing core NoC
// x, y; else: N tile, role, partials, 0.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"

void kernel_main() {
    constexpr uint32_t whole = 0, tail = 1, head = 2;  // Role
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t input_sem = get_compile_time_arg_val(2);
    constexpr uint32_t partial_sem = get_compile_time_arg_val(3);
    constexpr auto input_args = TensorAccessorArgs<4>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048, partial_bytes = 4096;

    const bool collector = get_arg_val<uint32_t>(0) == 1;
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(1));
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(2));
    const uint32_t mc_x0 = get_arg_val<uint32_t>(3), mc_y0 = get_arg_val<uint32_t>(4);
    const uint32_t mc_x1 = get_arg_val<uint32_t>(5), mc_y1 = get_arg_val<uint32_t>(6);
    const uint32_t mc_dests = get_arg_val<uint32_t>(7) - 1;
    const uint32_t segments = get_arg_val<uint32_t>(8);

    Noc noc;
    CircularBuffer in_cb(tt::CBIndex::c_1);
    CircularBuffer out_cb(tt::CBIndex::c_16);
    CircularBuffer part_cb(tt::CBIndex::c_17);
    CircularBuffer remote_cb(tt::CBIndex::c_18);
    CircularBuffer own_cb(tt::CBIndex::c_20);
    Semaphore chunks(input_sem), partials(partial_sem);
    const uint32_t remote_addr = remote_cb.get_write_ptr();  // the same on every compute core
    const UnicastEndpoint core;
    const uint32_t me_x = my_x[noc.get_noc_id()], me_y = my_y[noc.get_noc_id()];

    in_cb.reserve_back(k_tiles);
    const uint32_t base = in_cb.get_write_ptr();
    // Collector: chunk c + 1 is read while chunk c is multicast.
    auto read_chunk = [&](uint32_t k0) {
        for (uint32_t kt = k0; kt < k0 + chunk && kt < k_tiles; kt++) {
            noc.async_read(input, CoreLocalMem<uint32_t>(base + kt * tile_bytes), tile_bytes, {.page_id = kt}, {});
        }
    };
    if (collector) {
        read_chunk(0);
    }
    for (uint32_t k0 = 0, c = 1; k0 < k_tiles; k0 += chunk, c++) {
        const uint32_t n = k_tiles - k0 < chunk ? k_tiles - k0 : chunk;
        if (collector) {
            noc.async_read_barrier();
            if (mc_dests > 0) {
                noc.async_writes_flushed();  // the previous count has left before it is overwritten
                const MulticastEndpoint grid;
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(base + k0 * tile_bytes),
                    grid,
                    n * tile_bytes,
                    mc_dests,
                    {},
                    {.noc_x_start = mc_x0,
                     .noc_y_start = mc_y0,
                     .noc_x_end = mc_x1,
                     .noc_y_end = mc_y1,
                     .addr = base + k0 * tile_bytes},
                    true);
                chunks.set(c);
                chunks.set_multicast(noc, mc_x0, mc_y0, mc_x1, mc_y1, mc_dests);
            }
            if (k0 + chunk < k_tiles) {
                read_chunk(k0 + chunk);
            }
        } else {
            chunks.wait_min(c);
        }
        in_cb.push_back(n);
    }

    for (uint32_t s = 0; s < segments; s++) {
        const uint32_t n = get_arg_val<uint32_t>(9 + 4 * s);
        const uint32_t role = get_arg_val<uint32_t>(10 + 4 * s);
        if (role == tail) {
            const uint32_t x = get_arg_val<uint32_t>(11 + 4 * s), y = get_arg_val<uint32_t>(12 + 4 * s);
            part_cb.wait_front(1);
            noc.async_write(
                part_cb, core, partial_bytes, {}, {.noc_x = x, .noc_y = y, .addr = remote_addr + n * partial_bytes});
            noc.async_write_barrier();  // the partial has landed before it is counted
            partials.up(noc, x, y, 1);
            part_cb.pop_front(1);
        } else if (role == head) {
            part_cb.wait_front(1);
            own_cb.reserve_back(1);
            noc.async_write(
                part_cb, core, partial_bytes, {}, {.noc_x = me_x, .noc_y = me_y, .addr = own_cb.get_write_ptr()});
            noc.async_write_barrier();
            own_cb.push_back(1);
            part_cb.pop_front(1);
        } else {
            if (role != whole) {
                const uint32_t parts = get_arg_val<uint32_t>(11 + 4 * s);
                partials.wait_min(parts);
                remote_cb.reserve_back(parts);
                remote_cb.push_back(parts);
            }
            out_cb.wait_front(1);
            noc.async_write(out_cb, output, tile_bytes, {}, {.page_id = n});
            noc.async_writes_flushed();
            out_cb.pop_front(1);
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
