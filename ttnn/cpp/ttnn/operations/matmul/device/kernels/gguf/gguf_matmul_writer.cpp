// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Input and output mover of the GGUF matmul. Every compute core holds the input row block (K tiles in c_1, at the same
// L1 address on every core of the grid). The collector reads it chunk by chunk, multicasts each chunk to the grid and
// then sets the chunk count in the grid's semaphore with a multicast ordered behind the data; the others push each
// chunk once the count reaches it, so the matmul starts on chunk 0 while later chunks are in flight. Then, per segment
// (gguf_matmul_program_factory.cpp Role): whole and finish write the bf16 output tile (c_16); a tail sends its fp32
// partial (c_17) into its slot of the finishing core's c_18 and counts it on that core's semaphore; a head moves its
// partial into this core's c_20; a finish first waits for its partials and hands them to the compute kernel. Codebook
// formats: the writer also runs the second half of the grid lookups of each record the reader asks for, between its own
// waits; Q2_K: the conversion of each record's scale rows. Compile args: K tiles, tiles per chunk, input
// semaphore, partial semaphore, format id, the lookup request, answer and page address semaphores, the input's and the
// output's TensorAccessorArgs. Runtime args: collector, input address, output address, multicast rectangle (NoC x / y
// start, x / y end, in this writer's NoC order), cores in it, segments, then per segment: tail: slot, role, finishing
// core NoC x, y; else: N tile, role, partials, 0.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "experimental/gguf_page.h"

alignas(4) std::uint16_t table_copy[1024 / 2];  // level tables of up to 1 KiB, as in gguf_matmul_reader.cpp

void kernel_main() {
    using namespace experimental::gguf;
    constexpr uint32_t whole = 0, tail = 1, head = 2;  // Role
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t input_sem = get_compile_time_arg_val(2);
    constexpr uint32_t partial_sem = get_compile_time_arg_val(3);
    constexpr auto FMT = static_cast<Format>(get_compile_time_arg_val(4));
    constexpr uint32_t request_sem = get_compile_time_arg_val(5), answer_sem = get_compile_time_arg_val(6);
    constexpr uint32_t address_sem = get_compile_time_arg_val(7);
    constexpr auto input_args = TensorAccessorArgs<8>();
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

    // Codebook formats: per request of the reader (the record's first page in address_sem, then the count of requests
    // in request_sem), the second half of the index words of each page; the count served goes to answer_sem.
    constexpr PageLayout L = page_layout<FMT>();
    constexpr uint32_t P = L.group_tiles / L.page_tiles, page = L.rows * 64;
    constexpr bool split = is_codebook<FMT>() && P > 1;
    // Q2_K (two fp16 scale rows): per request, the record's fp32 scale copies (prepare_scales) in its first
    // page, then the scale block copied into the other pages.
    constexpr bool prep = L.fp16_rows > 1 && !is_codebook<FMT>() && P > 1;
    Semaphore requests(request_sem), answers(answer_sem), address(address_sem);
    uint32_t served = 0;
    [[maybe_unused]] const std::uint16_t* table = nullptr;
    if constexpr (split) {
        table = level_table<FMT>();
        if constexpr (table_entries<FMT>() * 2 <= 1024) {
            for (uint32_t j = 0; j < table_entries<FMT>() / 2; j++) {
                reinterpret_cast<uint32_t*>(table_copy)[j] = reinterpret_cast<const uint32_t*>(table)[j];
            }
            table = table_copy;
        }
    }
    auto serve = [&]() {
        if constexpr (split) {
            if (requests.value() > served) {
                const uint32_t first = address.value();
                invalidate_l1_cache();
                for (uint32_t p = 0; p < P; p++) {
                    lookup_grid<FMT>(
                        reinterpret_cast<const uint32_t*>(first + p * page + index_offset<FMT>()),
                        table,
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(first + p * page),
                        1,
                        2);
                }
                asm volatile("fence" ::: "memory");
                answers.set(++served);
            }
        } else if constexpr (prep) {
            if (requests.value() > served) {
                const uint32_t first = address.value();
                invalidate_l1_cache();
                prepare_scales<FMT>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(first + L.scale_offset));
                asm volatile("fence" ::: "memory");
                for (uint32_t p = 1; p < P; p++) {
                    noc.async_read(
                        core,
                        CoreLocalMem<uint32_t>(first + p * page + L.scale_offset),
                        L.scale_bytes,
                        {.noc_x = me_x, .noc_y = me_y, .addr = first + L.scale_offset},
                        {});
                }
                noc.async_read_barrier();
                answers.set(++served);
            }
        }
    };

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
            while (chunks.value() < c) {
                serve();
            }
        }
        in_cb.push_back(n);
    }

    for (uint32_t s = 0; s < segments; s++) {
        const uint32_t n = get_arg_val<uint32_t>(9 + 4 * s);
        const uint32_t role = get_arg_val<uint32_t>(10 + 4 * s);
        if (role == tail) {
            const uint32_t x = get_arg_val<uint32_t>(11 + 4 * s), y = get_arg_val<uint32_t>(12 + 4 * s);
            while (!part_cb.pages_available_at_front(1)) {
                serve();
            }
            noc.async_write(
                part_cb, core, partial_bytes, {}, {.noc_x = x, .noc_y = y, .addr = remote_addr + n * partial_bytes});
            noc.async_write_barrier();  // the partial has landed before it is counted
            partials.up(noc, x, y, 1);
            part_cb.pop_front(1);
        } else if (role == head) {
            while (!part_cb.pages_available_at_front(1)) {
                serve();
            }
            own_cb.reserve_back(1);
            noc.async_write(
                part_cb, core, partial_bytes, {}, {.noc_x = me_x, .noc_y = me_y, .addr = own_cb.get_write_ptr()});
            noc.async_write_barrier();
            own_cb.push_back(1);
            part_cb.pop_front(1);
        } else {
            if (role != whole) {
                const uint32_t parts = get_arg_val<uint32_t>(11 + 4 * s);
                while (partials.value() < parts) {
                    serve();
                }
                remote_cb.reserve_back(parts);
                remote_cb.push_back(parts);
            }
            while (!out_cb.pages_available_at_front(1)) {
                serve();
            }
            noc.async_write(out_cb, output, tile_bytes, {}, {.page_id = n});
            noc.async_writes_flushed();
            out_cb.pop_front(1);
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
