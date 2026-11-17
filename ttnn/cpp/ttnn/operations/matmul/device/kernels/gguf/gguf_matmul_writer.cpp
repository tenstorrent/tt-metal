// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Input and output mover of the GGUF matmul. Every compute core holds its band's K tiles of the input row block (c_1,
// at the same L1 address on every core of the grid). The band's collector reads them chunk by chunk, multicasts each
// chunk to the band's rows and then sets the chunk count in their semaphore with a multicast ordered behind the data;
// the others push each chunk once the count reaches it, so the matmul starts on chunk 0 while later chunks are in
// flight. Then, per segment (gguf_matmul_program_factory.cpp Role): whole and finish write the bf16 output tile (c_16);
// a tail sends its fp32 partial (c_17) into its slot of the finishing core's c_18 and counts it on that core's
// semaphore; a head moves its partial into this core's c_20; the first finish waits for all of this core's partials,
// and each hands its own to the compute kernel. Codebook formats: the writer also runs its part of the grid lookups of
// each record the reader asks for, between its own waits. GGUF_STREAM (decode, Q4_0 / Q4_1 / Q5_0 / Q5_1): between its
// waits, and once its own input reads are over, the writer also reads the weight units the reader grants it
// (gguf_matmul_reader.cpp) on this NoC. TQ2_0 with one input row (gguf_matmul_tq2_compute.cpp): the collector zeroes
// every row of each input tile but row 0 before the multicast, and c_1 counts 32-byte pages. Compile args: K tiles
// (c_1's; of the largest band), tiles per chunk, input semaphore, partial semaphore, format id, the lookup request,
// answer and page address semaphores, one-row TQ2_0 (0 / 1), K bands (0 / 1), the input's and the output's
// TensorAccessorArgs; GGUF_STREAM: c_0 pages, records per unit, the weight's TensorAccessorArgs. Runtime args:
// collector, input address, output address, multicast rectangle (NoC x / y start, x / y end, in this writer's NoC
// order), cores in it, segments, then per segment: tail: slot, role, finishing core NoC x, y; else: N tile, role,
// partials, 0; with K bands, the band's first K tile and K tiles; GGUF_STREAM: N tiles, runs, per run its N tile,
// first and end K group (the reader's), and the weight address.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "experimental/gguf_page.h"

alignas(4) std::uint16_t table_copy[1024 / 2];  // local_level_table

void kernel_main() {
    using namespace experimental::gguf;
    constexpr uint32_t whole = 0, tail = 1, head = 2;  // Role
    constexpr uint32_t c1_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t input_sem = get_compile_time_arg_val(2);
    constexpr uint32_t partial_sem = get_compile_time_arg_val(3);
    constexpr auto FMT = static_cast<Format>(get_compile_time_arg_val(4));
    constexpr uint32_t request_sem = get_compile_time_arg_val(5), answer_sem = get_compile_time_arg_val(6);
    constexpr uint32_t address_sem = get_compile_time_arg_val(7);
    constexpr bool zero_rows = get_compile_time_arg_val(8) == 1;
    constexpr bool banded = get_compile_time_arg_val(9) == 1;
    constexpr auto input_args = TensorAccessorArgs<10>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048, partial_bytes = 4096;
    constexpr uint32_t in_pages = zero_rows ? tile_bytes / 32 : 1;  // c_1 pages per input tile

    const bool collector = get_arg_val<uint32_t>(0) == 1;
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(1));
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(2));
    const uint32_t mc_x0 = get_arg_val<uint32_t>(3), mc_y0 = get_arg_val<uint32_t>(4);
    const uint32_t mc_x1 = get_arg_val<uint32_t>(5), mc_y1 = get_arg_val<uint32_t>(6);
    const uint32_t mc_dests = get_arg_val<uint32_t>(7) - 1;
    const uint32_t segments = get_arg_val<uint32_t>(8);
    auto segment_arg = [](uint32_t s, uint32_t i) { return get_arg_val<uint32_t>(9 + 4 * s + i); };
    const uint32_t k_first = banded ? get_arg_val<uint32_t>(9 + 4 * segments) : 0;
    const uint32_t k_tiles = banded ? get_arg_val<uint32_t>(10 + 4 * segments) : c1_tiles;

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
    // in request_sem), index-word parts [GGUF_READER_PARTS, GGUF_READER_PARTS + GGUF_WRITER_PARTS) of each page; the
    // count served goes to answer_sem.
    constexpr PageLayout L = page_layout<FMT>();
    constexpr uint32_t P = L.group_tiles / L.page_tiles, page = L.rows * 64;
    constexpr bool split = is_codebook<FMT>() && P > 1 && GGUF_WRITER_PARTS > 0;
    Semaphore requests(request_sem), answers(answer_sem), address(address_sem);
    uint32_t served = 0;
    [[maybe_unused]] const std::uint16_t* table = level_table<FMT>();  // in L1 until the first request is answered
#if GGUF_STREAM
    // Weight units granted by the reader: its records into their c_0 pages, one unit at a time, with transaction id 1;
    // the answer semaphore counts 1 + units landed (1: this writer streams).
    static_assert(P == 1);
    constexpr auto weight_args = TensorAccessorArgs<output_args.next_compile_time_args_offset() + 2>();
    constexpr uint32_t ring = get_compile_time_arg_val(output_args.next_compile_time_args_offset());
    constexpr uint32_t unit = get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t record = L.scale_rows * 64 + L.code_bytes, weight_trid = 1;
    const uint32_t runs_arg = 9 + 4 * segments + (banded ? 2 : 0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(runs_arg), runs = get_arg_val<uint32_t>(runs_arg + 1);
    auto run_arg = [&](uint32_t r, uint32_t i) { return get_arg_val<uint32_t>(runs_arg + 2 + 3 * r + i); };
    uint32_t total = 0;
    for (uint32_t r = 0; r < runs; r++) {
        total += run_arg(r, 2) - run_arg(r, 1);
    }
    const auto weights = TensorAccessor(weight_args, get_arg_val<uint32_t>(runs_arg + 2 + 3 * runs));
    const uint32_t ring_base = CircularBuffer(tt::CBIndex::c_0).get_write_ptr();  // never moved by this kernel
    uint32_t run = 0, group = runs > 0 ? run_arg(0, 1) : 0, passed = 0, taken = 0, landed = 0;
    bool reading = false;
    auto next_record = [&]() {
        const uint32_t id = group * n_tiles + run_arg(run, 0);
        if (++group == run_arg(run, 2) && ++run < runs) {
            group = run_arg(run, 1);
        }
        passed++;
        return id;
    };
    auto pump = [&]() {
        if (reading) {
            if (!noc_cmd_buf_ready(noc.get_noc_id(), read_cmd_buf) || !noc.is_read_trid_flushed(weight_trid)) {
                return;
            }
            reading = false;
            answers.set(1 + ++landed);
        }
        if (requests.value() > taken) {
            taken++;
            const uint32_t r0 = address.value();
            const uint32_t n = total - r0 < unit ? total - r0 : unit;
            while (passed < r0) {
                next_record();
            }
            for (uint32_t j = 0; j < n; j++) {
                noc.async_read<NocOptions::TXN_ID>(
                    weights,
                    CoreLocalMem<uint32_t>(ring_base + (r0 + j) % ring * page),
                    record,
                    {.page_id = next_record()},
                    {},
                    {.trid = weight_trid});
            }
            reading = true;
        }
    };
#endif
    auto serve = [&]() {
        if constexpr (split) {
            if (requests.value() > served) {
                const uint32_t first = address.value();
                invalidate_l1_cache();
                for (uint32_t p = 0; p < P; p++) {
                    for (uint32_t q = GGUF_READER_PARTS; q < GGUF_READER_PARTS + GGUF_WRITER_PARTS; q++) {
                        lookup_grid<FMT>(
                            reinterpret_cast<const uint32_t*>(first + p * page + index_offset<FMT>()),
                            table,
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(first + p * page),
                            q,
                            GGUF_LOOKUP_PARTS);
                    }
                }
                asm volatile("fence" ::: "memory");
                answers.set(++served);
                if (served == 1) {
                    // The local copy once the first request is answered, while the reader works on the next.
                    table = local_level_table<FMT>(table_copy);
                }
            }
        }
#if GGUF_STREAM
        pump();
#endif
    };

#if GGUF_STREAM
    if (!collector) {
        answers.set(1);
    }
#endif
    in_cb.reserve_back(k_tiles * in_pages);
    const uint32_t base = in_cb.get_write_ptr();
    // Collector: chunk c + 1 is read while chunk c is multicast.
    auto read_chunk = [&](uint32_t k0) {
        for (uint32_t kt = k0; kt < k0 + chunk && kt < k_tiles; kt++) {
            noc.async_read(
                input, CoreLocalMem<uint32_t>(base + kt * tile_bytes), tile_bytes, {.page_id = k_first + kt}, {});
        }
    };
    if (collector) {
        read_chunk(0);
    }
    for (uint32_t k0 = 0, c = 1; k0 < k_tiles; k0 += chunk, c++) {
        const uint32_t n = k_tiles - k0 < chunk ? k_tiles - k0 : chunk;
        if (collector) {
            noc.async_read_barrier();
            if constexpr (zero_rows) {
                // Rows 1..15 of faces 0 and 1, faces 2 and 3: the compute kernel's in0 tiles overlap them.
                noc.async_writes_flushed();
                for (uint32_t kt = k0; kt < k0 + n; kt++) {
                    const uint32_t tile = base + kt * tile_bytes;
                    noc.async_write_zeros(CoreLocalMem<uint32_t>(tile + 32), 480);
                    noc.async_write_zeros(CoreLocalMem<uint32_t>(tile + 544), tile_bytes - 544);
                }
                noc.write_zeros_l1_barrier();
            }
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
        in_cb.push_back(n * in_pages);
    }
#if GGUF_STREAM
    if (collector) {
        answers.set(1);  // its input reads are over
    }
#endif

    // K bands: a core may finish several N tiles, and partials of a later finish may land first, so the first finish
    // waits for all of them.
    uint32_t all_parts = 0;
    if constexpr (banded) {
        for (uint32_t s = 0; s < segments; s++) {
            if (segment_arg(s, 1) != tail) {
                all_parts += segment_arg(s, 2);
            }
        }
    }
    for (uint32_t s = 0; s < segments; s++) {
        const uint32_t n = segment_arg(s, 0);
        const uint32_t role = segment_arg(s, 1);
        if (role == tail) {
            const uint32_t x = segment_arg(s, 2), y = segment_arg(s, 3);
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
                const uint32_t parts = segment_arg(s, 2);
                while (partials.value() < (banded ? all_parts : parts)) {
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
#if GGUF_STREAM
    noc_async_read_set_trid(0, noc.get_noc_id());
#endif
}
