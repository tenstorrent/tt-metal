// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/dataflow/packed_summary.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/persistent_counter.hpp"

constexpr uint32_t cb_e = get_compile_time_arg_val(0);
constexpr uint32_t cb_b = get_compile_time_arg_val(1);
constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
constexpr uint32_t cb_chron = get_compile_time_arg_val(3);
constexpr uint32_t sp_size = get_compile_time_arg_val(4);
constexpr uint32_t local_rows = get_compile_time_arg_val(5);
constexpr uint32_t kt = get_compile_time_arg_val(6);  // key tiles; the state is kt x kt tiles
// cb_ein's storage again, as compute's exact (UnpackToDest) view of the entry state.
constexpr uint32_t cb_ein_exact = get_compile_time_arg_val(7);
// One BF16 32x32 identity tile, for compute's exact I @ B.
constexpr uint32_t cb_identity = get_compile_time_arg_val(8);
// The final carry: the final transition's E and B, the relayed final state P (the landing buffer's final slot) in its
// matmul and exact views.
constexpr uint32_t cb_fe = get_compile_time_arg_val(9);
constexpr uint32_t cb_fb = get_compile_time_arg_val(10);
constexpr uint32_t cb_p = get_compile_time_arg_val(11);
constexpr uint32_t cb_p_exact = get_compile_time_arg_val(12);
// Compute's flags and chunk counts.
constexpr uint32_t cb_ctl = get_compile_time_arg_val(13);
constexpr bool has_actual_end = get_compile_time_arg_val(14);
// The entry correction: the seeds its chunks start from (the entry state, and the relayed final state for a separated
// tail), and the chunk terms F(0) and F(I).
constexpr uint32_t cb_seed = get_compile_time_arg_val(15);
constexpr uint32_t cb_tail = get_compile_time_arg_val(16);
constexpr uint32_t cb_terms = get_compile_time_arg_val(17);
constexpr uint32_t core_chunks = get_compile_time_arg_val(18);
// Interleaved terms, streamed a chunk row of F(0) and F(I) at a time; otherwise they are in place in the core's shard.
constexpr bool streamed = get_compile_time_arg_val(19);
constexpr uint32_t landed_semaphore_id = get_compile_time_arg_val(20);
// The cores are heads x cores_per_head consecutive cores of the compute grid in row-major order, from first_core,
// head by head.
constexpr uint32_t cores_per_head = get_compile_time_arg_val(21);
constexpr uint32_t grid_x = get_compile_time_arg_val(22);
constexpr uint32_t first_core = get_compile_time_arg_val(23);
// Slab (slab_kernel_args): the initial state is entry `index * slab_scale + slab_offset`; zeros at actual_start == 0.
constexpr bool slab = get_compile_time_arg_val(24);
constexpr uint32_t slab_index_limit = get_compile_time_arg_val(25);
constexpr uint32_t slab_scale = get_compile_time_arg_val(26);
constexpr uint32_t slab_offset = get_compile_time_arg_val(27);
constexpr uint32_t entry_tiles = get_compile_time_arg_val(28);
// cb_chron's second DRAM-aligned slot, where the slab index lands (the writer reads it there too).
constexpr uint32_t slab_index_word = get_compile_time_arg_val(29);
// The head core's count of the sender's releases: 1 once the relay's last column has left, 2 once the carry's has.
constexpr uint32_t ack_semaphore_id = get_compile_time_arg_val(30);
constexpr auto packed_args = TensorAccessorArgs<31>();
constexpr auto init_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
constexpr auto start_args = TensorAccessorArgs<init_args.next_compile_time_args_offset()>();
constexpr auto fe_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();
constexpr auto fb_args = TensorAccessorArgs<fe_args.next_compile_time_args_offset()>();
constexpr auto end_args = TensorAccessorArgs<fb_args.next_compile_time_args_offset()>();
constexpr auto terms_args = TensorAccessorArgs<end_args.next_compile_time_args_offset()>();
constexpr auto slab_index_args = TensorAccessorArgs<terms_args.next_compile_time_args_offset()>();
constexpr uint32_t state_tiles = kt * kt;
constexpr uint32_t num_chunks = local_rows / tt::constants::TILE_HEIGHT;

// Every core of a head corrects its chunks of the head from the head's seeds; the head's first core is also the
// relay's head core. It reads this rank's complement-form [E | B] rows of the head (E = A - I) and its entry state
// column by column (the initial state on the chronologically first rank, else the predecessor's landed columns), then
// copies the landed entry state to the head's cores as their seed; on the first rank, the relayed final state P as
// well, which the final carry and a separated tail start from.
void kernel_main() {
    // Every core takes the same args (common, per device): its head and share follow from its place in the grid.
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x() - first_core;
    const uint32_t head = core / cores_per_head;
    const uint32_t first_chunk = core % cores_per_head * core_chunks;
    const bool relay_head = core % cores_per_head == 0;
    const uint32_t arrival_semaphore = get_common_arg_val<uint32_t>(8);
    const uint32_t final_semaphore = get_common_arg_val<uint32_t>(9);
    const uint32_t sp_rank = get_common_arg_val<uint32_t>(10);
    // Then every core's NoC x | y << 16, in order: this head's from head * cores_per_head.
    const size_t head_core_args = 11 + head * cores_per_head;

    // The bounds, each into its own page (a DRAM read keeps its source's alignment).
    cb_reserve_back(cb_chron, 1);
    cb_reserve_back(cb_ctl, 1);
    auto* chron = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_chron));
    auto* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl));
    noc_async_read(
        TensorAccessor(start_args, get_common_arg_val<uint32_t>(2)).get_noc_addr(0), get_write_ptr(cb_chron), 4);
    if constexpr (slab) {
        noc_async_read(
            TensorAccessor(slab_index_args, get_common_arg_val<uint32_t>(7)).get_noc_addr(0),
            get_write_ptr(cb_chron) + slab_index_word * sizeof(uint32_t),
            4);
    }
    if constexpr (has_actual_end) {
        noc_async_read(
            TensorAccessor(end_args, get_common_arg_val<uint32_t>(5)).get_noc_addr(0), get_write_ptr(cb_ctl), 4);
    }
    if (relay_head) {
        const auto packed = TensorAccessor(packed_args, get_common_arg_val<uint32_t>(0));
        cb_reserve_back(cb_e, state_tiles);
        cb_reserve_back(cb_b, state_tiles);
        const uint32_t eb_tile = get_tile_size(cb_e);
        // [E | B] rows of this head: kt tile rows of 2 kt tiles.
        for (uint32_t r = 0; r < kt; ++r) {
            for (uint32_t c = 0; c < kt; ++c) {
                const uint32_t page = (head * kt + r) * 2 * kt + c;
                noc_async_read_page(page, packed, get_write_ptr(cb_e) + (r * kt + c) * eb_tile);
                noc_async_read_page(page + kt, packed, get_write_ptr(cb_b) + (r * kt + c) * eb_tile);
            }
        }
        // The BF16 identity tile while the reads are in flight.
        cb_reserve_back(cb_identity, 1);
        auto* identity = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_identity));
        for (uint32_t i = 0; i < tt::constants::TILE_HW; ++i) {
            identity[i] = 0;
        }
        kda::write_identity_diagonal(identity);
        cb_push_back(cb_identity, 1);
    }
    noc_async_read_barrier();
    const uint32_t actual_start = chron[0];
    const uint32_t index = slab ? chron[slab_index_word] : 0;
    const auto topology = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    const auto interval =
        has_actual_end ? kda_chronology::derive_interval(actual_start, ctl[0], sp_rank, sp_size, local_rows) : topology;
    const uint32_t position = (topology.rank + sp_size - topology.first_rank) % sp_size;
    const bool holder = position == 0;
    // This core's valid chunks; those before the reset start from the entry state, the rest from the tail's seed.
    const uint32_t valid = interval.valid_chunks(0, 1);
    const uint32_t reset = interval.reset_chunk(0, 1);
    const uint32_t end_chunk = first_chunk + core_chunks < valid ? first_chunk + core_chunks : valid;
    const uint32_t chunks = end_chunk > first_chunk ? end_chunk - first_chunk : 0;
    uint32_t group_chunks = chunks;
    if (reset != 0) {
        group_chunks = reset <= first_chunk ? 0 : (reset - first_chunk < chunks ? reset - first_chunk : chunks);
    }
    const bool need_tail = valid != 0 && reset != 0;

    // The writer's chronology: this rank's position and this core's chunk counts.
    chron[0] = position;
    chron[1] = chunks;
    chron[2] = group_chunks;
    cb_push_back(cb_chron, 1);
    ctl[0] = holder;
    ctl[1] = interval.split;
    ctl[2] = chunks;
    ctl[3] = group_chunks;
    cb_push_back(cb_ctl, 1);

    // The head's first core copies each landed tile column of a K x K state (column-major by tile) into a seed buffer
    // of every core of the head (at the same address on each), then counts it there (a program semaphore, zero at
    // launch): each core's chunk outputs need only the seed's columns that have landed.
    const uint32_t state_tile = get_tile_size(cb_seed);
    const uint32_t column_bytes = kt * state_tile;
    const uint32_t landed = get_semaphore(landed_semaphore_id);
    // The landing buffer's slots on the head core: the entry state (cb_ein's storage), then the relayed final state.
    const uint32_t landing = relay_head ? get_write_ptr(cb_ein) : 0;
    // Cores [d_begin, d_end) of the head.
    const auto distribute = [&](uint32_t source, uint32_t seed, uint32_t j, uint32_t d_begin, uint32_t d_end) {
        for (uint32_t d = d_begin; d < d_end; ++d) {
            const uint32_t xy = get_common_arg_val<uint32_t>(head_core_args + d);
            noc_async_write(
                source + j * column_bytes, get_noc_addr(xy & 0xFFFF, xy >> 16, seed + j * column_bytes), column_bytes);
        }
        noc_async_write_barrier();
        for (uint32_t d = d_begin; d < d_end; ++d) {
            const uint32_t xy = get_common_arg_val<uint32_t>(head_core_args + d);
            noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, landed), 1);
        }
    };
    // The seeds leave the head core only once the relay is done with it: the copies would otherwise share its NoC with
    // the sender's reads of the next state, and hold up the relay at every hop. The correction has the time: a rank's
    // call ends when the final carry has reached it, well after its own hop.
    volatile tt_l1_ptr uint32_t* released =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ack_semaphore_id));
    const auto distribute_entry = [&]() {
        if (valid != 0) {
            noc_semaphore_wait_min(released, 1);
            for (uint32_t j = 0; j < kt; ++j) {
                distribute(landing, get_write_ptr(cb_seed), j, 0, cores_per_head);
            }
        }
    };
    // A seed's columns, as they land: the entry state's are counts 1..kt, the tail's kt + 1..2 kt.
    const auto receive = [&](uint32_t cb, uint32_t counted) {
        cb_reserve_back(cb, state_tiles);
        for (uint32_t j = 0; j < kt; ++j) {
            kda_counter::wait(landed, counted + j + 1);
            cb_push_back(cb, kt);
        }
    };

    if (relay_head) {
        cb_push_back(cb_e, state_tiles);
        cb_push_back(cb_b, state_tiles);
        // The entry state is stored column-major by tile, so each tile column (all the next state's column needs) can
        // land, be computed and leave on its own.
        cb_reserve_back(cb_ein, state_tiles);
        cb_reserve_back(cb_ein_exact, state_tiles);
        if (holder) {
            if (slab && (actual_start == 0 || index >= slab_index_limit)) {
                // A request's head (or no slab entry): the chunk starts from zeros.
                const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
                for (uint32_t offset = 0; offset < state_tiles * state_tile; offset += MEM_ZEROS_SIZE) {
                    noc_async_read(zeros, landing + offset, MEM_ZEROS_SIZE);
                }
            } else {
                const auto init = TensorAccessor(init_args, get_common_arg_val<uint32_t>(1));
                const uint32_t entry_base = slab ? (index * slab_scale + slab_offset) * entry_tiles : 0;
                for (uint32_t j = 0; j < kt; ++j) {
                    for (uint32_t r = 0; r < kt; ++r) {
                        noc_async_read_page(
                            entry_base + head * state_tiles + r * kt + j, init, landing + (j * kt + r) * state_tile);
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_ein, state_tiles);
            cb_push_back(cb_ein_exact, state_tiles);
            // The first rank applies its final transition to P when the interval ends in a separated tail. P lands only
            // after the whole lap, so E and B are read after the entry state: the relay starts from the first rank, and
            // every rank waits for it.
            if (interval.split) {
                const auto fe = TensorAccessor(fe_args, get_common_arg_val<uint32_t>(3));
                const auto fb = TensorAccessor(fb_args, get_common_arg_val<uint32_t>(4));
                cb_reserve_back(cb_fe, state_tiles);
                cb_reserve_back(cb_fb, state_tiles);
                for (uint32_t i = 0; i < state_tiles; ++i) {
                    noc_async_read_page(head * state_tiles + i, fe, get_write_ptr(cb_fe) + i * state_tile);
                    noc_async_read_page(head * state_tiles + i, fb, get_write_ptr(cb_fb) + i * state_tile);
                }
                noc_async_read_barrier();
                cb_push_back(cb_fe, state_tiles);
                cb_push_back(cb_fb, state_tiles);
            }
            distribute_entry();
        } else {
            // The predecessor's sender writes each tile column of this head's entry state into cb_ein's storage, then
            // counts it.
            for (uint32_t j = 0; j < kt; ++j) {
                kda_counter::wait(arrival_semaphore, j + 1);
                cb_push_back(cb_ein, kt);
                cb_push_back(cb_ein_exact, kt);
            }
            kda_counter::consume(arrival_semaphore, kt);
            distribute_entry();
        }
    }

    // The chunks that start from the entry state. Their terms are in place, or streamed a chunk row at a time: head h's
    // F(0) rows are tile rows h * 2 * num_chunks + chunk of the terms, its F(I) rows num_chunks further.
    const auto stream_terms = [&](uint32_t begin, uint32_t end) {
        const auto terms = TensorAccessor(terms_args, get_common_arg_val<uint32_t>(6));
        const uint32_t tile = get_tile_size(cb_terms);
        for (uint32_t c = begin; c < end; ++c) {
            const uint32_t zero_page = (head * 2 * num_chunks + first_chunk + c) * kt;
            cb_reserve_back(cb_terms, 2 * kt);
            for (uint32_t j = 0; j < kt; ++j) {
                noc_async_read_page(zero_page + j, terms, get_write_ptr(cb_terms) + j * tile);
                noc_async_read_page(zero_page + num_chunks * kt + j, terms, get_write_ptr(cb_terms) + (kt + j) * tile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_terms, 2 * kt);
        }
    };
    if (!streamed && chunks != 0) {
        cb_reserve_back(cb_terms, 2 * core_chunks * kt);
        cb_push_back(cb_terms, 2 * core_chunks * kt);
    }
    if (group_chunks != 0) {
        receive(cb_seed, 0);
    }
    if constexpr (streamed) {
        stream_terms(0, group_chunks);
    }

    if (relay_head && holder) {
        // The last rank's sender writes each tile column of the final state into the landing buffer's final slot, then
        // counts it.
        cb_reserve_back(cb_p, state_tiles);
        cb_reserve_back(cb_p_exact, state_tiles);
        for (uint32_t j = 0; j < kt; ++j) {
            kda_counter::wait(final_semaphore, j + 1);
            cb_push_back(cb_p, kt);
            cb_push_back(cb_p_exact, kt);
        }
        kda_counter::consume(final_semaphore, kt);
        // P seeds the tail's chunks [reset, valid), on the cores that hold them, once the carry formed from P has left.
        if (need_tail) {
            const uint32_t d_begin = reset / core_chunks;
            const uint32_t d_end = (valid + core_chunks - 1) / core_chunks;
            noc_semaphore_wait_min(released, 2);
            for (uint32_t j = 0; j < kt; ++j) {
                distribute(landing + state_tiles * state_tile, get_write_ptr(cb_tail), j, d_begin, d_end);
            }
        }
    }

    // The separated tail's chunks start from P.
    if (chunks != group_chunks) {
        receive(cb_tail, kt);
    }
    if constexpr (streamed) {
        stream_terms(group_chunks, chunks);
    }
    noc_async_atomic_barrier();
}
