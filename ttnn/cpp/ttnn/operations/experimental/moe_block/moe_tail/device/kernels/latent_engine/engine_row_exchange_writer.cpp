// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// engine_row_exchange_writer: a latent engine core's writer - shares along the row, partner exchange, output writes
// Runs on: latent engine cores (plan.engine_cores, engine_columns x 2 num_pairs), RISCV_1 (NCRISC), NoC 1
// Responsibilities:
//  - multicasts my landed / tilized share of each step along my row (NoC 1) into the row's ring slots + its share word;
//    unicasts my squares tile to every core of my row; sends the partner half its partials + its sum of squares
//    (CB_EXCHANGE_RECEIVE / PARTNER_EXCHANGE_SEM)
//  - writes the finished output tiles from the compute to the output (DRAM) over NoC 1
//  - watcher: counts a landed step for the other cores of its column group; frees ring / squares slots
//    (SLOTS_FREE_SEM); feeds the compute its CBs; waits the gate's go (BLOCK_GATE_GO_SEM) each row block and counts in
//    on the lead (BLOCK_GATE_COUNT_SEM) after each block but the last
// Note: the landing, ring, squares and exchange slots are CBs at one address on every engine core of every chip; their
//       bases are read before this kernel pushes any of them

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_tail/device/kernels/latent_engine/engine_common.hpp"
#include "api/debug/kernel_profiler.hpp"

constexpr auto output_args = TensorAccessorArgs<ENGINE_ACCESSOR_ARGS>();
// row blocks: their count and, a block, the output's rows that much further in every bank (bytes)
constexpr uint32_t row_blocks = get_compile_time_arg_val(output_args.next_compile_time_args_offset());
constexpr uint32_t output_block_stride_bytes =
    get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 1);

void kernel_main() {
    moe_block_args::load();  // (my args: this chip's block of the DRAM table, common/args_table.hpp)
    constexpr uint32_t tile_bytes = moe_block_tiles::kBf16TileBytes;       // (landing, ring, output: bfloat16)
    constexpr uint32_t fp32_tile_bytes = moe_block_tiles::kFp32TileBytes;  // (squares, exchange: fp32)
    const uint32_t landing = get_write_ptr(CB_LANDING), ring = get_write_ptr(CB_RING);
    const uint32_t row_squares = get_write_ptr(CB_ROW_SQUARES), exchange_receive = get_write_ptr(CB_EXCHANGE_RECEIVE);
    size_t arg_index = 0;
    const uint32_t pair = moe_block_args::arg(arg_index++), column_tiles = moe_block_args::arg(arg_index++);
    const uint32_t finished_first = moe_block_args::arg(arg_index++);
    const uint32_t finished_count = moe_block_args::arg(arg_index++), share_k_first = moe_block_args::arg(arg_index++);
    const uint32_t share_k_end = moe_block_args::arg(arg_index++);
    // (my row's left and right end NoC x, my column)
    const uint32_t row_left_x = moe_block_args::arg(arg_index++), row_right_x = moe_block_args::arg(arg_index++);
    const uint32_t my_column = moe_block_args::arg(arg_index++);
    const uint32_t output_address = moe_block_args::arg(arg_index++), output_tiles = moe_block_args::arg(arg_index++);
    const uint32_t max_share_tiles = moe_block_args::arg(arg_index++);
    // (my NoC x / y, my partner's NoC y)
    const uint32_t my_x = moe_block_args::arg(arg_index++), my_y = moe_block_args::arg(arg_index++);
    const uint32_t partner_y = moe_block_args::arg(arg_index++);
    const uint32_t words = get_write_ptr(CB_STEP_WORDS), steps = moe_block_args::arg(arg_index++);
    const uint32_t own_steps = moe_block_args::arg(arg_index++);
    // (chunk order; my row's NoC x by column; W' K range bounds)
    const uint32_t chunk_order = arg_index, row_xs = chunk_order + num_chunks, w_k_bounds = row_xs + engine_columns;
    // (the steps I watch: a landed row tile's last share is mine - its port's one fused increment, behind the row's
    // other shares on that link - so I count it landed for the other cores of its group: step | first column << 8 | end
    // column << 16)
    const uint32_t num_watched = moe_block_args::arg(w_k_bounds + num_pairs + 1);
    const uint32_t watched = w_k_bounds + num_pairs + 2;
    const uint32_t lead_xy = moe_block_args::arg(watched + max_steps);  // (the gate's lead core: x | y << 16)
    auto word = [&](uint32_t kind, uint32_t step) { return words + step_word_offset(kind, step); };
    auto word_value = [&](uint32_t kind, uint32_t step) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word(kind, step));
    };
    // (the row's shares of `step` in: a word a column, set to the block's count by its share's multicast right behind
    // the data - same NoC, same VC, so in order: no write barrier between, the chain's next core goes on seeing it)
    auto share_word = [&](uint32_t step, uint32_t column) { return words + share_word_offset(step, column); };
    auto share_word_value = [&](uint32_t step, uint32_t column) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(share_word(step, column));
    };
    // (NOC1: the multicast rectangle's right end first)
    auto row = [&](uint32_t address) { return get_noc_multicast_addr(row_right_x, my_y, row_left_x, my_y, address); };
    const uint32_t share_tiles = share_k_end - share_k_first;
    uint32_t w_ranges_pushed = 0;  // (W' K ranges pushed to the compute: once a call - resident for every block)
    for (uint32_t block = 0; block < row_blocks; ++block) {
        const uint32_t block_first_step = block * steps;  // (global: the slots and the cumulative semaphores)
        // (block 0: this call's words zeroed on every engine core; block b > 0: every engine writer and port reader
        // done with block b - 1 - the gate's go, TL-ZGATE / TL-LGATE)
        {
            volatile tt_l1_ptr uint32_t* go = semaphore(BLOCK_GATE_GO_SEM);
            while (*go == 0) {
                invalidate_l1_cache();
            }
            *go = 0;  // (re-arm: one set a gate)
        }
        // (block b - 1's landing slots read by everyone: freed whole - the compute's pointer back at the base)
        if (block > 0) {
            cb_wait_front(CB_LANDING, max_steps * max_share_tiles);
            cb_pop_front(CB_LANDING, max_steps * max_share_tiles);
        }
        const uint32_t expected = block + 1;     // (block b's words: a step's word reaches (l + 1) x its count a block)
        *word_value(STEP_SHARES, 0) = expected;  // (the share words' source)
        const auto output = TensorAccessor(output_args, output_address + block * output_block_stride_bytes, tile_bytes);
        uint32_t next_share = 0, owned = 0, watched_done = 0;
        auto pump = [&]() {
            invalidate_l1_cache();
            for (uint32_t i = 0; i < num_watched;
                 ++i) {  // (any order: the directions land their row tiles independently)
                const uint32_t entry = moe_block_args::arg(watched + i), step = entry & 0xFF;
                if ((watched_done >> i & 1) || *word_value(STEP_LANDED, step) < expected) {
                    continue;
                }
                for (uint32_t column = entry >> 8 & 0xFF; column < (entry >> 16); ++column) {
                    if (column != my_column) {
                        noc_semaphore_inc(
                            get_noc_addr(moe_block_args::arg(row_xs + column), my_y, word(STEP_LANDED, step)), 1);
                    }
                }
                watched_done |= 1u << i;
            }
            // (my own share tilized)
            while (owned < own_steps && cb_pages_available_at_front(CB_LANDING, max_share_tiles * (owned + 1))) {
                noc_semaphore_inc(get_noc_addr(word(STEP_LANDED, owned)), 1);
                ++owned;
            }
            // (W' K range w_ranges_pushed landed: to the compute, in K order)
            while (w_ranges_pushed < num_pairs && (*semaphore(W_LOADED_SEM) >> w_ranges_pushed & 1)) {
                const uint32_t tiles = (moe_block_args::arg(w_k_bounds + w_ranges_pushed + 1) -
                                        moe_block_args::arg(w_k_bounds + w_ranges_pushed)) *
                                       column_tiles;
                cb_reserve_back(CB_UP_WEIGHTS, tiles);
                cb_push_back(CB_UP_WEIGHTS, tiles);
                ++w_ranges_pushed;
            }
            // my share of step next_share: landed, its ring slot free on the whole row, my turn in the row (column
            // order)
            const uint32_t global_step = block_first_step + next_share;
            if (next_share >= steps || *word_value(STEP_LANDED, next_share) < expected ||
                (global_step >= ring_slots &&
                 *semaphore(SLOTS_FREE_SEM) < engine_columns * (global_step - ring_slots + 1))) {
                return;
            }
            if (my_column > 0 && *share_word_value(next_share, my_column - 1) < expected) {
                return;
            }
            DeviceZoneScopedN("TL-SHARE");
            const uint32_t source = landing + next_share * max_share_tiles * tile_bytes;
            const uint32_t destination =
                ring + ((global_step % ring_slots) * k_half_tiles + share_k_first) * tile_bytes;
            noc_async_write_multicast_loopback_src(source, row(destination), share_tiles * tile_bytes, engine_columns);
            noc_semaphore_set_multicast_loopback_src(
                word(STEP_SHARES, 0), row(share_word(next_share, my_column)), engine_columns);
            ++next_share;
        };
        auto wait = [&](volatile tt_l1_ptr uint32_t* counter, uint32_t target) {
            while (*counter < target) {
                pump();
            }
        };
        auto barrier = [&]() {  // (my writes acked - the chain's next share of mine meanwhile: a hop waits on my pump)
            while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
                pump();
            }
        };
        auto wait_front = [&](uint32_t cb, uint32_t pages) {
            while (!cb_pages_available_at_front(cb, pages)) {
                pump();
            }
            cb_wait_front(cb, pages);
        };
        // my finished columns of `step`; then its ring / squares slots are free here (SLOTS_FREE_SEM)
        auto write_output = [&](uint32_t step) {
            const uint32_t row_tile = step_row_tile(step, pair);
            const uint32_t output_row_tile =
                moe_block_args::arg(chunk_order + row_tile / row_tiles_per_chunk) * row_tiles_per_chunk +
                row_tile % row_tiles_per_chunk;
            wait_front(CB_OUTPUT, finished_count);
            const uint32_t source = get_read_ptr(CB_OUTPUT);
            for (uint32_t j = 0; j < finished_count; ++j) {
                noc_async_write(
                    source + j * tile_bytes,
                    output.get_noc_addr(output_row_tile * output_tiles + finished_first + j),
                    tile_bytes);
            }
            barrier();
            cb_pop_front(CB_OUTPUT, finished_count);
            if (engine_columns > 1) {
                noc_semaphore_inc_multicast(row(get_semaphore(SLOTS_FREE_SEM)), 1, engine_columns - 1);
            }
            noc_semaphore_inc(get_noc_addr(get_semaphore(SLOTS_FREE_SEM)), 1);
        };
        for (uint32_t step = 0; step < steps; ++step) {
            const uint32_t global_step = block_first_step + step;
            {
                DeviceZoneScopedN("TL-AWAIT");
                for (uint32_t column = 0; column < engine_columns; ++column) {
                    wait(share_word_value(step, column), expected);
                }
            }
            cb_reserve_back(CB_RING, k_half_tiles);
            cb_push_back(CB_RING, k_half_tiles);
            {  // my squares tile to every core of my row (unicasts: one fp32 tile - no multicast path to reserve), once
               // its slot is free
                wait_front(CB_MY_SQUARES, 1);
                if (global_step >= ring_slots) {
                    wait(semaphore(SLOTS_FREE_SEM), engine_columns * (global_step - ring_slots + 1));
                }
                const uint32_t source = get_read_ptr(CB_MY_SQUARES);
                const uint32_t destination =
                    row_squares + ((global_step % ring_slots) * engine_columns + my_column) * fp32_tile_bytes;
                for (uint32_t i = 0; i < engine_columns; ++i) {
                    noc_async_write(
                        source, get_noc_addr(moe_block_args::arg(row_xs + i), my_y, destination), fp32_tile_bytes);
                }
                barrier();
                for (uint32_t i = 0; i < engine_columns; ++i) {
                    noc_semaphore_inc(get_noc_addr(moe_block_args::arg(row_xs + i), my_y, word(STEP_SQUARES, step)), 1);
                }
                cb_pop_front(CB_MY_SQUARES, 1);
            }
            {
                DeviceZoneScopedN("TL-PWAIT");
                wait(word_value(STEP_SQUARES, step), expected * engine_columns);
            }
            cb_reserve_back(CB_ROW_SQUARES, engine_columns);
            cb_push_back(CB_ROW_SQUARES, engine_columns);
            {  // the partner's columns' partials + my half's sum of squares to its CB_EXCHANGE_RECEIVE slot of the
               // step (a column unicast; the slot g % max_steps: the partner's block b - 1 slots all consumed - its
               // writer counted in at the gate only after its compute's last output)
                wait_front(CB_EXCHANGE_SEND, exchange_tiles);
                noc_async_write(
                    get_read_ptr(CB_EXCHANGE_SEND),
                    get_noc_addr(
                        my_x,
                        partner_y,
                        exchange_receive + (global_step % max_steps) * exchange_tiles * fp32_tile_bytes),
                    exchange_tiles * fp32_tile_bytes);
                barrier();
                noc_semaphore_inc(get_noc_addr(my_x, partner_y, get_semaphore(PARTNER_EXCHANGE_SEM)), 1);
                cb_pop_front(CB_EXCHANGE_SEND, exchange_tiles);
            }
            {
                DeviceZoneScopedN("TL-XWAIT");
                wait(semaphore(PARTNER_EXCHANGE_SEM), global_step + 1);
            }
            cb_reserve_back(CB_EXCHANGE_RECEIVE, exchange_tiles);
            cb_push_back(CB_EXCHANGE_RECEIVE, exchange_tiles);
            if (step > 0) {
                write_output(step - 1);
            }
        }
        if (steps > 0) {
            write_output(steps - 1);
        }
        while (w_ranges_pushed < num_pairs || next_share < steps || owned < own_steps ||
               watched_done != (num_watched ? (2u << (num_watched - 1)) - 1 : 0u)) {
            pump();
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        if (block + 1 < row_blocks) {  // (done with block b - its landing slots, words, ring / squares slots: count in)
            noc_semaphore_inc(get_noc_addr(lead_xy & 0xFFFF, lead_xy >> 16, get_semaphore(BLOCK_GATE_COUNT_SEM)), 1);
            noc_async_atomic_barrier();
        }
    }
}
