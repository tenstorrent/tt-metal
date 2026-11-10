// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// engine_dram_reader: a latent engine core's DRAM reader - W', own rows' shares, shared expert tiles - and the gates
// Runs on: latent engine cores (plan.engine_cores, engine_columns x 2 num_pairs), RISCV_0 (BRISC), NoC 0
// Responsibilities:
//  - reads from DRAM over NoC 0: its W' K range once a call, its share of this chip's own rows (the combine output,
//    row-major) and its shared expert tiles every row block; pushes them to engine_norm_projection_compute's CBs
//  - multicasts W' down its half column (NoC 0) with a W_LOADED_SEM bit
//  - zeroes this call's step words and counts in on the lead (BLOCK_GATE_COUNT_SEM); the lead core runs the gate a
//    row block (engine_common.hpp): releases the writers (BLOCK_GATE_GO_SEM multicast) and the gather ports
//    (PORT_READER_GO_SEM / PORT_WRITER_GO_SEM increments)

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_tail/device/kernels/latent_engine/engine_common.hpp"
#include "api/debug/kernel_profiler.hpp"

constexpr auto own_rows_args =
    TensorAccessorArgs<ENGINE_ACCESSOR_ARGS>();  // (the combine's output: this chip's owned rows, row-major)
constexpr auto shared_expert_args = TensorAccessorArgs<own_rows_args.next_compile_time_args_offset()>();
// row blocks: their count and, a block, the own rows' / shared expert's rows that much further in every bank (bytes)
constexpr uint32_t row_blocks = get_compile_time_arg_val(shared_expert_args.next_compile_time_args_offset());
constexpr uint32_t rows_block_stride_bytes =
    get_compile_time_arg_val(shared_expert_args.next_compile_time_args_offset() + 1);
constexpr uint32_t shared_expert_block_stride_bytes =
    get_compile_time_arg_val(shared_expert_args.next_compile_time_args_offset() + 2);
// (a share's tiles at most: the CB_OWN_ROWS block pushed for every own step)
constexpr uint32_t max_share_tiles = get_compile_time_arg_val(shared_expert_args.next_compile_time_args_offset() + 3);

void kernel_main() {
    moe_block_args::load();  // (my args: this chip's block of the DRAM table, common/args_table.hpp)
    constexpr uint32_t tile_bytes = moe_block_tiles::kBf16TileBytes;
    constexpr uint32_t tile_row_bytes = tile_bytes / tt::constants::TILE_HEIGHT;  // (a tile column's row-major row)
    size_t arg_index = 0;
    const uint32_t pair = moe_block_args::arg(arg_index++), column_tiles = moe_block_args::arg(arg_index++);
    const uint32_t k_half_first = moe_block_args::arg(arg_index++);
    const uint32_t w_address = moe_block_args::arg(arg_index++), w_bank = moe_block_args::arg(arg_index++);
    const uint32_t w_offset = moe_block_args::arg(arg_index++);
    const uint32_t w_k_first = moe_block_args::arg(arg_index++), w_k_end = moe_block_args::arg(arg_index++);
    // (my half column's NoC x, top and bottom NoC y: the W' multicast)
    const uint32_t my_noc_x = moe_block_args::arg(arg_index++), half_column_top = moe_block_args::arg(arg_index++);
    const uint32_t half_column_bottom = moe_block_args::arg(arg_index++);
    const uint32_t own_rows_address = moe_block_args::arg(arg_index++),
                   share_k_first = moe_block_args::arg(arg_index++);
    const uint32_t share_k_end = moe_block_args::arg(arg_index++);
    const uint32_t shared_expert_address = moe_block_args::arg(arg_index++),
                   output_tiles = moe_block_args::arg(arg_index++);
    const uint32_t finished_first = moe_block_args::arg(arg_index++);
    const uint32_t finished_count = moe_block_args::arg(arg_index++), steps = moe_block_args::arg(arg_index++);
    const uint32_t own_steps = moe_block_args::arg(arg_index++);
    // (the gates: the lead core (its NoC x | y << 16), the engine's NoC rectangle, its cores, am I the lead)
    const uint32_t lead_xy = moe_block_args::arg(arg_index++), engine_start_x = moe_block_args::arg(arg_index++);
    const uint32_t engine_start_y = moe_block_args::arg(arg_index++);
    const uint32_t engine_end_x = moe_block_args::arg(arg_index++), engine_end_y = moe_block_args::arg(arg_index++);
    const uint32_t engine_cores = moe_block_args::arg(arg_index++);
    const bool is_lead = moe_block_args::arg(arg_index++) != 0;
    // (the lead: the port cores it releases, x | y << 16)
    const uint32_t num_port_cores = moe_block_args::arg(arg_index++), port_cores = arg_index;
    const uint32_t chunk_order = port_cores + num_port_cores;  // (chunk position -> its owner's TP position)
    volatile tt_l1_ptr uint32_t* step_words =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_STEP_WORDS));
    volatile tt_l1_ptr uint32_t* block_go_source = step_words + step_word_offset(STEP_BLOCK_GO, 0) / sizeof(uint32_t);
    // the lead's gate a block: wait for `expected` count-ins on its BLOCK_GATE_COUNT_SEM, then release the writers
    // (BLOCK_GATE_GO_SEM) and the ports. Loop 0: every engine core zeroed its step words; block b > 0: every engine
    // writer and every port reader is done with block b - 1 (they count in at their block's end).
    auto release = [&](uint32_t expected) {
        volatile tt_l1_ptr uint32_t* count = semaphore(BLOCK_GATE_COUNT_SEM);
        noc_semaphore_wait(count, expected);
        *count = 0;  // (re-arm: every count-in of this gate landed; the next gate's come only after this release)
        // (the go's source: the step words' spare slot, not my BLOCK_GATE_GO_SEM - my writer re-arms that one,
        // possibly before the multicast has read it)
        *block_go_source = 1;
        noc_semaphore_set_multicast_loopback_src(
            reinterpret_cast<uint32_t>(block_go_source),
            get_noc_multicast_addr(
                engine_start_x, engine_start_y, engine_end_x, engine_end_y, get_semaphore(BLOCK_GATE_GO_SEM)),
            engine_cores);
        for (uint32_t i = 0; i < num_port_cores; ++i) {
            const uint32_t port_xy = moe_block_args::arg(port_cores + i);
            noc_semaphore_inc(get_noc_addr(port_xy & 0xFFFF, port_xy >> 16, get_semaphore(PORT_READER_GO_SEM)), 1);
            noc_semaphore_inc(get_noc_addr(port_xy & 0xFFFF, port_xy >> 16, get_semaphore(PORT_WRITER_GO_SEM)), 1);
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    };
    {  // this call's step words zeroed here (once: the blocks count on from them), then counted in on the lead; the
       // lead releases the writers and the ports
        DeviceZoneScopedN("TL-ZGATE");
        for (uint32_t i = 0; i < STEP_WORDS_BYTES / sizeof(uint32_t); ++i) {
            step_words[i] = 0;
        }
        noc_semaphore_inc(get_noc_addr(lead_xy & 0xFFFF, lead_xy >> 16, get_semaphore(BLOCK_GATE_COUNT_SEM)), 1);
        if (is_lead) {
            release(engine_cores);
        }
    }
    cb_reserve_back(CB_ONES, 1);  // (the all-ones tile of the squares' row sums: once, never popped)
    {
        // (two bfloat16 1.0 values a word: 0x3F80 each)
        constexpr uint32_t two_bf16_ones = 0x3F803F80u;
        volatile tt_l1_ptr uint32_t* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_ONES));
        for (uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
            ones[i] = two_bf16_ones;
        }
    }
    cb_push_back(CB_ONES, 1);
    const uint32_t share_tiles = share_k_end - share_k_first;
    // my share of my own rows of a block, row-major (the combine's output: the TILE_HEIGHT rows of row tile t, K tiles
    // k_half_first + share_k_first ..), own step by own step, a whole max_share_tiles block each (the CB's pointer
    // then wraps on whole blocks, block after block): the compute tilizes each into its landing slot (the other steps'
    // shares land by the fabric)
    auto read_own_share = [&](const auto& own_rows, uint32_t own_step) {
        const uint32_t row_tile = step_row_tile(own_step, pair);
        cb_reserve_back(CB_OWN_ROWS, max_share_tiles);
        const uint32_t destination = get_write_ptr(CB_OWN_ROWS);
        for (uint32_t i = 0; i < tt::constants::TILE_HEIGHT; ++i) {
            noc_async_read(
                own_rows.get_noc_addr(
                    row_tile * tt::constants::TILE_HEIGHT + i, (k_half_first + share_k_first) * tile_row_bytes),
                destination + i * share_tiles * tile_row_bytes,
                share_tiles * tile_row_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(CB_OWN_ROWS, max_share_tiles);
    };
    for (uint32_t block = 0; block < row_blocks; ++block) {
        // (a page: one latent row)
        const auto own_rows = TensorAccessor(
            own_rows_args, own_rows_address + block * rows_block_stride_bytes, latent_tiles * tile_row_bytes);
        const auto shared_expert = TensorAccessor(
            shared_expert_args, shared_expert_address + block * shared_expert_block_stride_bytes, tile_bytes);
        if (block > 0 &&
            is_lead) {  // (block b's gate: every engine writer and every port reader done with block b - 1)
            DeviceZoneScopedN("TL-LGATE");
            release(engine_cores + num_port_cores);
        }
        if (own_steps > 0) {
            DeviceZoneScopedN("TL-RMREAD");
            read_own_share(own_rows, 0);  // (the writer multicasts it while W' loads)
        }
        if (block == 0) {  // W' once a call: resident in CB_UP_WEIGHTS for every block
            {  // every engine core's first own rows read before any W' load: the rows' small strided reads queued
               // behind the W' bursts in the DRAM took 13-26 us, holding the gather's first sends (the ports read the
               // tilized rows)
                DeviceZoneScopedN("TL-WGATE");
                noc_semaphore_inc(get_noc_addr(lead_xy & 0xFFFF, lead_xy >> 16, get_semaphore(W_GATE_COUNT_SEM)), 1);
                volatile tt_l1_ptr uint32_t* go = semaphore(W_GATE_GO_SEM);
                if (is_lead) {
                    volatile tt_l1_ptr uint32_t* count = semaphore(W_GATE_COUNT_SEM);
                    noc_semaphore_wait(count, engine_cores);
                    *count = 0;  // (re-arm: every core's one increment landed)
                    *go = 1;
                    noc_semaphore_set_multicast_loopback_src(
                        get_semaphore(W_GATE_GO_SEM),
                        get_noc_multicast_addr(
                            engine_start_x, engine_start_y, engine_end_x, engine_end_y, get_semaphore(W_GATE_GO_SEM)),
                        engine_cores);
                    noc_async_write_barrier();
                }
                while (*go == 0) {
                    invalidate_l1_cache();
                }
                *go = 0;  // (re-arm: one set a call)
            }
            {  // W': my K range of my half column's [K half][column_tiles] (one contiguous bank region: one burst),
               // then down the half column
                DeviceZoneScopedN("TL-WLOAD");
                // (in K order a wave at a time: my half column's range pair - 1 in first - the chip's DRAM serves the
                // ranges the compute takes next, not all of them at once)
                if (pair > 0) {
                    volatile tt_l1_ptr uint32_t* w_loaded = semaphore(W_LOADED_SEM);
                    while (!((*w_loaded >> (pair - 1)) & 1)) {
                        invalidate_l1_cache();
                    }
                }
                const uint32_t source = get_write_ptr(CB_UP_WEIGHTS) + w_k_first * column_tiles * w_tile_bytes;
                const uint32_t bytes = (w_k_end - w_k_first) * column_tiles * w_tile_bytes;
                noc_async_read(get_noc_addr_from_bank_id<true>(w_bank, w_address + w_offset), source, bytes);
                noc_async_read_barrier();
                if (num_pairs > 1) {
                    if (bytes) {
                        noc_async_write_multicast(
                            source,
                            get_noc_multicast_addr(my_noc_x, half_column_top, my_noc_x, half_column_bottom, source),
                            bytes,
                            num_pairs - 1);
                    }
                    noc_async_writes_flushed();
                    noc_semaphore_inc_multicast(
                        get_noc_multicast_addr(
                            my_noc_x, half_column_top, my_noc_x, half_column_bottom, get_semaphore(W_LOADED_SEM)),
                        1u << pair,
                        num_pairs - 1);
                }
                noc_semaphore_inc(get_noc_addr(get_semaphore(W_LOADED_SEM)), 1u << pair);
            }
        }
        for (uint32_t own_step = 1; own_step < own_steps; ++own_step) {
            read_own_share(own_rows, own_step);
        }
        for (uint32_t step = 0; step < steps; ++step) {  // my shared expert tiles (the columns I finish), step by step
            const uint32_t row_tile = step_row_tile(step, pair);
            const uint32_t output_row_tile =
                moe_block_args::arg(chunk_order + row_tile / row_tiles_per_chunk) * row_tiles_per_chunk +
                row_tile % row_tiles_per_chunk;
            cb_reserve_back(CB_SHARED_EXPERT, finished_count);
            const uint32_t destination = get_write_ptr(CB_SHARED_EXPERT);
            for (uint32_t j = 0; j < finished_count; ++j) {
                noc_async_read(
                    shared_expert.get_noc_addr(output_row_tile * output_tiles + finished_first + j),
                    destination + j * tile_bytes,
                    tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(CB_SHARED_EXPERT, finished_count);
        }
    }
    noc_async_atomic_barrier();
}
