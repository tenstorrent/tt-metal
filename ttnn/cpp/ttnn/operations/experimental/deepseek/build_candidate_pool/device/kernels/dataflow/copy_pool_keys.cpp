// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Copies half (compile arg ``half``) of this core's share of the candidate blocks' index keys into
// pool_keys, in candidate order. With WRITE_ROWS it also writes the kv cache row of every pool key
// in the core's share to pool_rows.
//
// key_cache and pool_keys are bfloat16 [blocks, 1, page_block_size, D] in 32x32 tiles. A candidate
// block is candidate_block_size rows starting at a multiple of it, so in each tile along the row it
// is one contiguous run of cbs 32 B face rows in the left face and one in the right face. The run
// sits at the same place in the destination tile pattern, so keys are copied run by run with no
// reformatting.
//
// UNION_ROWS (num_rows > 1 query rows of a verify block, candidate_blocks [num_rows, topk_blocks]):
// the pool is the union of the rows' blocks, ascending, with each block's mask of the rows that chose
// it (pool_masks, per key). The WRITE_ROWS kernel builds it: core 0 multicasts every row's list and
// count; core c merges the rows' blocks in its range of block ids into a local run; core 0 collects the
// run sizes, turns them into offsets and finds each row's pool count (its newest block's position in
// the union plus its keys there), and multicasts them back. Each core then reads the union entries of
// its share of the pool from the cores whose runs hold them and copies them as above; the other kernel
// takes the entries from it through ready.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#ifdef UNION_ROWS
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#endif

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t cbs = get_arg(args::candidate_block_size);
    constexpr uint32_t page_block_size = get_arg(args::page_block_size);
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);
    constexpr uint32_t stage_chunks = get_arg(args::stage_chunks);
    constexpr uint32_t half = get_arg(args::half);
    // Blocks per core are a multiple of this, so every core's pool_rows run starts 64 B aligned.
    constexpr uint32_t align_blocks = get_arg(args::align_blocks);
    // Lookup scratch offsets, all 64 B aligned.
    constexpr uint32_t chunk_offset = get_arg(args::chunk_offset);
    constexpr uint32_t list_offset = get_arg(args::list_offset);
    constexpr uint32_t phys_offset = get_arg(args::phys_offset);
    [[maybe_unused]] constexpr uint32_t rows_offset = get_arg(args::rows_offset);
    [[maybe_unused]] constexpr uint32_t mask_list_offset = get_arg(args::mask_list_offset);
    [[maybe_unused]] constexpr uint32_t mask_rows_offset = get_arg(args::mask_rows_offset);
    [[maybe_unused]] constexpr uint32_t num_rows = get_arg(args::num_rows);
    [[maybe_unused]] constexpr uint32_t topk_blocks = get_arg(args::topk_blocks);

    constexpr uint32_t kFaceRowBytes = 16 * sizeof(uint16_t);
    constexpr uint32_t kFaceBytes = 16 * kFaceRowBytes;
    constexpr uint32_t chunk_bytes = cbs * kFaceRowBytes;
    constexpr uint32_t tiles_per_block = page_block_size / 32 * tiles_per_row;
    constexpr uint32_t chunks_per_block = 2 * tiles_per_row;
    static_assert(stage_chunks % (2 * chunks_per_block) == 0, "the stage must split into two banks of whole blocks");
    constexpr uint32_t bank_blocks = stage_chunks / 2 / chunks_per_block;

    // ---- Runtime args ----
    const uint32_t core_index = get_arg(args::core_index);
    const uint32_t num_cores = get_arg(args::num_cores);

    // ---- Tensors ----
    [[maybe_unused]] const auto candidate_blocks = TensorAccessor(tensor::candidate_blocks);  // [1, topk_blocks] INT32
    [[maybe_unused]] const auto pool_count = TensorAccessor(tensor::pool_count);              // [1] INT32
    [[maybe_unused]] const auto page_table = TensorAccessor(tensor::page_table);              // [1, max_blocks] INT32
    const auto key_cache = TensorAccessor(tensor::key_cache);
    const auto pool_keys = TensorAccessor(tensor::pool_keys);
#ifdef WRITE_ROWS
    const auto pool_rows = TensorAccessor(tensor::pool_rows);  // [1, topk_blocks * cbs] INT32
#endif

    // ---- Dataflow buffers ----
    DataflowBuffer lookup_dfb(dfb::lookup);  // pool_count, a page-table chunk, list slice, physical blocks, rows
    DataflowBuffer stage_dfb(dfb::stage);    // key runs in flight, two banks

    Noc noc;

#ifdef UNION_ROWS
    uint32_t start = 0;
    uint32_t end = 0;
    volatile tt_l1_ptr uint32_t* list = nullptr;
    volatile tt_l1_ptr uint32_t* phys = nullptr;
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* mask_list = nullptr;
    [[maybe_unused]] uint32_t lookup_addr = 0;
#ifdef WRITE_ROWS
    {
        DataflowBuffer lists_dfb(dfb::lists);    // every row's list, then the counts: same address on every core
        DataflowBuffer run_dfb(dfb::run);        // this core's union run: blocks, then their row masks
        DataflowBuffer gather_dfb(dfb::gather);  // core 0: every core's {run size, pinned indices}
        DataflowBuffer result_dfb(dfb::result);  // {pool size, pad, per-row pool counts, per-core offsets}
        DataflowBuffer fetch_dfb(dfb::fetch);    // message staging, then other cores' run entries
        DataflowBuffer ready_dfb(dfb::ready);    // -> the other kernel: {start, end, lookup address}
        Semaphore lists_ready(sem::lists_ready);
        Semaphore gathered(sem::gathered);
        Semaphore result_ready(sem::result_ready);
        const uint32_t coord_x = get_arg(args::coord_x);
        const uint32_t coord_y = get_arg(args::coord_y);
        // NOC_1 multicasts from the bottom-right corner.
        const bool mcast_reversed = noc_index == 1;
        const uint32_t mcast_x_start = mcast_reversed ? get_arg(args::mcast_x_end) : get_arg(args::mcast_x_start);
        const uint32_t mcast_y_start = mcast_reversed ? get_arg(args::mcast_y_end) : get_arg(args::mcast_y_start);
        const uint32_t mcast_x_end = mcast_reversed ? get_arg(args::mcast_x_start) : get_arg(args::mcast_x_end);
        const uint32_t mcast_y_end = mcast_reversed ? get_arg(args::mcast_y_start) : get_arg(args::mcast_y_end);
        const uint32_t num_mcast_dests = num_cores - 1;

        constexpr uint32_t list_stride = (topk_blocks + 15) / 16 * 16;
        constexpr uint32_t counts_word = num_rows * list_stride;
        constexpr uint32_t lists_words = counts_word + 16;
        constexpr uint32_t slot_words = (1 + num_rows + 15) / 16 * 16;
        constexpr uint32_t kResultCountsWord = 16;
        constexpr uint32_t kResultOffsetsWord = 32;
        constexpr uint32_t kNone = 0xFFFFFFFFu;

        lookup_dfb.reserve_back(1);
        lookup_addr = lookup_dfb.get_write_ptr();
        lists_dfb.reserve_back(1);
        run_dfb.reserve_back(1);
        gather_dfb.reserve_back(1);
        result_dfb.reserve_back(1);
        fetch_dfb.reserve_back(1);
        const uint32_t lists_addr = lists_dfb.get_write_ptr();
        const uint32_t run_addr = run_dfb.get_write_ptr();
        const uint32_t gather_addr = gather_dfb.get_write_ptr();
        const uint32_t result_addr = result_dfb.get_write_ptr();
        const uint32_t fetch_addr = fetch_dfb.get_write_ptr();
        volatile tt_l1_ptr uint32_t* lists = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lists_addr);
        volatile tt_l1_ptr uint32_t* run_blocks = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(run_addr);
        volatile tt_l1_ptr uint32_t* run_masks = run_blocks + num_rows * topk_blocks;
        volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);
        volatile tt_l1_ptr uint32_t* fetch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fetch_addr);

        // ---- Every row's list and count, read once and multicast ----
        if (core_index == 0) {
            for (uint32_t r = 0; r < num_rows; ++r) {
                noc.async_read(
                    candidate_blocks,
                    lists_dfb,
                    topk_blocks * sizeof(uint32_t),
                    {.page_id = r},
                    {.offset_bytes = r * list_stride * sizeof(uint32_t)});
            }
            noc.async_read(pool_count, lists_dfb, 64, {.page_id = 0}, {.offset_bytes = counts_word * sizeof(uint32_t)});
            noc.async_read_barrier();
            if (num_mcast_dests > 0) {
                noc.async_write_multicast(
                    UnicastEndpoint{},
                    MulticastEndpoint{},
                    lists_words * sizeof(uint32_t),
                    num_mcast_dests,
                    {.addr = lists_addr},
                    {.noc_x_start = mcast_x_start,
                     .noc_y_start = mcast_y_start,
                     .noc_x_end = mcast_x_end,
                     .noc_y_end = mcast_y_end,
                     .addr = lists_addr});
                noc.async_write_barrier();
                lists_ready.set(1);
                lists_ready.set_multicast(noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_mcast_dests);
                noc.async_write_barrier();
            }
        } else {
            lists_ready.wait(1);
        }
        invalidate_l1_cache();

        // ---- This core's run: the union of the rows' blocks in its range of block ids ----
        uint32_t count[num_rows];
        uint32_t num_chosen[num_rows];
        uint32_t block_end = 0;
        for (uint32_t r = 0; r < num_rows; ++r) {
            count[r] = lists[counts_word + r];
            num_chosen[r] = std::min(tt::data_movement::common::div_up(count[r], cbs), topk_blocks);
            if (num_chosen[r] != 0) {
                block_end = std::max(block_end, lists[r * list_stride + num_chosen[r] - 1] + 1);
            }
        }
        const uint32_t range = tt::data_movement::common::div_up(block_end, num_cores);
        const uint32_t lo = std::min(core_index * range, block_end);
        const uint32_t hi = std::min(lo + range, block_end);
        auto lower_bound = [&](uint32_t r, uint32_t value) {
            uint32_t first = 0;
            uint32_t n = num_chosen[r];
            volatile tt_l1_ptr uint32_t* row_list = lists + r * list_stride;
            while (n > 0) {
                const uint32_t half_n = n / 2;
                if (row_list[first + half_n] < value) {
                    first += half_n + 1;
                    n -= half_n + 1;
                } else {
                    n = half_n;
                }
            }
            return first;
        };
        uint32_t pos[num_rows];
        uint32_t stop[num_rows];
        uint32_t pinned_index[num_rows];
        for (uint32_t r = 0; r < num_rows; ++r) {
            pos[r] = lower_bound(r, lo);
            stop[r] = lower_bound(r, hi);
            pinned_index[r] = kNone;
        }
        uint32_t run_size = 0;
        while (true) {
            uint32_t block = 0xFFFFFFFFu;
            for (uint32_t r = 0; r < num_rows; ++r) {
                if (pos[r] < stop[r]) {
                    block = std::min<uint32_t>(block, static_cast<uint32_t>(lists[r * list_stride + pos[r]]));
                }
            }
            if (block == 0xFFFFFFFFu) {
                break;
            }
            uint32_t mask = 0;
            for (uint32_t r = 0; r < num_rows; ++r) {
                if (pos[r] < stop[r] && lists[r * list_stride + pos[r]] == block) {
                    mask |= 1u << r;
                    if (pos[r] == num_chosen[r] - 1) {
                        pinned_index[r] = run_size;
                    }
                    ++pos[r];
                }
            }
            run_blocks[run_size] = block;
            run_masks[run_size] = mask;
            ++run_size;
        }

        // ---- Offsets: core 0 collects {run size, index of each row's newest block}, multicasts the result ----
        fetch[0] = run_size;
        for (uint32_t r = 0; r < num_rows; ++r) {
            fetch[1 + r] = pinned_index[r];
        }
        (void)ckernel::load_blocking(&fetch[num_rows]);
        // The runs are complete before core 0 hears of them, so other cores may read them once the result is out.
        if (run_size != 0) {
            (void)ckernel::load_blocking(&run_masks[run_size - 1]);
        }
        noc.async_write(
            UnicastEndpoint{},
            UnicastEndpoint{},
            slot_words * sizeof(uint32_t),
            {.addr = fetch_addr},
            {.noc_x = coord_x, .noc_y = coord_y, .addr = gather_addr + core_index * slot_words * sizeof(uint32_t)});
        noc.async_write_barrier();
        gathered.up(noc, coord_x, coord_y, 1);
        noc.async_atomic_barrier();
        if (core_index == 0) {
            gathered.wait(num_cores);
            gathered.set(0);
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* gather = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_addr);
            uint32_t total = 0;
            for (uint32_t c = 0; c < num_cores; ++c) {
                result[kResultOffsetsWord + c] = total;
                total += gather[c * slot_words];
            }
            result[0] = total;
            for (uint32_t r = 0; r < num_rows; ++r) {
                uint32_t pool_keys_of_row = 0;
                if (num_chosen[r] != 0) {
                    for (uint32_t c = 0; c < num_cores; ++c) {
                        const uint32_t index = gather[c * slot_words + 1 + r];
                        if (index != kNone) {
                            // Row r's keys end in its newest block, which holds count - (chosen - 1) * cbs of them.
                            pool_keys_of_row =
                                (result[kResultOffsetsWord + c] + index) * cbs + count[r] - (num_chosen[r] - 1) * cbs;
                            break;
                        }
                    }
                }
                result[kResultCountsWord + r] = pool_keys_of_row;
            }
            const uint32_t result_words = kResultOffsetsWord + num_cores;
            (void)ckernel::load_blocking(&result[result_words - 1]);
            if (num_mcast_dests > 0) {
                noc.async_write_multicast(
                    UnicastEndpoint{},
                    MulticastEndpoint{},
                    (result_words + 3) / 4 * 4 * sizeof(uint32_t),
                    num_mcast_dests,
                    {.addr = result_addr},
                    {.noc_x_start = mcast_x_start,
                     .noc_y_start = mcast_y_start,
                     .noc_x_end = mcast_x_end,
                     .noc_y_end = mcast_y_end,
                     .addr = result_addr});
                noc.async_write_barrier();
                result_ready.set(1);
                result_ready.set_multicast(
                    noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_mcast_dests);
                noc.async_write_barrier();
            }
            const auto pool_counts = TensorAccessor(tensor::pool_counts);  // [num_rows] INT32
            noc.async_write(
                result_dfb,
                pool_counts,
                num_rows * sizeof(uint32_t),
                {.offset_bytes = kResultCountsWord * sizeof(uint32_t)},
                {.page_id = 0, .offset_bytes = 0});
        } else {
            result_ready.wait(1);
            invalidate_l1_cache();
        }

        // ---- This core's share of the pool, read from the runs that hold it ----
        const uint32_t pool_blocks = result[0];
        const uint32_t per_core =
            tt::data_movement::common::div_up(tt::data_movement::common::div_up(pool_blocks, num_cores), align_blocks) *
            align_blocks;
        start = std::min(core_index * per_core, pool_blocks);
        end = std::min(start + per_core, pool_blocks);
        list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lookup_addr + list_offset);
        mask_list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lookup_addr + mask_list_offset);
        phys = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lookup_addr + phys_offset);
        for (uint32_t c = 0; c < num_cores && start < end; ++c) {
            const uint32_t first = result[kResultOffsetsWord + c];
            const uint32_t last = c + 1 < num_cores ? result[kResultOffsetsWord + c + 1] : pool_blocks;
            const uint32_t from = std::max(start, first);
            const uint32_t to = std::min(end, last);
            if (from >= to) {
                continue;
            }
            const uint32_t a = from - first;
            const uint32_t n = to - from;
            volatile tt_l1_ptr uint32_t* src_blocks = run_blocks + a;
            volatile tt_l1_ptr uint32_t* src_masks = run_masks + a;
            if (c != core_index) {
                // 64 B aligned superset of the entries, so source and destination share their alignment.
                const uint32_t a16 = a / 16 * 16;
                const uint32_t words = (a + n - a16 + 15) / 16 * 16;
                const uint32_t remote_x = get_common_vararg(2 * c);
                const uint32_t remote_y = get_common_vararg(2 * c + 1);
                noc.async_read(
                    UnicastEndpoint{},
                    UnicastEndpoint{},
                    words * sizeof(uint32_t),
                    {.noc_x = remote_x, .noc_y = remote_y, .addr = run_addr + a16 * sizeof(uint32_t)},
                    {.addr = fetch_addr});
                noc.async_read(
                    UnicastEndpoint{},
                    UnicastEndpoint{},
                    words * sizeof(uint32_t),
                    {.noc_x = remote_x,
                     .noc_y = remote_y,
                     .addr = run_addr + (num_rows * topk_blocks + a16) * sizeof(uint32_t)},
                    {.addr = fetch_addr + words * sizeof(uint32_t)});
                noc.async_read_barrier();
                invalidate_l1_cache();
                src_blocks = fetch + (a - a16);
                src_masks = fetch + words + (a - a16);
            }
            for (uint32_t i = 0; i < n; ++i) {
                list[from - start + i] = src_blocks[i];
                mask_list[from - start + i] = src_masks[i];
            }
        }

        // Physical page block of each pool block. The list ascends, so each 64 B page-table chunk is read once.
        volatile tt_l1_ptr uint32_t* chunk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lookup_addr + chunk_offset);
#ifdef PAGE_TABLE_ROW
        const auto page_table_row_gen = TensorAccessor(tensor::page_table_row);
        noc.async_read(page_table_row_gen, lookup_dfb, 64, {.page_id = 0}, {.offset_bytes = chunk_offset});
        noc.async_read_barrier();
        invalidate_l1_cache();
        const uint32_t page_table_row = chunk[0];
#else
        const uint32_t page_table_row = 0;
#endif
        uint32_t loaded_chunk = UINT32_MAX;
        for (uint32_t c = start; c < end; ++c) {
            const uint32_t logical_block = list[c - start] * cbs / page_block_size;
            if (logical_block / 16 != loaded_chunk) {
                loaded_chunk = logical_block / 16;
                noc.async_read(
                    page_table,
                    lookup_dfb,
                    64,
                    {.page_id = page_table_row, .offset_bytes = 64 * loaded_chunk},
                    {.offset_bytes = chunk_offset});
                noc.async_read_barrier();
                invalidate_l1_cache();
            }
            phys[c - start] = chunk[logical_block % 16];
        }
        if (end > start) {
            (void)ckernel::load_blocking(&phys[end - start - 1]);
        }

        ready_dfb.reserve_back(1);
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_dfb.get_write_ptr());
        ready[0] = start;
        ready[1] = end;
        ready[2] = lookup_addr;
        (void)ckernel::load_blocking(&ready[2]);
        ready_dfb.push_back(1);
        lists_ready.set(0);
        result_ready.set(0);
        noc.async_write_barrier();
        if (start == end) {
            return;
        }
    }
#else
    {
        // The WRITE_ROWS kernel of this core hands over its slice: {start, end, lookup address}.
        DataflowBuffer ready_dfb(dfb::ready);
        ready_dfb.wait_front(1);
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_dfb.get_read_ptr());
        start = ready[0];
        end = ready[1];
        const uint32_t addr = ready[2];
        list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr + list_offset);
        phys = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr + phys_offset);
        ready_dfb.pop_front(1);
    }
    if (start == end) {
        return;
    }
#endif
#else
    lookup_dfb.reserve_back(1);
    const uint32_t lookup_addr = lookup_dfb.get_write_ptr();
    noc.async_read(pool_count, lookup_dfb, 64, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    invalidate_l1_cache();
    const uint32_t count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lookup_addr)[0];

    // Only blocks holding the pool's count keys are used, split evenly over the cores.
    const uint32_t num_used = tt::data_movement::common::div_up(count, cbs);
    const uint32_t per_core =
        tt::data_movement::common::div_up(tt::data_movement::common::div_up(num_used, num_cores), align_blocks) *
        align_blocks;
    const uint32_t start = std::min(core_index * per_core, num_used);
    const uint32_t end = std::min(start + per_core, num_used);
    if (start == end) {
        return;
    }

    // This core's slice of the list, read from the 64 B boundary at or below it.
    const uint32_t first_byte = start * sizeof(uint32_t) / 64 * 64;
    const uint32_t last_byte = tt::data_movement::common::div_up(end * sizeof(uint32_t), 64) * 64;
    noc.async_read(
        candidate_blocks,
        lookup_dfb,
        last_byte - first_byte,
        {.page_id = 0, .offset_bytes = first_byte},
        {.offset_bytes = list_offset});
    noc.async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        lookup_addr + list_offset + start * sizeof(uint32_t) - first_byte);

    // Physical page block of each of the core's candidate blocks. The list ascends, so each 64 B
    // page-table chunk (16 entries) is read once.
    volatile tt_l1_ptr uint32_t* chunk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lookup_addr + chunk_offset);
    volatile tt_l1_ptr uint32_t* phys = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lookup_addr + phys_offset);
#ifdef PAGE_TABLE_ROW
    // This session's page-table row, staged where the chunks then land.
    const auto page_table_row_gen = TensorAccessor(tensor::page_table_row);
    noc.async_read(page_table_row_gen, lookup_dfb, 64, {.page_id = 0}, {.offset_bytes = chunk_offset});
    noc.async_read_barrier();
    invalidate_l1_cache();
    const uint32_t page_table_row = chunk[0];
#else
    const uint32_t page_table_row = 0;
#endif
    uint32_t loaded_chunk = UINT32_MAX;
    for (uint32_t c = start; c < end; ++c) {
        const uint32_t logical_block = list[c - start] * cbs / page_block_size;
        if (logical_block / 16 != loaded_chunk) {
            loaded_chunk = logical_block / 16;
            noc.async_read(
                page_table,
                lookup_dfb,
                64,
                {.page_id = page_table_row, .offset_bytes = 64 * loaded_chunk},
                {.offset_bytes = chunk_offset});
            noc.async_read_barrier();
            invalidate_l1_cache();
        }
        phys[c - start] = chunk[logical_block % 16];
    }

#endif

#ifdef WRITE_ROWS
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lookup_addr + rows_offset);
    for (uint32_t c = start; c < end; ++c) {
        const uint32_t key = list[c - start] * cbs;
        const uint32_t first_row = phys[c - start] * page_block_size + key % page_block_size;
        for (uint32_t i = 0; i < cbs; ++i) {
            rows[(c - start) * cbs + i] = first_row + i;
        }
    }
    (void)ckernel::load_blocking(&rows[(end - start) * cbs - 1]);
    noc.async_write(
        lookup_dfb,
        pool_rows,
        (end - start) * cbs * sizeof(uint32_t),
        {.offset_bytes = rows_offset},
        {.page_id = 0, .offset_bytes = start * cbs * sizeof(uint32_t)});
#ifdef UNION_ROWS
    {
        const auto pool_masks = TensorAccessor(tensor::pool_masks);  // [1, num_rows * topk_blocks * cbs] INT32
        volatile tt_l1_ptr uint32_t* mask_rows =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lookup_addr + mask_rows_offset);
        for (uint32_t c = start; c < end; ++c) {
            for (uint32_t i = 0; i < cbs; ++i) {
                mask_rows[(c - start) * cbs + i] = mask_list[c - start];
            }
        }
        (void)ckernel::load_blocking(&mask_rows[(end - start) * cbs - 1]);
        noc.async_write(
            lookup_dfb,
            pool_masks,
            (end - start) * cbs * sizeof(uint32_t),
            {.offset_bytes = mask_rows_offset},
            {.page_id = 0, .offset_bytes = start * cbs * sizeof(uint32_t)});
    }
#endif
#endif

    // Keys: the first half of the core's blocks on one kernel, the rest on the other.
    const uint32_t mid = start + (end - start + 1) / 2;
    const uint32_t key_start = half == 0 ? start : mid;
    const uint32_t key_end = half == 0 ? mid : end;
    stage_dfb.reserve_back(stage_chunks);
    uint32_t bank = 0;
    for (uint32_t c = key_start; c < key_end; c += bank_blocks, bank ^= 1) {
        const uint32_t n = std::min(bank_blocks, key_end - c);
        const uint32_t base = bank * bank_blocks * chunks_per_block;
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t row = list[c + j - start] * cbs % page_block_size;
            const uint32_t first_tile = phys[c + j - start] * tiles_per_block + row / 32 * tiles_per_row;
            const uint32_t run_offset = (row % 32) / 16 * 2 * kFaceBytes + (row % 16) * kFaceRowBytes;
            for (uint32_t t = 0; t < tiles_per_row; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc.async_read(
                        key_cache,
                        stage_dfb,
                        chunk_bytes,
                        {.page_id = first_tile + t, .offset_bytes = run_offset + f * kFaceBytes},
                        {.offset_bytes = (base + j * chunks_per_block + 2 * t + f) * chunk_bytes});
                }
            }
        }
        noc.async_read_barrier();
        // Drains the other bank's writes, which were overlapping these reads, so it can be read into next.
        noc.async_writes_flushed();
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t entry = (c + j) * cbs;
            const uint32_t row = entry % page_block_size;
            const uint32_t first_tile = entry / page_block_size * tiles_per_block + row / 32 * tiles_per_row;
            const uint32_t run_offset = (row % 32) / 16 * 2 * kFaceBytes + (row % 16) * kFaceRowBytes;
            for (uint32_t t = 0; t < tiles_per_row; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc.async_write(
                        stage_dfb,
                        pool_keys,
                        chunk_bytes,
                        {.offset_bytes = (base + j * chunks_per_block + 2 * t + f) * chunk_bytes},
                        {.page_id = first_tile + t, .offset_bytes = run_offset + f * kFaceBytes});
                }
            }
        }
    }
    noc.async_write_barrier();
}
