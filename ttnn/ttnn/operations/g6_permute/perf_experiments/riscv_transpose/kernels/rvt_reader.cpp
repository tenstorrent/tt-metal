// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// riscv_transpose bake-off: verbatim copy of kernels/g6_permute_blocks_reader.cpp (the op's shared reader).
// g6_permute reader of the ROW_MAJOR regimes (blocks_rm_gather, nhwc_rm_transpose): load_block.
// A unit is a run of consecutive input pages, and a core's units are consecutive, so a block is one
// page range. Units come in rows of units_per_row whose last unit holds tail_pages pages and the
// others pages_per_unit: blocks_rm_gather has one unit (n, c, h) of W * I pages per row;
// nhwc_rm_transpose cuts each row (n, h) of W pages into units of block_w pages and a shorter tail.
// Every page lands in its own slot of slot_bytes (the page's bank stride), so local and remote
// address agree mod 64 as Blackhole DRAM reads require. One read barrier and one push per block.
// Each page costs one request at the DRAM port it enters, and a port retires requests at a fixed
// rate, so the reader deals a share alt_num / alt_den of its pages to the second NoC, which enters
// every bank through a different port. Pages go to a NoC a whole bank round (num_banks consecutive
// pages, one per bank) at a time, so every bank sees the same split.
// bank_major issues a block chunk_pages at a time, each chunk bank by bank: pages k, k + num_banks, ...
// of one bank sit at consecutive bank addresses, so a core's requests to a bank arrive as one run, and
// one set_state per (bank, NoC) run serves every read of it. chunk_pages bounds the run length, which
// a DRAM port pays for in head-of-line blocking of the other cores' requests.
// stagger rotates the issue order of every block (pages, or banks when bank_major or coalesced) so cores
// that start on the same bank do not walk the banks in lockstep.
// coalesced (regime blocks_rm_bank_coalesced) reads each bank's pages of a block in one transaction of
// whole slots instead of one request per page: the pages of a bank are consecutive slots there. The
// slots land bank by bank in a staging area (cb_bank_staging, private to this kernel), and the reader
// copies each page's payload into cb_input_slots at page_stride, in page order. The reads of the next
// block are in flight during that copy when staging_depth is 2.
// Raw dataflow: no kernel_lib helper reads ROW_MAJOR pages into strided CB slots.

#include "api/dataflow/dataflow_api.h"
#include "g6_permute_common.hpp"
#if defined(RVT_SPLIT_DEN) || defined(RVT_NC_HI)
#include "rvt_sync.hpp"
#endif
#ifdef RVT_NC_HI
#include "rvt_transpose.hpp"
#endif

#include <utility>

constexpr uint32_t cb_input_slots = get_compile_time_arg_val(0);
constexpr uint32_t block_h = get_compile_time_arg_val(1);
constexpr uint32_t pages_per_unit = get_compile_time_arg_val(2);
constexpr uint32_t in_page_bytes = get_compile_time_arg_val(3);
constexpr uint32_t slot_bytes = get_compile_time_arg_val(4);
constexpr uint32_t alt_num = get_compile_time_arg_val(5);
constexpr uint32_t alt_den = get_compile_time_arg_val(6);
constexpr uint32_t num_banks = get_compile_time_arg_val(7);
constexpr bool bank_major = get_compile_time_arg_val(8) != 0;
constexpr uint32_t chunk_pages = get_compile_time_arg_val(9);
constexpr uint32_t units_per_row = get_compile_time_arg_val(10);
constexpr uint32_t tail_pages = get_compile_time_arg_val(11);
constexpr bool coalesced = get_compile_time_arg_val(12) != 0;
constexpr uint32_t cb_bank_staging = get_compile_time_arg_val(13);
constexpr uint32_t staging_rounds = get_compile_time_arg_val(14);
constexpr uint32_t staging_depth = get_compile_time_arg_val(15);
constexpr uint32_t page_stride = get_compile_time_arg_val(16);
constexpr uint32_t compact_batch_words = get_compile_time_arg_val(17);
constexpr auto input_args = TensorAccessorArgs<18>();

// One staging region per bank holds staging_rounds slots; region b of a block starting at page p0 holds
// page g of bank b at slot g / num_banks - p0 / num_banks.
constexpr uint32_t region_bytes = staging_rounds * slot_bytes;
constexpr uint32_t stage_bytes = num_banks * region_bytes;

static_assert(slot_bytes >= in_page_bytes);
static_assert(coalesced || page_stride == slot_bytes, "the page-wise paths land each page in its own slot");
static_assert(!coalesced || (page_stride >= in_page_bytes && page_stride % sizeof(uint32_t) == 0));
static_assert(!coalesced || (staging_depth == 1 || staging_depth == 2));
static_assert(!bank_major || chunk_pages % num_banks == 0, "a chunk is whole bank rounds");
static_assert(alt_num < alt_den && (alt_den & (alt_den - 1)) == 0, "alt share is a power-of-two fraction below 1");
static_assert(units_per_row >= 1 && tail_pages >= 1 && tail_pages <= pages_per_unit);

// Pages of the next `extent` units; row_pos is the next unit's position in its row.
FORCE_INLINE uint32_t block_pages(uint32_t extent, uint32_t& row_pos) {
    if constexpr (tail_pages == pages_per_unit) {
        return extent * pages_per_unit;
    } else {
        uint32_t pages = extent * pages_per_unit;
        for (uint32_t u = 0; u < extent; ++u) {
            if (++row_pos == units_per_row) {
                row_pos = 0;
                pages -= pages_per_unit - tail_pages;
            }
        }
        return pages;
    }
}

// Round r goes to the second NoC when a Bresenham accumulator stepping by alt_num wraps mod alt_den.
FORCE_INLINE uint8_t page_noc(uint32_t page) {
    if constexpr (alt_num == 0) {
        return noc_index;
    } else {
        const uint32_t round = page / num_banks;
        return ((round * alt_num) & (alt_den - 1)) >= alt_den - alt_num ? 1 - noc_index : noc_index;
    }
}

// The share of a run of `count` reads that goes on the second NoC; alt_carry spreads the rounding of
// alt_num / alt_den over the runs of the core.
FORCE_INLINE uint32_t alt_run_count(uint32_t count, uint32_t& alt_carry) {
    if constexpr (alt_num == 0) {
        return 0;
    } else {
        alt_carry += count * alt_num;
        const uint32_t alt_count = alt_carry / alt_den;
        alt_carry %= alt_den;
        return alt_count;
    }
}

template <typename Accessor>
FORCE_INLINE void issue_page_range(
    const Accessor& input, uint32_t first_page, uint32_t slots, uint32_t k_begin, uint32_t k_end) {
    uint32_t slot = slots + k_begin * slot_bytes;
    for (uint32_t k = k_begin; k < k_end; ++k) {
        const uint32_t page = first_page + k;
        if constexpr (alt_num == 0) {
            noc_async_read<in_page_bytes>(input.get_noc_addr(page), slot, in_page_bytes);
        } else {
            const uint8_t noc = page_noc(page);
            noc_async_read<in_page_bytes>(input.get_noc_addr(page, 0, noc), slot, in_page_bytes, noc);
        }
        slot += slot_bytes;
    }
}

// count pages k, k + num_banks, ... of the block on one NoC; slot_bytes is also the bank address stride.
template <typename Accessor>
FORCE_INLINE void issue_bank_run(
    const Accessor& input, uint32_t first_page, uint32_t slots, uint32_t k, uint32_t count, uint8_t noc) {
    if (count == 0) {
        return;
    }
    const uint64_t src = input.get_noc_addr(first_page + k, 0, noc);
    noc_async_read_one_packet_set_state(src, in_page_bytes, 0, noc);
    uint32_t src_addr = static_cast<uint32_t>(src);
    uint32_t slot = slots + k * slot_bytes;
    for (uint32_t r = 0; r < count; ++r) {
        noc_async_read_one_packet_with_state(src_addr, slot, 0, noc);
        src_addr += slot_bytes;
        slot += num_banks * slot_bytes;
    }
}

// Each bank's run is split between the NoCs; alt_carry spreads the rounding of alt_num / alt_den over
// the runs of the core.
template <typename Accessor>
FORCE_INLINE void issue_chunk_bank_major(
    const Accessor& input,
    uint32_t first_page,
    uint32_t slots,
    uint32_t num_pages,
    uint32_t stagger,
    uint32_t& alt_carry) {
    const uint32_t runs = num_pages < num_banks ? num_pages : num_banks;
    for (uint32_t b = 0; b < runs; ++b) {
        const uint32_t k = (b + stagger) % runs;
        const uint32_t count = (num_pages - k + num_banks - 1) / num_banks;
        const uint32_t alt_count = alt_run_count(count, alt_carry);
        const uint32_t own_count = count - alt_count;
        issue_bank_run(input, first_page, slots, k, own_count, noc_index);
        issue_bank_run(input, first_page, slots, k + own_count * num_banks, alt_count, 1 - noc_index);
    }
}

template <typename Accessor>
FORCE_INLINE void issue_block_bank_major(
    const Accessor& input,
    uint32_t first_page,
    uint32_t slots,
    uint32_t num_pages,
    uint32_t stagger,
    uint32_t& alt_carry) {
    for (uint32_t done = 0; done < num_pages; done += chunk_pages) {
        const uint32_t left = num_pages - done;
        issue_chunk_bank_major(
            input,
            first_page + done,
            slots + done * slot_bytes,
            left < chunk_pages ? left : chunk_pages,
            stagger,
            alt_carry);
    }
}

template <typename Accessor>
FORCE_INLINE void issue_block_reads(
    const Accessor& input, uint32_t& page, uint32_t num_pages, uint32_t stagger, uint32_t& alt_carry) {
    const uint32_t slots = get_write_ptr(cb_input_slots);
    if constexpr (bank_major) {
        issue_block_bank_major(input, page, slots, num_pages, stagger, alt_carry);
    } else {
        const uint32_t first = stagger < num_pages ? stagger : stagger % num_pages;
        issue_page_range(input, page, slots, first, num_pages);
        issue_page_range(input, page, slots, 0, first);
    }
    page += num_pages;
}

FORCE_INLINE void block_read_barrier() {
    noc_async_read_barrier(noc_index);
    if constexpr (alt_num != 0) {
        noc_async_read_barrier(1 - noc_index);
    }
}

// Bank runs of the block [first_page, first_page + num_pages): run r starts at page first_page + r, in
// bank (first_page + r) mod num_banks, and holds every num_banks-th page after it.
struct BankRun {
    uint32_t page, count, region_slot;
};

FORCE_INLINE uint32_t bank_runs(uint32_t num_pages) { return num_pages < num_banks ? num_pages : num_banks; }

FORCE_INLINE BankRun bank_run(uint32_t first_page, uint32_t bank0, uint32_t num_pages, uint32_t r, uint32_t& bank) {
    bank = bank0 + r;
    // A run whose bank precedes the block's first bank starts one round later.
    const bool next_round = bank >= num_banks;
    if (next_round) {
        bank -= num_banks;
    }
    return {first_page + r, (num_pages - r + num_banks - 1) / num_banks, next_round ? 1u : 0u};
}

// One read of whole slots per bank run (two when the run is split over the NoCs). Region and slot bases
// are slot_bytes-aligned like the bank address, so local and remote agree mod 64.
template <typename Accessor>
FORCE_INLINE void issue_block_coalesced(
    const Accessor& input,
    uint32_t first_page,
    uint32_t num_pages,
    uint32_t stage,
    uint32_t stagger,
    uint32_t& alt_carry) {
    const uint32_t bank0 = first_page % num_banks;
    const uint32_t runs = bank_runs(num_pages);
    const uint32_t r0 = stagger < runs ? stagger : stagger % runs;
    for (uint32_t i = 0; i < runs; ++i) {
        const uint32_t r = r0 + i < runs ? r0 + i : r0 + i - runs;
        uint32_t bank;
        const BankRun run = bank_run(first_page, bank0, num_pages, r, bank);
        const uint32_t dst = stage + bank * region_bytes + run.region_slot * slot_bytes;
        const uint32_t alt_count = alt_run_count(run.count, alt_carry);
        const uint32_t own_count = run.count - alt_count;
        if (own_count != 0) {
            noc_async_read(input.get_noc_addr(run.page, 0, noc_index), dst, own_count * slot_bytes, noc_index);
        }
        if constexpr (alt_num != 0) {
            if (alt_count != 0) {
                const uint8_t alt_noc = 1 - noc_index;
                noc_async_read(
                    input.get_noc_addr(run.page + own_count * num_banks, 0, alt_noc),
                    dst + own_count * slot_bytes,
                    alt_count * slot_bytes,
                    alt_noc);
            }
        }
    }
}

// Pages per batch of the compact: a batch issues all its loads before its first store, so their L1
// latencies overlap (the kernel builds at -Os, which does not unroll). 1 copies page by page.
constexpr uint32_t compact_batch =
    compact_batch_words <= page_stride / sizeof(uint32_t) ? 1 : compact_batch_words / (page_stride / sizeof(uint32_t));

template <uint32_t words, uint32_t src_step, uint32_t dst_step, uint32_t... b>
FORCE_INLINE void copy_pages(
    const tt_l1_ptr uint32_t* src, tt_l1_ptr uint32_t* dst, std::integer_sequence<uint32_t, b...>) {
    const uint32_t v[] = {src[(b / words) * src_step + b % words]...};
    ((dst[(b / words) * dst_step + b % words] = v[b]), ...);
}

// Copies the payload of every page of the block from its staging slot to page_stride-spaced pages of
// cb_input_slots, in page order: run r's pages are num_banks pages apart there.
FORCE_INLINE void compact_block(uint32_t first_page, uint32_t num_pages, uint32_t stage, uint32_t pages_base) {
    constexpr uint32_t words = page_stride / sizeof(uint32_t);
    constexpr uint32_t src_step = slot_bytes / sizeof(uint32_t);
    constexpr uint32_t dst_step = num_banks * page_stride / sizeof(uint32_t);
    const uint32_t bank0 = first_page % num_banks;
    const uint32_t runs = bank_runs(num_pages);
    // Plain loads and stores: after the read barrier that made the slots current, before the push.
    asm volatile("" ::: "memory");
    for (uint32_t r = 0; r < runs; ++r) {
        uint32_t bank;
        const BankRun run = bank_run(first_page, bank0, num_pages, r, bank);
        const auto* src =
            reinterpret_cast<const tt_l1_ptr uint32_t*>(stage + bank * region_bytes + run.region_slot * slot_bytes);
        auto* dst = reinterpret_cast<tt_l1_ptr uint32_t*>(pages_base + r * page_stride);
        uint32_t m = 0;
        if constexpr (compact_batch > 1) {
            for (; m + compact_batch <= run.count; m += compact_batch) {
                copy_pages<words, src_step, dst_step>(
                    src, dst, std::make_integer_sequence<uint32_t, compact_batch * words>{});
                src += compact_batch * src_step;
                dst += compact_batch * dst_step;
            }
        }
        for (; m < run.count; ++m) {
            for (uint32_t w = 0; w < words; ++w) {
                dst[w] = src[w];
            }
            src += src_step;
            dst += dst_step;
        }
    }
    asm volatile("" ::: "memory");
}

// The block whose reads are in flight (or landed) in one stage.
struct StagedBlock {
    uint32_t page, num_pages, stage;
};

template <bool zoned, typename Accessor>
FORCE_INLINE void issue_staged(const Accessor& input, const StagedBlock& block, uint32_t stagger, uint32_t& alt_carry) {
    G6_ZONED_STEP(
        zoned,
        "PERM-RD-ISSUE",
        issue_block_coalesced(input, block.page, block.num_pages, block.stage, stagger, alt_carry));
}

// Block `cur` has been issued. Waits for it, issues `next` (when has_next) so its reads fly during the
// copy, copies `cur` into cb_input_slots and pushes it. With one stage, `next` reuses cur's stage and is
// issued after the copy.
template <bool zoned, typename Accessor>
FORCE_INLINE void load_block_coalesced(
    const Accessor& input,
    const StagedBlock& cur,
    const StagedBlock& next,
    bool has_next,
    uint32_t stagger,
    uint32_t& alt_carry) {
    G6_ZONED_STEP(zoned, "PERM-RD-BARRIER", block_read_barrier());
    if (staging_depth == 2 && has_next) {
        issue_staged<zoned>(input, next, stagger, alt_carry);
    }
    G6_ZONED_STEP(zoned, "PERM-RD-RESERVE", cb_reserve_back(cb_input_slots, block_h));
    G6_ZONED_STEP(
        zoned, "PERM-RD-COMPACT", compact_block(cur.page, cur.num_pages, cur.stage, get_write_ptr(cb_input_slots)));
    cb_push_back(cb_input_slots, block_h);
    if (staging_depth == 1 && has_next) {
        issue_staged<zoned>(input, next, stagger, alt_carry);
    }
}

template <typename Accessor>
FORCE_INLINE void read_coalesced(
    const Accessor& input, uint32_t page, uint32_t remaining, uint32_t row_pos, uint32_t stagger, uint32_t& alt_carry) {
    const uint32_t staging = get_write_ptr(cb_bank_staging);
    uint32_t extent = remaining < block_h ? remaining : block_h;
    StagedBlock cur{page, block_pages(extent, row_pos), staging};
    issue_staged<false>(input, cur, stagger, alt_carry);
    uint32_t stage_idx = 0;
    for (uint32_t block_idx = 0;; ++block_idx) {
        remaining -= extent;
        const bool has_next = remaining > 0;
        StagedBlock next{cur.page + cur.num_pages, 0, cur.stage};
        if (has_next) {
            extent = remaining < block_h ? remaining : block_h;
            next.num_pages = block_pages(extent, row_pos);
            stage_idx = staging_depth == 2 ? 1 - stage_idx : 0;
            next.stage = staging + stage_idx * stage_bytes;
        }
        if (block_idx < g6_permute::ZONED_BLOCKS) {
            DeviceZoneScopedN("PERM-RD-LOAD-BLOCK");
            load_block_coalesced<true>(input, cur, next, has_next, stagger, alt_carry);
        } else {
            load_block_coalesced<false>(input, cur, next, has_next, stagger, alt_carry);
        }
        if (!has_next) {
            return;
        }
        cur = next;
    }
}

template <bool zoned, typename Accessor>
FORCE_INLINE void load_block(
    const Accessor& input, uint32_t& page, uint32_t num_pages, uint32_t stagger, uint32_t& alt_carry) {
    G6_ZONED_STEP(zoned, "PERM-RD-RESERVE", cb_reserve_back(cb_input_slots, block_h));
    G6_ZONED_STEP(zoned, "PERM-RD-ISSUE", issue_block_reads(input, page, num_pages, stagger, alt_carry));
    G6_ZONED_STEP(zoned, "PERM-RD-BARRIER", block_read_barrier());
    // Nominal count on a ragged last block too: the writer waits and pops the same quantum.
    cb_push_back(cb_input_slots, block_h);
}

#ifdef RVT_SPLIT_DEN
// riscv_transpose bake-off (S1 read split): BRISC reads the last width * RVT_SPLIT_NUM / RVT_SPLIT_DEN
// pages of every unit on its own NoC; this kernel reads the others (page order) and pushes a block once
// BRISC's counter RVT_CB_BRD says its share of the block has landed too.
constexpr uint32_t split_num = RVT_SPLIT_NUM;
constexpr uint32_t split_den = RVT_SPLIT_DEN;
#ifdef RVT_SPLIT0_DEN
// Block 0 (the pipeline fill) can use its own split.
constexpr uint32_t split0_num = RVT_SPLIT0_NUM;
constexpr uint32_t split0_den = RVT_SPLIT0_DEN;
#else
constexpr uint32_t split0_num = split_num;
constexpr uint32_t split0_den = split_den;
#endif

template <typename Accessor>
FORCE_INLINE void issue_block_split(
    const Accessor& input, uint32_t& page, uint32_t extent, uint32_t& row_pos, uint32_t block_idx) {
    uint32_t slots = get_write_ptr(cb_input_slots);
    for (uint32_t u = 0; u < extent; ++u) {
        const uint32_t width = row_pos == units_per_row - 1 ? tail_pages : pages_per_unit;
        const uint32_t brisc = block_idx == 0 ? width * split0_num / split0_den : width * split_num / split_den;
        issue_page_range(input, page, slots, 0, width - brisc);
        page += width;
        slots += width * slot_bytes;
        if (++row_pos == units_per_row) {
            row_pos = 0;
        }
    }
}

template <bool zoned, typename Accessor>
FORCE_INLINE void load_block_split(
    const Accessor& input, uint32_t& page, uint32_t extent, uint32_t& row_pos, uint32_t block_idx) {
    G6_ZONED_STEP(zoned, "PERM-RD-RESERVE", cb_reserve_back(cb_input_slots, block_h));
    G6_ZONED_STEP(zoned, "PERM-RD-ISSUE", issue_block_split(input, page, extent, row_pos, block_idx));
    G6_ZONED_STEP(zoned, "PERM-RD-BARRIER", block_read_barrier());
    G6_ZONED_STEP(zoned, "PERM-RD-WAIT-BRISC", rvt::ctr_wait(RVT_CB_BRD, block_idx + 1));
    cb_push_back(cb_input_slots, block_h);
}
#endif

#ifdef RVT_NC_HI
// riscv_transpose bake-off (NCRISC helper share): this RISC also transposes row items
// [RVT_NC_LO, RVT_NC_HI) of every unit of block b while the reads of block b + 1 are in flight, then posts
// its done counter RVT_CB_DONE_N = b + 1 (the writer pops a block once every helper has posted it).
constexpr uint32_t nc_lo = RVT_NC_LO;
constexpr uint32_t nc_hi = RVT_NC_HI;
constexpr uint32_t nc_c = RVT_NC_C;
constexpr uint32_t nc_row_bytes = RVT_NC_ROW_BYTES;
constexpr uint32_t nc_rows_depth = RVT_NC_ROWS_DEPTH;
constexpr uint32_t nc_input_depth = RVT_NC_INPUT_DEPTH;
constexpr uint32_t nc_block_slot_bytes = block_h * pages_per_unit * slot_bytes;
constexpr uint32_t nc_block_row_bytes = block_h * nc_c * nc_row_bytes;

FORCE_INLINE void nc_transpose_block(
    uint32_t slots_base, uint32_t rows_base, uint32_t b, uint32_t extent, uint32_t& wb) {
    if (b >= nc_rows_depth) {
        rvt::ctr_wait(RVT_CB_FREE, b + 1 - nc_rows_depth);
    }
    rvt::l1_invalidate();
    uint32_t src = slots_base + (b % nc_input_depth) * nc_block_slot_bytes;
    uint32_t dst = rows_base + (b % nc_rows_depth) * nc_block_row_bytes;
    asm volatile("" ::: "memory");
    for (uint32_t u = 0; u < extent; ++u) {
        if (tail_pages != pages_per_unit && wb == units_per_row - 1) {
            rvt::transpose_rows<tail_pages, nc_c, slot_bytes, nc_row_bytes, 16>(src, dst, nc_lo, nc_hi);
            src += tail_pages * slot_bytes;
        } else {
            rvt::transpose_rows<pages_per_unit, nc_c, slot_bytes, nc_row_bytes, 16>(src, dst, nc_lo, nc_hi);
            src += pages_per_unit * slot_bytes;
        }
        dst += nc_c * nc_row_bytes;
        if (++wb == units_per_row) {
            wb = 0;
        }
    }
    asm volatile("" ::: "memory");
    rvt::ctr_post(RVT_CB_DONE_N, b + 1);
}

template <typename Accessor>
FORCE_INLINE void read_with_share(
    const Accessor& input, uint32_t page, uint32_t remaining, uint32_t row_pos, uint32_t stagger, uint32_t& alt_carry) {
    const uint32_t slots_base = get_write_ptr(cb_input_slots);
    const uint32_t rows_base = get_write_ptr(RVT_CB_ROWS);
    uint32_t wb = row_pos;
    uint32_t extent = remaining < block_h ? remaining : block_h;
    {
        DeviceZoneScopedN("PERM-RD-FIRST-BLOCK");
        cb_reserve_back(cb_input_slots, block_h);
        issue_block_reads(input, page, block_pages(extent, row_pos), stagger, alt_carry);
        block_read_barrier();
        cb_push_back(cb_input_slots, block_h);
    }
    for (uint32_t b = 0;; ++b) {
        remaining -= extent;
        const uint32_t next_extent = remaining < block_h ? remaining : block_h;
        if (remaining > 0) {
            DeviceZoneScopedN("PERM-RD-ISSUE");
            cb_reserve_back(cb_input_slots, block_h);
            issue_block_reads(input, page, block_pages(next_extent, row_pos), stagger, alt_carry);
        }
        {
            DeviceZoneScopedN("PERM-RD-XPOSE");
            nc_transpose_block(slots_base, rows_base, b, extent, wb);
        }
        if (remaining == 0) {
            return;
        }
        {
            DeviceZoneScopedN("PERM-RD-BARRIER");
            block_read_barrier();
        }
        cb_push_back(cb_input_slots, block_h);
        extent = next_extent;
    }
}
#endif

void kernel_main() {
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    uint32_t page = get_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(1);
    const uint32_t stagger = get_arg_val<uint32_t>(2);
    uint32_t row_pos = get_arg_val<uint32_t>(3);
    uint32_t alt_carry = 0;

    const auto input = TensorAccessor(input_args, input_addr, in_page_bytes);

    if constexpr (coalesced) {
        if (remaining > 0) {
            read_coalesced(input, page, remaining, row_pos, stagger, alt_carry);
        }
        return;
    }

#ifdef RVT_SPLIT_DEN
    for (uint32_t b = 0; remaining > 0; ++b) {
        const uint32_t extent = remaining < block_h ? remaining : block_h;
        if (b < g6_permute::ZONED_BLOCKS) {
            DeviceZoneScopedN("PERM-RD-LOAD-BLOCK");
            load_block_split<true>(input, page, extent, row_pos, b);
        } else {
            load_block_split<false>(input, page, extent, row_pos, b);
        }
        remaining -= extent;
    }
    return;
#endif

#ifdef RVT_NC_HI
    if (remaining > 0) {
        read_with_share(input, page, remaining, row_pos, stagger, alt_carry);
    }
    return;
#endif

    uint32_t block_idx = 0;
    for (; block_idx < g6_permute::ZONED_BLOCKS && remaining > 0; ++block_idx) {
        DeviceZoneScopedN("PERM-RD-LOAD-BLOCK");
        const uint32_t extent = remaining < block_h ? remaining : block_h;
        load_block<true>(input, page, block_pages(extent, row_pos), stagger, alt_carry);
        remaining -= extent;
    }
    if (remaining > 0) {
        DeviceZoneScopedN("PERM-RD-TAIL");
        while (remaining > 0) {
            const uint32_t extent = remaining < block_h ? remaining : block_h;
            load_block<false>(input, page, block_pages(extent, row_pos), stagger, alt_carry);
            remaining -= extent;
        }
    }
}
