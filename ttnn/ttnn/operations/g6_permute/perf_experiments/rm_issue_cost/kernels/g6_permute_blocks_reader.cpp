// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// perf experiment rm_issue_cost: copy of kernels/g6_permute_blocks_reader.cpp with variant guards.
//   RIC_READ_CURSOR  raw-register page issue on the page-order path (alt_num == 0, !bank_major, !coalesced):
//                    bypasses noc_async_read + TensorAccessor::get_noc_addr. One division per range gives the
//                    first page's (bank, bank address); a cursor then steps the bank per page (bank address +1
//                    slot at each wrap), the NoC XY and bank offset come from the firmware's bank tables, and
//                    NOC_AT_LEN_BE is set once per range, so a page costs 4 register writes and one ready poll.
//                    The DRAM bank-major path and the coalesced path (FLUX) are unchanged.
//   RIC_READ_2BUF    with RIC_READ_CURSOR: pages alternate between read_cmd_buf and write_reg_cmd_buf (unused by
//                    this kernel; programmed as a read buffer at kernel start, restored at the end). The ready
//                    poll of the next buffer is loaded before the current buffer's register writes, so its MMIO
//                    latency overlaps them, and the next bank's table entries are loaded one page ahead.
//   RIC_UNIT_PUSH    page-order path: one barrier and one push per unit (the writer waits per unit), so the
//                    consumer starts after the first unit instead of the first block. block_h sets the ring.
//   RIC_TRID_PIPE    with RIC_UNIT_PUSH: unit k + 1 is issued under the other read transaction id before the
//                    reader waits for unit k (noc_async_read_barrier_with_trid), so a unit's wait overlaps the
//                    next unit's issue.
//   RIC_TRID_FIRST_WAIT  with RIC_TRID_PIPE: the first unit is waited for and pushed before the second is issued.
//   ABL_NO_READS     stub the reads (ablation).
// The rest is the op's reader verbatim.
//
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

#if defined(RIC_READ_CURSOR)
constexpr bool in_is_dram = input_args.is_dram;
static_assert(num_banks == (in_is_dram ? NUM_DRAM_BANKS : NUM_L1_BANKS), "host bank count = firmware bank count");

FORCE_INLINE uint32_t in_bank_xy(uint32_t bank) {
    if constexpr (in_is_dram) {
        return dram_bank_to_noc_xy[noc_index][bank];
    } else {
        return l1_bank_to_noc_xy[noc_index][bank];
    }
}

FORCE_INLINE uint32_t in_bank_offset(uint32_t bank) {
    if constexpr (in_is_dram) {
        return bank_to_dram_offset[bank];
    } else {
        return bank_to_l1_offset[bank];
    }
}

// Pages [first_page + k_begin, first_page + k_end) into slots k_begin.. of `slots`; slot_bytes is also the bank
// address stride (InterleavedAddrGen::aligned_page_size). Dedicated NoC mode only (alt_num == 0).
template <uint8_t mode = noc_mode>
FORCE_INLINE void issue_page_range_cursor(
    uint32_t input_addr, uint32_t first_page, uint32_t slots, uint32_t k_begin, uint32_t k_end) {
    static_assert(mode == DM_DEDICATED_NOC);
    if (k_begin >= k_end) {
        return;
    }
    constexpr uint32_t noc = noc_index;
    constexpr uint32_t buf = read_cmd_buf;
    const uint32_t page = first_page + k_begin;
    const uint32_t round = page / num_banks;
    uint32_t bank = page - round * num_banks;
    uint32_t src_round = input_addr + round * slot_bytes;
    uint32_t slot = slots + k_begin * slot_bytes;
    while (!noc_cmd_buf_ready(noc, buf));
    NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_AT_LEN_BE, in_page_bytes);
    for (uint32_t k = k_begin; k < k_end; ++k) {
        const uint32_t xy = in_bank_xy(bank);
        const uint32_t src = src_round + in_bank_offset(bank);
        while (!noc_cmd_buf_ready(noc, buf));
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_RET_ADDR_LO, slot);
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_TARG_ADDR_LO, src);
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_TARG_ADDR_COORDINATE, xy);
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        slot += slot_bytes;
        if (++bank == num_banks) {
            bank = 0;
            src_round += slot_bytes;
        }
    }
    noc_reads_num_issued[noc] += k_end - k_begin;
}
#endif

#if defined(RIC_READ_2BUF)
// Only the one-NoC page-order path (dedicated mode) issues through two buffers; the other paths are unchanged.
constexpr bool rd_2buf = noc_mode == DM_DEDICATED_NOC && alt_num == 0 && !bank_major && !coalesced;
constexpr uint32_t rd_buf_a = read_cmd_buf;
constexpr uint32_t rd_buf_b = write_reg_cmd_buf;

FORCE_INLINE volatile uint32_t* noc_reg(uint32_t buf, uint32_t reg) {
    return reinterpret_cast<volatile uint32_t*>(NOC_CMD_BUF_INSTANCE_OFFSET(noc_index, buf) + reg);
}

NocCmdBufState rd_buf_b_saved;

// write_reg_cmd_buf becomes a second read buffer: read command, responses to this core, no PCIe routing.
FORCE_INLINE void rd_2buf_begin() {
    if constexpr (!rd_2buf) {
        return;
    }
    while (!noc_cmd_buf_ready(noc_index, rd_buf_b));
    noc_cmd_buf_save_state(noc_index, rd_buf_b, &rd_buf_b_saved);
    NOC_CMD_BUF_WRITE_REG(noc_index, rd_buf_b, NOC_CTRL, NOC_CMD_BUF_READ_REG(noc_index, rd_buf_a, NOC_CTRL));
    NOC_CMD_BUF_WRITE_REG(
        noc_index,
        rd_buf_b,
        NOC_RET_ADDR_COORDINATE,
        NOC_CMD_BUF_READ_REG(noc_index, rd_buf_a, NOC_RET_ADDR_COORDINATE));
    NOC_CMD_BUF_WRITE_REG(noc_index, rd_buf_b, NOC_RET_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(noc_index, rd_buf_b, NOC_TARG_ADDR_MID, 0);
}

FORCE_INLINE void rd_2buf_end() {
    if constexpr (!rd_2buf) {
        return;
    }
    while (!noc_cmd_buf_ready(noc_index, rd_buf_b));
    noc_cmd_buf_restore_state(noc_index, rd_buf_b, &rd_buf_b_saved);
}

// One page on buffer x while buffer y's ready status is in flight; returns once y is ready.
#define RIC_RD_STEP(x, y)                              \
    {                                                  \
        uint32_t busy = *noc_reg(y, NOC_CMD_CTRL);     \
        const uint32_t src = src_round + off;          \
        *noc_reg(x, NOC_RET_ADDR_LO) = slot;           \
        *noc_reg(x, NOC_TARG_ADDR_LO) = src;           \
        *noc_reg(x, NOC_TARG_ADDR_COORDINATE) = xy;    \
        *noc_reg(x, NOC_CMD_CTRL) = NOC_CTRL_SEND_REQ; \
        slot += slot_bytes;                            \
        if (++bank == num_banks) {                     \
            bank = 0;                                  \
            src_round += slot_bytes;                   \
        }                                              \
        xy = in_bank_xy(bank);                         \
        off = in_bank_offset(bank);                    \
        while (busy != NOC_CTRL_STATUS_READY) {        \
            busy = *noc_reg(y, NOC_CMD_CTRL);          \
        }                                              \
    }

FORCE_INLINE void issue_page_range_cursor2(
    uint32_t input_addr, uint32_t first_page, uint32_t slots, uint32_t k_begin, uint32_t k_end) {
    if (k_begin >= k_end) {
        return;
    }
    const uint32_t page = first_page + k_begin;
    const uint32_t round = page / num_banks;
    uint32_t bank = page - round * num_banks;
    uint32_t src_round = input_addr + round * slot_bytes;
    uint32_t slot = slots + k_begin * slot_bytes;
    uint32_t xy = in_bank_xy(bank);
    uint32_t off = in_bank_offset(bank);
    while (!noc_cmd_buf_ready(noc_index, rd_buf_a));
    while (!noc_cmd_buf_ready(noc_index, rd_buf_b));
    *noc_reg(rd_buf_a, NOC_AT_LEN_BE) = in_page_bytes;
    *noc_reg(rd_buf_b, NOC_AT_LEN_BE) = in_page_bytes;
    // Same transaction id on both buffers (noc_async_read_set_trid programs read_cmd_buf only).
    *noc_reg(rd_buf_b, NOC_PACKET_TAG) = *noc_reg(rd_buf_a, NOC_PACKET_TAG);
    uint32_t k = k_begin;
    for (; k + 2 <= k_end; k += 2) {
        RIC_RD_STEP(rd_buf_a, rd_buf_b);
        RIC_RD_STEP(rd_buf_b, rd_buf_a);
    }
    if (k < k_end) {
        RIC_RD_STEP(rd_buf_a, rd_buf_b);
    }
    // Both buffers' last commands reach the NIU before a transaction-id barrier reads its counters.
    while (!noc_cmd_buf_ready(noc_index, rd_buf_a));
    while (!noc_cmd_buf_ready(noc_index, rd_buf_b));
    noc_reads_num_issued[noc_index] += k_end - k_begin;
}
#endif

template <typename Accessor>
FORCE_INLINE void issue_page_range(
    const Accessor& input, uint32_t first_page, uint32_t slots, uint32_t k_begin, uint32_t k_end) {
#if defined(RIC_READ_2BUF)
    if constexpr (rd_2buf) {
        issue_page_range_cursor2(input.bank_base_address, first_page, slots, k_begin, k_end);
        return;
    }
#endif
#if defined(RIC_READ_CURSOR)
    if constexpr (alt_num == 0) {
        issue_page_range_cursor(input.bank_base_address, first_page, slots, k_begin, k_end);
        return;
    }
#endif
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

#if defined(RIC_UNIT_PUSH)
constexpr uint32_t push_quantum = 1;
#else
constexpr uint32_t push_quantum = block_h;
#endif

template <bool zoned, typename Accessor>
FORCE_INLINE void load_block(
    const Accessor& input, uint32_t& page, uint32_t num_pages, uint32_t stagger, uint32_t& alt_carry) {
    G6_ZONED_STEP(zoned, "PERM-RD-RESERVE", cb_reserve_back(cb_input_slots, push_quantum));
#ifndef ABL_NO_READS
    G6_ZONED_STEP(zoned, "PERM-RD-ISSUE", issue_block_reads(input, page, num_pages, stagger, alt_carry));
#else
    page += num_pages;
#endif
    G6_ZONED_STEP(zoned, "PERM-RD-BARRIER", block_read_barrier());
    // Nominal count on a ragged last block too: the writer waits and pops the same quantum.
    cb_push_back(cb_input_slots, push_quantum);
}

#if defined(RIC_UNIT_PUSH)
FORCE_INLINE uint32_t unit_pages(uint32_t& row_pos) {
    uint32_t pages = pages_per_unit;
    if (++row_pos == units_per_row) {
        row_pos = 0;
        pages = tail_pages;
    }
    return pages;
}

#if defined(RIC_TRID_PIPE)
constexpr bool trid_pipe = alt_num == 0 && !bank_major;
#else
constexpr bool trid_pipe = false;
#endif

template <typename Accessor>
FORCE_INLINE void read_units_blocking(
    const Accessor& input, uint32_t page, uint32_t remaining, uint32_t row_pos, uint32_t stagger, uint32_t& alt_carry) {
    for (uint32_t k = 0; remaining > 0; ++k, --remaining) {
        const uint32_t pages = unit_pages(row_pos);
        if (k < g6_permute::ZONED_BLOCKS) {
            DeviceZoneScopedN("PERM-RD-LOAD-BLOCK");
            load_block<true>(input, page, pages, stagger, alt_carry);
        } else {
            load_block<false>(input, page, pages, stagger, alt_carry);
        }
    }
}

#if defined(RIC_TRID_PIPE)
// The slot of the unit after the one at `wr` in the ring (units are contiguous: the ring is whole units).
FORCE_INLINE uint32_t next_unit_slot(uint32_t wr) {
    constexpr uint32_t unit_bytes = pages_per_unit * slot_bytes;
    wr += unit_bytes;
    return wr == get_local_cb_interface(cb_input_slots).fifo_limit
               ? wr - get_local_cb_interface(cb_input_slots).fifo_size
               : wr;
}

template <typename Accessor>
FORCE_INLINE void issue_unit_trid(const Accessor& input, uint32_t page, uint32_t pages, uint32_t slots, uint32_t trid) {
    noc_async_read_set_trid(trid);
    issue_page_range(input, page, slots, 0, pages);
}

template <bool zoned>
FORCE_INLINE void wait_unit_trid(uint32_t trid) {
    G6_ZONED_STEP(zoned, "PERM-RD-BARRIER", noc_async_read_barrier_with_trid(trid));
}

template <typename Accessor>
FORCE_INLINE void read_units(
    const Accessor& input, uint32_t page, uint32_t remaining, uint32_t row_pos, uint32_t stagger, uint32_t& alt_carry) {
    // The pipeline holds two units: a one-unit ring takes the blocking walk.
    if (!trid_pipe || get_local_cb_interface(cb_input_slots).fifo_size < 2 * pages_per_unit * slot_bytes) {
        read_units_blocking(input, page, remaining, row_pos, stagger, alt_carry);
        return;
    }
    if (remaining == 0) {
        return;
    }
    // Unit k is in flight under trid 1 + (k & 1); unit k + 1 is issued before unit k is waited for.
    cb_reserve_back(cb_input_slots, 1);
    uint32_t cur_slot = get_write_ptr(cb_input_slots);
    uint32_t pages = unit_pages(row_pos);
    issue_unit_trid(input, page, pages, cur_slot, 1);
    page += pages;
    uint32_t k0 = 0;
#if defined(RIC_TRID_FIRST_WAIT)
    // The first unit is the pipeline fill: push it before issuing the second.
    wait_unit_trid<true>(1);
    cb_push_back(cb_input_slots, 1);
    if (--remaining == 0) {
        noc_async_read_set_trid(0);
        return;
    }
    cb_reserve_back(cb_input_slots, 1);
    cur_slot = get_write_ptr(cb_input_slots);
    pages = unit_pages(row_pos);
    issue_unit_trid(input, page, pages, cur_slot, 2);
    page += pages;
    k0 = 1;
#endif
    for (uint32_t k = k0; remaining > 0; ++k) {
        --remaining;
        if (remaining > 0) {
            const uint32_t next_pages = unit_pages(row_pos);
            {
                DeviceZoneScopedN("PERM-RD-RESERVE");
                cb_reserve_back(cb_input_slots, 2);
            }
            {
                DeviceZoneScopedN("PERM-RD-ISSUE");
                issue_unit_trid(input, page, next_pages, next_unit_slot(cur_slot), 1 + ((k + 1) & 1));
            }
            page += next_pages;
        }
        wait_unit_trid<true>(1 + (k & 1));
        cb_push_back(cb_input_slots, 1);
        cur_slot = get_write_ptr(cb_input_slots);
    }
    noc_async_read_set_trid(0);
}
#else
template <typename Accessor>
FORCE_INLINE void read_units(
    const Accessor& input, uint32_t page, uint32_t remaining, uint32_t row_pos, uint32_t stagger, uint32_t& alt_carry) {
    read_units_blocking(input, page, remaining, row_pos, stagger, alt_carry);
}
#endif
#endif

template <typename Accessor>
FORCE_INLINE void read_all(
    const Accessor& input, uint32_t page, uint32_t remaining, uint32_t stagger, uint32_t row_pos, uint32_t& alt_carry) {
    if constexpr (coalesced) {
        if (remaining > 0) {
            read_coalesced(input, page, remaining, row_pos, stagger, alt_carry);
        }
        return;
    }

#if defined(RIC_UNIT_PUSH)
    read_units(input, page, remaining, row_pos, stagger, alt_carry);
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

void kernel_main() {
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    uint32_t page = get_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(1);
    const uint32_t stagger = get_arg_val<uint32_t>(2);
    uint32_t row_pos = get_arg_val<uint32_t>(3);
    uint32_t alt_carry = 0;

    const auto input = TensorAccessor(input_args, input_addr, in_page_bytes);
#if defined(RIC_READ_2BUF)
    rd_2buf_begin();
#endif
    read_all(input, page, remaining, stagger, row_pos, alt_carry);
#if defined(RIC_READ_2BUF)
    rd_2buf_end();
#endif
}
