// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t sp_size = get_compile_time_arg_val(0);
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr uint32_t table_bytes = get_compile_time_arg_val(2);
constexpr uint32_t allocated_bytes = get_compile_time_arg_val(3);
constexpr uint32_t free_window_bytes = get_compile_time_arg_val(4);
constexpr uint32_t counts_bytes = get_compile_time_arg_val(5);
constexpr uint32_t free_row_bytes = get_compile_time_arg_val(6);
constexpr uint32_t free_window_entries = free_window_bytes / sizeof(uint32_t);

// Both sides of every partial transfer take the same offset, so one granularity must satisfy the DRAM
// and L1 alignments together. Quasar reports one byte; uint32 metadata still needs entry granularity.
constexpr uint32_t dram_alignment_bytes = NOC_DRAM_READ_ALIGNMENT_BYTES > NOC_DRAM_WRITE_ALIGNMENT_BYTES
                                              ? NOC_DRAM_READ_ALIGNMENT_BYTES
                                              : NOC_DRAM_WRITE_ALIGNMENT_BYTES;
constexpr uint32_t l1_alignment_bytes = NOC_L1_READ_ALIGNMENT_BYTES > NOC_L1_WRITE_ALIGNMENT_BYTES
                                            ? NOC_L1_READ_ALIGNMENT_BYTES
                                            : NOC_L1_WRITE_ALIGNMENT_BYTES;
constexpr uint32_t noc_alignment_bytes =
    dram_alignment_bytes > l1_alignment_bytes ? dram_alignment_bytes : l1_alignment_bytes;
constexpr uint32_t alignment_bytes = noc_alignment_bytes > sizeof(uint32_t) ? noc_alignment_bytes
                                                                            : uint32_t(sizeof(uint32_t));
constexpr uint32_t alignment_entries = alignment_bytes / sizeof(uint32_t);

struct ByteRange {
    uint32_t begin;
    uint32_t end;
};

struct AllocationUpdate {
    uint32_t slot;
    bool reset;
    uint32_t old_pages;
    uint32_t next_pages;
    ByteRange allocated_range;
};

struct MetadataRanges {
    ByteRange table;
    ByteRange counts;
    ByteRange wrapped_counts;
};

uint32_t range_size(ByteRange range) { return range.end - range.begin; }

bool needs_update(const AllocationUpdate& update) {
    return update.reset ? update.old_pages != 0 || update.next_pages != 0 : update.next_pages != update.old_pages;
}

// Read/modify/write only the aligned boundary bytes needed to preserve neighbours.
ByteRange aligned_range(uint32_t first, uint32_t end, uint32_t row_bytes) {
    const uint32_t begin_bytes = (first / alignment_entries) * alignment_bytes;
    const uint32_t end_bytes = ((end + alignment_entries - 1) / alignment_entries) * alignment_bytes;
    return {begin_bytes, end_bytes < row_bytes ? end_bytes : row_bytes};
}

constexpr bool use_slot_tensor = get_compile_time_arg_val(7);
constexpr bool use_start_tensor = get_compile_time_arg_val(8);
constexpr bool use_end_tensor = get_compile_time_arg_val(9);
constexpr auto table_args = TensorAccessorArgs<10>();
constexpr auto allocated_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
constexpr auto free_args = TensorAccessorArgs<allocated_args.next_compile_time_args_offset()>();
constexpr auto count_args = TensorAccessorArgs<free_args.next_compile_time_args_offset()>();

constexpr auto slot_args = TensorAccessorArgs<count_args.next_compile_time_args_offset()>();
constexpr auto start_args = TensorAccessorArgs<slot_args.next_compile_time_args_offset()>();
constexpr auto end_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();

// Resolve scalar or device request values before entering the shared allocation path.
template <bool use_tensor, typename AccessorArgs>
uint32_t read_request_value(const Noc& noc, AccessorArgs accessor_args, uint32_t& rt_args_idx) {
    const uint32_t value = get_arg_val<uint32_t>(rt_args_idx++);
    if constexpr (use_tensor) {
        CircularBuffer cb_request(4);
        const CoreLocalMem<volatile uint32_t> scratch(cb_request.get_write_ptr());
        const auto accessor = TensorAccessor(accessor_args, value);
        noc.async_read(accessor, scratch, sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        // CBs sit at fixed L1 addresses reused by every invocation and trace replay, so the RISC data
        // cache may hold a prior call's value for this line (barrier orders the DMA, volatile still
        // reads cache). Force a refetch of the freshly written value.
        invalidate_l1_cache();
        return scratch[0];
    }
    return value;
}

struct MetadataBuffers {
    decltype(TensorAccessor(table_args, 0)) table_acc;
    decltype(TensorAccessor(allocated_args, 0)) allocated_acc;
    decltype(TensorAccessor(free_args, 0)) free_acc;
    decltype(TensorAccessor(count_args, 0)) count_acc;
    CoreLocalMem<volatile uint32_t> table;
    CoreLocalMem<volatile uint32_t> allocated;
    CoreLocalMem<volatile uint32_t> counts;
    uint32_t free_l1_address;
};

// Bind the four DRAM tensors and their L1 scratch buffers in host argument order.
MetadataBuffers make_metadata_buffers(uint32_t& rt_args_idx) {
    CircularBuffer cb_table(0);
    CircularBuffer cb_allocated(1);
    CircularBuffer cb_free(2);
    CircularBuffer cb_counts(3);
    return {
        TensorAccessor(table_args, get_arg_val<uint32_t>(rt_args_idx++)),
        TensorAccessor(allocated_args, get_arg_val<uint32_t>(rt_args_idx++)),
        TensorAccessor(free_args, get_arg_val<uint32_t>(rt_args_idx++)),
        TensorAccessor(count_args, get_arg_val<uint32_t>(rt_args_idx++)),
        CoreLocalMem<volatile uint32_t>(cb_table.get_write_ptr()),
        CoreLocalMem<volatile uint32_t>(cb_allocated.get_write_ptr()),
        CoreLocalMem<volatile uint32_t>(cb_counts.get_write_ptr()),
        cb_free.get_write_ptr()};
}

// Count this SP's pages in a slot's interleaved logical-page prefix.
uint32_t pages_on_sp(uint32_t pages, uint32_t sp) { return pages / sp_size + (sp < pages % sp_size); }

// A no-growth call reads only the cache line containing this slot's counter.
uint32_t read_allocated_pages(const Noc& noc, const MetadataBuffers& buffers, uint32_t slot, ByteRange range) {
    noc.async_read(
        buffers.allocated_acc,
        buffers.allocated,
        range_size(range),
        {.page_id = 0, .offset_bytes = range.begin},
        {.offset_bytes = range.begin});
    noc.async_read_barrier();
    invalidate_l1_cache();  // same fresh-metadata refetch as above
    return buffers.allocated[slot];
}

// Decode the request and read the selected slot's current allocation.
AllocationUpdate read_allocation_update(const Noc& noc, const MetadataBuffers& buffers, uint32_t& rt_args_idx) {
    const uint32_t slot = read_request_value<use_slot_tensor>(noc, slot_args, rt_args_idx);
    const bool reset = read_request_value<use_start_tensor>(noc, start_args, rt_args_idx) == 0;
    const uint32_t end = read_request_value<use_end_tensor>(noc, end_args, rt_args_idx);
    const auto allocated_range = aligned_range(slot, slot + 1, allocated_bytes);
    const uint32_t old_pages = read_allocated_pages(noc, buffers, slot, allocated_range);
    const uint32_t required_pages = end / page_size + (end % page_size != 0);
    const uint32_t next_pages = reset || required_pages > old_pages ? required_pages : old_pages;
    return {slot, reset, old_pages, next_pages, allocated_range};
}

// Plan only changed table entries and counter lines, splitting wrapped SP intervals.
MetadataRanges plan_metadata_ranges(const AllocationUpdate& update) {
    const uint32_t first_page = update.reset ? 0 : update.old_pages;
    const uint32_t last_page =
        update.reset && update.old_pages > update.next_pages ? update.old_pages : update.next_pages;
    MetadataRanges ranges{aligned_range(first_page, last_page, table_bytes), {0, 0}, {0, 0}};
    const uint32_t first_sp = first_page % sp_size;
    const uint32_t changed_pages = last_page - first_page;
    if (changed_pages >= sp_size) {
        ranges.counts = aligned_range(0, sp_size, counts_bytes);
    } else if (changed_pages > sp_size - first_sp) {
        ranges.counts = aligned_range(first_sp, sp_size, counts_bytes);
        ranges.wrapped_counts = aligned_range(0, changed_pages - (sp_size - first_sp), counts_bytes);
        if (ranges.wrapped_counts.end >= ranges.counts.begin) {
            ranges.counts.begin = 0;
            ranges.wrapped_counts = {0, 0};
        }
    } else {
        ranges.counts = aligned_range(first_sp, first_sp + changed_pages, counts_bytes);
    }
    return ranges;
}

// Growth overwrites every new table entry. Read only partial boundary lines;
// reset additionally reads the old live prefix so its IDs can be returned.
void read_table_boundaries(
    const Noc& noc,
    const MetadataBuffers& buffers,
    uint32_t slot,
    uint32_t old_pages,
    uint32_t next_pages,
    bool reset,
    ByteRange range) {
    uint32_t loaded_end = range.begin;
    if (reset && old_pages != 0) {
        loaded_end = aligned_range(0, old_pages, table_bytes).end;
        noc.async_read(buffers.table_acc, buffers.table, loaded_end, {.page_id = slot}, {});
    } else if (!reset && old_pages % alignment_entries != 0) {
        loaded_end = range.begin + alignment_bytes;
        noc.async_read(
            buffers.table_acc,
            buffers.table,
            alignment_bytes,
            {.page_id = slot, .offset_bytes = range.begin},
            {.offset_bytes = range.begin});
    }
    const uint32_t end_pages = reset && old_pages > next_pages ? old_pages : next_pages;
    const uint32_t tail = (end_pages / alignment_entries) * alignment_bytes;
    if (end_pages % alignment_entries != 0 && tail >= loaded_end) {
        noc.async_read(
            buffers.table_acc,
            buffers.table,
            range.end - tail,
            {.page_id = slot, .offset_bytes = tail},
            {.offset_bytes = tail});
    }
}

// Load preserved table entries and affected free counts before changing ownership.
void read_slot_metadata(
    const Noc& noc, const MetadataBuffers& buffers, const AllocationUpdate& update, const MetadataRanges& ranges) {
    read_table_boundaries(noc, buffers, update.slot, update.old_pages, update.next_pages, update.reset, ranges.table);
    noc.async_read(
        buffers.count_acc,
        buffers.counts,
        range_size(ranges.counts),
        {.page_id = 0, .offset_bytes = ranges.counts.begin},
        {.offset_bytes = ranges.counts.begin});
    if (range_size(ranges.wrapped_counts) != 0) {
        noc.async_read(
            buffers.count_acc,
            buffers.counts,
            range_size(ranges.wrapped_counts),
            {.page_id = 0, .offset_bytes = ranges.wrapped_counts.begin},
            {.offset_bytes = ranges.wrapped_counts.begin});
    }
    noc.async_read_barrier();
    invalidate_l1_cache();  // same fresh-metadata refetch as above
}

// Publish only the modified table range and counter cache lines.
void write_slot_metadata(
    const Noc& noc, const MetadataBuffers& buffers, const AllocationUpdate& update, const MetadataRanges& ranges) {
    buffers.allocated[update.slot] = update.next_pages;
    noc.async_write(
        buffers.table,
        buffers.table_acc,
        range_size(ranges.table),
        {.offset_bytes = ranges.table.begin},
        {.page_id = update.slot, .offset_bytes = ranges.table.begin});
    noc.async_write(
        buffers.allocated,
        buffers.allocated_acc,
        range_size(update.allocated_range),
        {.offset_bytes = update.allocated_range.begin},
        {.page_id = 0, .offset_bytes = update.allocated_range.begin});
    noc.async_write(
        buffers.counts,
        buffers.count_acc,
        range_size(ranges.counts),
        {.offset_bytes = ranges.counts.begin},
        {.page_id = 0, .offset_bytes = ranges.counts.begin});
    if (range_size(ranges.wrapped_counts) != 0) {
        noc.async_write(
            buffers.counts,
            buffers.count_acc,
            range_size(ranges.wrapped_counts),
            {.offset_bytes = ranges.wrapped_counts.begin},
            {.page_id = 0, .offset_bytes = ranges.wrapped_counts.begin});
    }
    noc.async_write_barrier();
}

struct FreeListWindow {
    CoreLocalMem<volatile uint32_t> data;
    ByteRange range;
    uint32_t sp;
    uint32_t offset = 0;
    uint32_t bytes = 0;
    bool dirty = false;
};

// Flush returned IDs before reusing the window. Popped IDs need no writeback.
void flush_free_list_window(const Noc& noc, const MetadataBuffers& buffers, FreeListWindow& window) {
    if (window.bytes == 0) {
        return;
    }
    if (window.dirty) {
        noc.async_write(
            window.data, buffers.free_acc, window.bytes, {}, {.page_id = window.sp, .offset_bytes = window.offset});
        noc.async_write_barrier();
    }
    window.bytes = 0;
    window.dirty = false;
}

// Load the required aligned free-list range, preserving neighbouring entries and padding.
volatile uint32_t& free_list_entry(
    const Noc& noc, const MetadataBuffers& buffers, FreeListWindow& window, uint32_t index) {
    const uint32_t window_begin = (index / free_window_entries) * free_window_bytes;
    const uint32_t offset = window_begin > window.range.begin ? window_begin : window.range.begin;
    if (window.bytes == 0 || offset != window.offset) {
        flush_free_list_window(noc, buffers, window);
        window.offset = offset;
        const uint32_t remaining = window.range.end - window_begin;
        const uint32_t window_size = remaining < free_window_bytes ? remaining : free_window_bytes;
        window.bytes = window_size - (offset - window_begin);
        noc.async_read(
            buffers.free_acc, window.data, window.bytes, {.page_id = window.sp, .offset_bytes = window.offset}, {});
        noc.async_read_barrier();
        invalidate_l1_cache();  // same fresh-metadata refetch as above
    }
    return window.data[index - window.offset / sizeof(uint32_t)];
}

void return_free_id(
    const Noc& noc, const MetadataBuffers& buffers, FreeListWindow& window, uint32_t index, uint32_t id) {
    free_list_entry(noc, buffers, window, index) = id;
    window.dirty = true;
}

// Return old IDs on reset, then allocate new IDs and update each SP's free count.
void update_free_lists(
    const Noc& noc, const MetadataBuffers& buffers, uint32_t old_pages, uint32_t next_pages, bool reset) {
    const uint32_t changed_pages = reset ? (old_pages > next_pages ? old_pages : next_pages) : next_pages - old_pages;
    const uint32_t affected_sps = changed_pages < sp_size ? changed_pages : sp_size;
    uint32_t sp = reset ? 0 : old_pages % sp_size;
    for (uint32_t n = 0; n < affected_sps; ++n, sp = sp + 1 == sp_size ? 0 : sp + 1) {
        const uint32_t old_count = pages_on_sp(old_pages, sp);
        const uint32_t next_count = pages_on_sp(next_pages, sp);
        if (reset ? old_count == 0 && next_count == 0 : old_count == next_count) {
            continue;
        }
        uint32_t count = buffers.counts[sp];
        const uint32_t stack_end = reset ? count + old_count : count;
        const uint32_t stack_begin =
            reset ? (next_count > old_count ? stack_end - next_count : count) : count - (next_count - old_count);
        FreeListWindow window{
            CoreLocalMem<volatile uint32_t>(buffers.free_l1_address),
            aligned_range(stack_begin, stack_end, free_row_bytes),
            sp};
        if (reset) {
            for (uint32_t i = 0; i < old_count; ++i) {
                const uint32_t page = sp + i * sp_size;
                return_free_id(noc, buffers, window, count++, buffers.table[page]);
                buffers.table[page] = 0;
            }
        }
        for (uint32_t i = reset ? 0 : old_count; i < next_count; ++i) {
            buffers.table[sp + i * sp_size] = free_list_entry(noc, buffers, window, --count);
        }
        flush_free_list_window(noc, buffers, window);
        buffers.counts[sp] = count;
    }
}
}  // namespace

// Apply one server-admitted request to this device's replicated metadata.
void kernel_main() {
    Noc noc;
    uint32_t rt_args_idx = 0;
    const auto buffers = make_metadata_buffers(rt_args_idx);
    const auto update = read_allocation_update(noc, buffers, rt_args_idx);
    if (!needs_update(update)) {
        return;
    }

    const auto ranges = plan_metadata_ranges(update);
    read_slot_metadata(noc, buffers, update, ranges);
    update_free_lists(noc, buffers, update.old_pages, update.next_pages, update.reset);
    write_slot_metadata(noc, buffers, update, ranges);
}
