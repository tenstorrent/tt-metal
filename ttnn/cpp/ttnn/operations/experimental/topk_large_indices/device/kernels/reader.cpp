// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
// Shared metadata read invalidates the reused L1 address before loading the next value.
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "topk_large_indices_metadata.hpp"
#include "topk_large_indices_runtime_args.hpp"

namespace {

namespace rl = topk_rescan_layout;

FORCE_INLINE uint32_t ordered_image(uint32_t u) {
    return u ^ (static_cast<uint32_t>(static_cast<int32_t>(u) >> 31) | 0x80000000u);
}

// Appends the block's band hits (index, ordered image) to the hit list. Positive: the band is in [+0, +inf], so a
// raw word is a candidate iff (v - raw_lo) < len; candidates are confirmed on the ordered image. Returns false when
// the list is full.
template <bool Positive>
FORCE_INLINE bool collect_band_hits(
    const uint32_t* vals,
    uint32_t count,
    uint32_t base_j,
    uint32_t lo,
    uint32_t len,
    volatile uint32_t* hits,
    uint32_t& n) {
    const auto off = [](uint32_t v) { return Positive ? v : ordered_image(v); };
    const uint32_t raw_lo = Positive ? lo - 0x80000000u : lo;
    const auto take = [&](uint32_t e) {
        const uint32_t o = ordered_image(vals[e]);
        if (o - lo < len) {
            if (n == rl::max_helper_hits) {
                return false;
            }
            hits[2 * n] = base_j + e;
            hits[2 * n + 1] = o;
            ++n;
        }
        return true;
    };
    uint32_t i = 0;
    const uint32_t end8 = count & ~7u;
    while (i < end8) {
        for (; i < end8; i += 8) {
            const uint32_t h = static_cast<uint32_t>(off(vals[i + 0]) - raw_lo < len) |
                               static_cast<uint32_t>(off(vals[i + 1]) - raw_lo < len) |
                               static_cast<uint32_t>(off(vals[i + 2]) - raw_lo < len) |
                               static_cast<uint32_t>(off(vals[i + 3]) - raw_lo < len) |
                               static_cast<uint32_t>(off(vals[i + 4]) - raw_lo < len) |
                               static_cast<uint32_t>(off(vals[i + 5]) - raw_lo < len) |
                               static_cast<uint32_t>(off(vals[i + 6]) - raw_lo < len) |
                               static_cast<uint32_t>(off(vals[i + 7]) - raw_lo < len);
            if (__builtin_expect(h != 0, 0)) {
                break;
            }
        }
        if (i == end8) {
            break;
        }
        for (uint32_t e = i; e < i + 8; ++e) {
            if (!take(e)) {
                return false;
            }
        }
        i += 8;
    }
    for (; i < count; ++i) {
        if (!take(i)) {
            return false;
        }
    }
    return true;
}

FORCE_INLINE bool deadline_passed(uint32_t deadline) {
    return static_cast<int32_t>(get_timestamp_32b() - deadline) >= 0;
}

// FP32 k == K exact row rescan, reader side: once its last row's chunks are read, the reader waits for the writer's
// post for that row. On go_job it scans blocks [first_block, end_block) of the row against the writer's band, under
// the writer's deadline, and hands the hits back. On go_skip it returns at once.
template <typename InputAccessorT>
void help_rescan_last_row(volatile uint32_t* mbox, uint32_t base, const Noc& noc, const InputAccessorT& input) {
    uint32_t go;
    while (true) {
        invalidate_l1_cache();
        go = mbox[rl::go];
        if (go != 0) {
            break;
        }
        riscv_wait(64);
    }
    if (go != rl::go_job) {
        return;
    }
    invalidate_l1_cache();
    const uint32_t lo = mbox[rl::lo];
    const uint32_t len = mbox[rl::len];
    const bool positive = mbox[rl::positive] != 0;
    const uint32_t first_block = mbox[rl::first_block];
    const uint32_t end_block = mbox[rl::end_block];
    const uint32_t deadline = mbox[rl::deadline];
    const uint32_t row = mbox[rl::row];
    const uint32_t search_len = mbox[rl::search_len];
    volatile uint32_t* hits = reinterpret_cast<volatile uint32_t*>(base + rl::hits);
    const uint32_t blocks = base + rl::reader_blocks;
    const uint32_t row_bytes = search_len * sizeof(uint32_t);
    const auto issue = [&](uint32_t b) {
        const uint32_t off = b * rl::block_bytes;
        const uint32_t bytes = row_bytes - off < rl::block_bytes ? row_bytes - off : rl::block_bytes;
        noc.async_read(
            input,
            CoreLocalMem<uint8_t>(blocks + ((b - first_block) & 1) * rl::block_bytes),
            bytes,
            {.page_id = row, .offset_bytes = off},
            {});
    };

    uint32_t n = 0;
    bool go_on = first_block < end_block;
    if (go_on) {
        issue(first_block);
        noc.async_read_barrier();
        invalidate_l1_cache();
        asm volatile("" ::: "memory");
        go_on = !deadline_passed(deadline);
    }
    for (uint32_t b = first_block; b < end_block && go_on; ++b) {
        const bool more = b + 1 < end_block;
        if (more) {
            issue(b + 1);
        }
        const uint32_t base_j = b * rl::block_elements;
        const uint32_t count =
            search_len - base_j < rl::block_elements ? search_len - base_j : rl::block_elements;
        const uint32_t* vals =
            reinterpret_cast<const uint32_t*>(blocks + ((b - first_block) & 1) * rl::block_bytes);
        for (uint32_t s = 0; s < count && go_on; s += rl::poll_elements) {
            const uint32_t c = count - s < rl::poll_elements ? count - s : rl::poll_elements;
            go_on = positive ? collect_band_hits<true>(vals + s, c, base_j + s, lo, len, hits, n)
                             : collect_band_hits<false>(vals + s, c, base_j + s, lo, len, hits, n);
            go_on = go_on && !deadline_passed(deadline);
        }
        if (more) {
            noc.async_read_barrier();
            invalidate_l1_cache();
            asm volatile("" ::: "memory");
        }
    }
    mbox[rl::hit_count] = n;
    asm volatile("fence" ::: "memory");
    mbox[rl::done] = 1;
}

}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_common_arg_val<uint32_t>(topk_common_args::input_address);
    const uint32_t start_row = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    uint32_t search_len = get_common_arg_val<uint32_t>(topk_common_args::search_length);
    const uint32_t input_page_bytes = get_common_arg_val<uint32_t>(topk_common_args::input_row_bytes);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
    // BFLOAT16 or FLOAT32 input: a 32x32 tile page holds 1024 elements.
    constexpr uint32_t element_bytes = tile_bytes / (32 * 32);
    constexpr uint32_t llk_k = chunk_bytes / element_bytes;
    // Input CB capacity in chunks (program factory cb_depth).
    constexpr uint32_t input_cb_depth_chunks = 2;
    constexpr auto input_args = TensorAccessorArgs<4>();
    constexpr uint32_t metadata_args_base = input_args.next_compile_time_args_offset();
    // THE metadata-mode selector for this kernel. One flag, not one per metadata argument: the host
    // validates the metadata tensors as a group, so inside this mode every metadata read is legal.
    constexpr bool metadata_mode = get_compile_time_arg_val(metadata_args_base) != 0;
    constexpr uint32_t meta_cb = get_compile_time_arg_val(metadata_args_base + 1);
    constexpr uint32_t meta_offset = get_compile_time_arg_val(metadata_args_base + 2);
    constexpr auto meta_args = TensorAccessorArgs<metadata_args_base + 3>();
    // Real-token-end block, appended after the accessor above so those offsets are untouched.
    constexpr uint32_t vend_args_base = meta_args.next_compile_time_args_offset();
    constexpr auto vend_args = TensorAccessorArgs<vend_args_base>();
    // FP32 k == K exact row rescan: this core's last row is scanned by the writer and the reader together.
    constexpr uint32_t rescan_args_base = vend_args.next_compile_time_args_offset();
    constexpr bool rescan_rows = get_compile_time_arg_val(rescan_args_base) != 0;
    constexpr uint32_t cb_rescan = get_compile_time_arg_val(rescan_args_base + 1);

    const auto input = TensorAccessor(input_args, src_addr, input_page_bytes);
    const uint32_t input_width = input_page_bytes / element_bytes;
    CircularBuffer input_cb(cb_in);
    Noc noc;

    TopkMetadataBounds bounds;
    if constexpr (metadata_mode) {
        CircularBuffer meta_cb_obj(meta_cb);
        meta_cb_obj.reserve_back(1);
        const uint32_t scratch = meta_cb_obj.get_write_ptr();
        const uint32_t metadata_length = trace_metadata::read_metadata_scalar_u32(
            noc, meta_args, get_common_arg_val<uint32_t>(topk_common_args::metadata_address), scratch);
        // Validate before addition so the offset cannot wrap and no malformed metadata can produce a NoC read
        // outside the input row. ASSERT is watcher-gated, so it compiles to nothing in a normal build; pair it
        // with a clamp, as bounded_kv_actual_isl / bounded_cache_batch_idx do, or the bound would hold only
        // under the watcher. Clamping to input_width also matches the score kernel's coupled derivation, which
        // saturates its own extent (kv_len_tiles is clamped to k_len_tiles) rather than reading past the row.
        const bool metadata_valid = meta_offset <= input_width && metadata_length <= input_width - meta_offset &&
                                    (metadata_length != 0 || meta_offset != 0);
        ASSERT(metadata_valid);
        search_len = metadata_valid ? metadata_length + meta_offset : input_width;
        // 0 = no bound supplied (full chunk): the host zeroes the slot rather than compiling a second
        // variant, so presence costs a compare here instead of a program hash split.
        const uint32_t valid_end_addr = get_common_arg_val<uint32_t>(topk_common_args::valid_end_address);
        if (valid_end_addr != 0) {
            // Same scratch page as the read above: the helper invalidates the L1 line before loading, which
            // is exactly what it documents for a reused address.
            const uint32_t valid_end =
                trace_metadata::read_metadata_scalar_u32(noc, vend_args, valid_end_addr, scratch);
            const uint32_t capped = ((valid_end + 31) / 32) * 32;
            if (capped < search_len) {
                search_len = capped;
            }
        }
        bounds = calculate_topk_bounds(search_len, llk_k);
        CoreLocalMem<TopkMetadataBounds> mailbox(scratch);
        mailbox->num_chunks = bounds.num_chunks;
        mailbox->tail_elements = bounds.tail_elements;
        clobber_all_memory();
        meta_cb_obj.push_back(1);
    } else {
        bounds = calculate_topk_bounds(search_len, llk_k);
    }
    const uint32_t tail_chunk_bytes = bounds.tail_elements * element_bytes;

    // The writer posts the last row's rescan job only after compute has consumed every chunk read below, so this
    // clear (stale words of the previous launch) always lands before the post.
    uint32_t rescan_base = 0;
    if constexpr (rescan_rows) {
        rescan_base = (CircularBuffer(cb_rescan).get_write_ptr() + 63) & ~63u;
        if (num_rows != 0) {
            reinterpret_cast<volatile uint32_t*>(rescan_base + topk_rescan_layout::mailbox)[topk_rescan_layout::go] =
                0;
        }
    }

    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const uint32_t row = start_row + local_row;
        for (uint32_t chunk = 0; chunk < bounds.num_chunks; ++chunk) {
            const uint32_t active_chunk_bytes = (chunk + 1 == bounds.num_chunks) ? tail_chunk_bytes : chunk_bytes;
            input_cb.reserve_back(tiles_per_chunk);
            for (uint32_t tile = 0; tile < tiles_per_chunk; ++tile) {
                const uint32_t tile_offset = tile * tile_bytes;
                const uint32_t read_bytes =
                    tile_offset < active_chunk_bytes
                        ? (active_chunk_bytes - tile_offset < tile_bytes ? active_chunk_bytes - tile_offset
                                                                         : tile_bytes)
                        : 0;
                if (read_bytes != 0) {
                    noc.async_read(
                        input,
                        input_cb,
                        read_bytes,
                        {.page_id = row, .offset_bytes = chunk * chunk_bytes + tile_offset},
                        {.offset_bytes = tile_offset});
                }
            }
            noc.async_read_barrier();
            input_cb.push_back(tiles_per_chunk);
            if (local_row == 0 && chunk == 0) {
                // Every core starts by reading its first row at the same moment. Requesting only chunk 0 then
                // (instead of the whole 2-chunk CB) halves that DRAM burst, so the cores served last get their
                // first chunk sooner. Compute pops chunk 0 right after copying it to DEST and needs chunk 1
                // only once chunk 0 is sorted, so chunk 1 is still read in time.
                input_cb.reserve_back(input_cb_depth_chunks * tiles_per_chunk);
            }
        }
    }

    if constexpr (rescan_rows) {
        if (num_rows != 0) {
            help_rescan_last_row(
                reinterpret_cast<volatile uint32_t*>(rescan_base + topk_rescan_layout::mailbox),
                rescan_base,
                noc,
                input);
        }
    }
}
