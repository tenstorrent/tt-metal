// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_runtime_args.hpp"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {

// pack_untilize emits 16-element slices in face-pair order:
// [top-left, bottom-left, top-right, bottom-right] for each tile column.
template <uint32_t source_slices_per_row>
FORCE_INLINE uint32_t src_slice_of(uint32_t dst_slice) {
    const uint32_t tile_col = dst_slice >> 2;
    const uint32_t face_col = dst_slice & 0x1;
    const uint32_t face_row_offset = (dst_slice & 0x2) ? source_slices_per_row / 2 : 0;
    return (2 * tile_col) + face_col + face_row_offset;
}

template <uint32_t source_slices_per_row, uint32_t output_slices_per_row, uint32_t slice_bytes>
FORCE_INLINE void copy_row_to_scratch(CircularBuffer& src_cb, CircularBuffer& scratch_cb, const Noc& noc) {
    static_assert(source_slices_per_row == 64 || source_slices_per_row == 128);
    static_assert(output_slices_per_row >= 1 && output_slices_per_row <= source_slices_per_row);
    // Output slices 2m and 2m + 1 (same tile column and face row, face columns 0 and 1) are
    // adjacent in the source too, so each pair moves as one transfer: half the NoC requests.
    constexpr uint32_t pair_bytes = 2 * slice_bytes;
    static_assert(pair_bytes <= NOC_MAX_BURST_SIZE);
    constexpr uint32_t output_pairs = output_slices_per_row / 2;

    const uint32_t src_base = src_cb.get_read_ptr();
    const uint32_t dst_base = scratch_cb.get_write_ptr();
    const uint32_t noc_id = noc.get_noc_id();
    const auto local_src = [noc_id](uint32_t addr) {
        return noc_traits_t<UnicastEndpoint>::src_args_type{
            .noc_x = static_cast<uint32_t>(my_x[noc_id]), .noc_y = static_cast<uint32_t>(my_y[noc_id]), .addr = addr};
    };

    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{}, pair_bytes, local_src(src_base));

    for (uint32_t pair = 0; pair < output_pairs; ++pair) {
        const uint32_t dst_slice = 2 * pair;
        const uint32_t src_addr = src_base + src_slice_of<source_slices_per_row>(dst_slice) * slice_bytes;
        const uint32_t dst_addr = dst_base + dst_slice * slice_bytes;
        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(dst_addr),
            pair_bytes,
            local_src(src_addr),
            {.offset_bytes = 0});
    }
    if constexpr (output_slices_per_row % 2 != 0) {
        constexpr uint32_t dst_slice = output_slices_per_row - 1;
        const uint32_t src_addr = src_base + src_slice_of<source_slices_per_row>(dst_slice) * slice_bytes;
        noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{}, slice_bytes, local_src(src_addr));
        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(dst_base + dst_slice * slice_bytes),
            slice_bytes,
            local_src(src_addr),
            {.offset_bytes = 0});
    }
    noc.async_read_barrier();
}

// FP32 input with k < K: compute packs the whole sorted top-K survivor, but the key is truncated, so the
// output can cut a key-tie bucket at rank k in the wrong place (a kept x below the FP32 k-th value, a
// dropped one above it). Every such miss is a tie at the boundary, and the K - k spare survivor ranks hold
// the rest of that bucket. Re-rank survivor ranks [k - W, k + W) by their exact input values (read back
// from the input row) and keep the top W at ranks [k - W, k). The result is a selection of distinct
// survivors, still in descending order (exact inside the window, key order outside it).
constexpr uint32_t boundary_window_max = 16;
constexpr uint32_t boundary_read_bytes = 64;  // DRAM-aligned read holding the candidate's FP32 word

template <
    uint32_t source_slices_per_row,
    uint32_t output_slices_per_row,
    uint32_t slice_bytes,
    typename InputAccessorT>
FORCE_INLINE void rerank_boundary_exact(
    CircularBuffer& src_cb,
    CircularBuffer& scratch_cb,
    uint32_t values_base,
    const Noc& noc,
    const InputAccessorT& input,
    uint32_t row,
    uint32_t input_width) {
    constexpr uint32_t slice_elements = slice_bytes / sizeof(uint32_t);
    constexpr uint32_t k = output_slices_per_row * slice_elements;
    constexpr uint32_t K = source_slices_per_row * slice_elements;
    constexpr uint32_t spare = k < K ? K - k : 0;
    constexpr uint32_t W0 = spare < boundary_window_max ? spare : boundary_window_max;
    constexpr uint32_t W = k < W0 ? k : W0;
    constexpr uint32_t N = 2 * W;
    if constexpr (W == 0) {
        return;
    } else {
        // The scratch row was just written by NoC reads and the indices page by the packer.
        invalidate_l1_cache();
        const uint32_t src_base = src_cb.get_read_ptr();
        volatile uint32_t* scratch = reinterpret_cast<volatile uint32_t*>(scratch_cb.get_write_ptr());
        uint32_t idx[N];
        uint32_t n = 0;
        for (uint32_t j = 0; j < N; ++j) {
            const uint32_t r = k - W + j;
            uint32_t v;
            if (j < W) {
                v = scratch[r];
            } else {
                const uint32_t addr = src_base +
                                      src_slice_of<source_slices_per_row>(r / slice_elements) * slice_bytes +
                                      (r % slice_elements) * sizeof(uint32_t);
                v = *reinterpret_cast<volatile uint32_t*>(addr);
            }
            // Survivors are ordered, so the first sentinel (-inf / invalid, 0xFFFFFFFF) ends the candidates.
            if (v >= input_width) {
                break;
            }
            idx[n++] = v;
        }
        if (n <= W) {
            return;
        }

        for (uint32_t j = 0; j < n; ++j) {
            noc.async_read(
                input,
                CoreLocalMem<uint8_t>(values_base + j * boundary_read_bytes),
                boundary_read_bytes,
                {.page_id = row, .offset_bytes = (idx[j] * sizeof(uint32_t)) & ~(boundary_read_bytes - 1)},
                {});
        }
        noc.async_read_barrier();
        invalidate_l1_cache();

        // Order-preserving integer image of the FP32 value (descending sort on it = descending by value).
        uint32_t key[N];
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t u = *reinterpret_cast<volatile uint32_t*>(
                values_base + j * boundary_read_bytes + ((idx[j] * sizeof(uint32_t)) & (boundary_read_bytes - 1)));
            if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
                return;  // NaN: leave the row exactly as ranked by the key
            }
            key[j] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        }
        // Stable insertion sort, descending: equal values keep the key order.
        for (uint32_t i = 1; i < n; ++i) {
            const uint32_t ki = key[i];
            const uint32_t ii = idx[i];
            uint32_t j = i;
            while (j > 0 && key[j - 1] < ki) {
                key[j] = key[j - 1];
                idx[j] = idx[j - 1];
                --j;
            }
            key[j] = ki;
            idx[j] = ii;
        }
        for (uint32_t j = 0; j < W; ++j) {
            scratch[k - W + j] = idx[j];
        }
    }
}

// FP32 input with k == K: the survivor has no spare ranks, so the dropped part of the boundary key-tie bucket is
// gone and only a pass over the row's exact values can recover it. Window = survivor ranks [K - W, K) in exact
// order, m = its minimum, U = the exact value of rank K - W - 1. A dropped element's key is <= rank K - 1's, so it
// can only beat m with x <= U (unless the tie bucket is longer than W ranks): stream the row and test each raw word
// against the band (m, U] with one unsigned compare. Hits are the band's own survivors plus the real outsiders, and
// an outsider replaces the window minimum. The window is written back in descending exact order.
constexpr uint32_t rescan_window = 32;
constexpr uint32_t rescan_band_ranks = rescan_window + 1;
constexpr uint32_t rescan_block_elements = topk_rescan_layout::block_elements;
constexpr uint32_t rescan_block_bytes = topk_rescan_layout::block_bytes;
// Hits outside the band's survivors are checked against the whole returned row; past this many the row is
// left as it is (many equal values around the boundary).
constexpr uint32_t rescan_max_row_checks = 32;
// Every row's blocks are scanned in spans of topk_rescan_layout::poll_elements; between spans the scan checks its
// deadline and, on a row that is not the core's last, whether the next row's survivor has arrived (the scan then
// stops at once, so it never delays the next row). That bound also makes rows of <= 4 windows safe to rescan.

FORCE_INLINE uint32_t ordered_image(uint32_t u) {
    return u ^ (static_cast<uint32_t>(static_cast<int32_t>(u) >> 31) | 0x80000000u);
}

FORCE_INLINE bool deadline_passed(uint32_t deadline) {
    return static_cast<int32_t>(get_timestamp_32b() - deadline) >= 0;
}

// Blackhole's RISC-V L1 data cache is off by default (firmware configure_l1_data_cache), so every scan load is an
// uncached L1 access. The scan turns it on only while it streams the row blocks, which are written by NoC reads that
// complete before the fence that follows every read barrier. Same CSR sequence as set_l1_data_cache<true>() (cache
// bit 3 cleared, HW periodic invalidation off with bit 24); off restores the firmware defaults after a fence.
FORCE_INLINE void scan_cache_on() {
    asm volatile(
        "fence\n"
        "li t1, 0x8\n"
        "csrrc zero, 0x7c0, t1\n"
        "li t1, 0x1\n"
        "slli t1, t1, 24\n"
        "fence\n"
        "csrrs zero, 0x7c0, t1\n" ::
            : "t1", "memory");
}

FORCE_INLINE void scan_cache_off() {
    asm volatile(
        "fence\n"
        "li t1, 0x8\n"
        "csrrs zero, 0x7c0, t1\n"
        "li t1, 0x1\n"
        "slli t1, t1, 24\n"
        "csrrc zero, 0x7c0, t1\n"
        "fence\n" ::
            : "t1", "memory");
}

struct RescanWindow {
    uint32_t key[rescan_window];  // ordered images, descending
    uint32_t idx[rescan_window];
    uint32_t band_idx[rescan_band_ranks];  // survivor ranks [K - W - 1, K): the band's own members
    uint32_t upper;                        // ordered image of U (inclusive band top)
    uint32_t lo;                           // band = ordered images in [lo, lo + len)
    uint32_t len;
    uint32_t row_checks;
};

// Returns false when the row should not be scanned any further.
template <uint32_t K>
FORCE_INLINE bool rescan_offer(RescanWindow& w, volatile uint32_t* scratch, uint32_t o, uint32_t j) {
    if (o - w.lo >= w.len) {
        return true;
    }
    for (uint32_t r = 0; r < rescan_band_ranks; ++r) {
        if (w.band_idx[r] == j) {
            return true;
        }
    }
    if (++w.row_checks > rescan_max_row_checks) {
        return false;
    }
    // Band hits that are survivors sit just above the band ranks (their key is close to the boundary key), so the
    // search runs from rank K - B - 1 down: a member is found in a few loads, only real outsiders pay the full pass.
    for (uint32_t r = K - rescan_band_ranks; r-- > 0;) {
        if (scratch[r] == j) {
            return true;
        }
    }
    // A real outsider above the window minimum: sorted insert, dropping the minimum.
    uint32_t p = rescan_window - 1;
    while (p > 0 && w.key[p - 1] < o) {
        w.key[p] = w.key[p - 1];
        w.idx[p] = w.idx[p - 1];
        --p;
    }
    w.key[p] = o;
    w.idx[p] = j;
    w.lo = w.key[rescan_window - 1] + 1;
    w.len = w.upper - w.key[rescan_window - 1];
    return w.len != 0;
}

// Scans one block; Positive: the band is in [+0, +inf], so a raw word is in it iff (v - raw_lo) < len (negative
// words and NaNs wrap past len). Otherwise every word goes through the ordered image first.
template <uint32_t K, bool Positive>
FORCE_INLINE bool rescan_block(
    RescanWindow& w, volatile uint32_t* scratch, const uint32_t* vals, uint32_t count, uint32_t base_j) {
    const auto off = [](uint32_t v) { return Positive ? v : ordered_image(v); };
    uint32_t i = 0;
    const uint32_t end8 = count & ~7u;
    while (i < end8) {
        const uint32_t lo = Positive ? w.lo - 0x80000000u : w.lo;
        const uint32_t len = w.len;
        for (; i < end8; i += 8) {
            const uint32_t h = static_cast<uint32_t>(off(vals[i + 0]) - lo < len) |
                               static_cast<uint32_t>(off(vals[i + 1]) - lo < len) |
                               static_cast<uint32_t>(off(vals[i + 2]) - lo < len) |
                               static_cast<uint32_t>(off(vals[i + 3]) - lo < len) |
                               static_cast<uint32_t>(off(vals[i + 4]) - lo < len) |
                               static_cast<uint32_t>(off(vals[i + 5]) - lo < len) |
                               static_cast<uint32_t>(off(vals[i + 6]) - lo < len) |
                               static_cast<uint32_t>(off(vals[i + 7]) - lo < len);
            if (__builtin_expect(h != 0, 0)) {
                break;
            }
        }
        if (i == end8) {
            break;
        }
        for (uint32_t e = i; e < i + 8; ++e) {
            if (!rescan_offer<K>(w, scratch, ordered_image(vals[e]), base_j + e)) {
                return false;
            }
        }
        i += 8;
    }
    for (; i < count; ++i) {
        if (!rescan_offer<K>(w, scratch, ordered_image(vals[i]), base_j + i)) {
            return false;
        }
    }
    return true;
}

// mbox != nullptr (the core's last row; the reader is idle by then): the reader scans the second half of the row's
// blocks against the same band and hands its hits back through the mailbox. Returns true when the job was posted to
// the reader (it then has been waited for); otherwise the caller posts go_skip.
template <uint32_t K, typename InputAccessorT>
bool rescan_row_band(
    volatile uint32_t* scratch,
    uint32_t values_base,
    uint32_t blocks_base,
    const Noc& noc,
    const InputAccessorT& input,
    uint32_t row,
    uint32_t search_len,
    uint32_t deadline,
    volatile uint32_t* mbox,
    const CircularBuffer* next_row_cb) {
    namespace rl = topk_rescan_layout;
    constexpr uint32_t W = rescan_window;
    constexpr uint32_t B = rescan_band_ranks;
    static_assert(B <= K);
    const auto stop = [&]() {
        return deadline_passed(deadline) || (next_row_cb != nullptr && next_row_cb->pages_available_at_front(1));
    };
    if (stop()) {
        return false;
    }
    RescanWindow w;
    invalidate_l1_cache();
    for (uint32_t r = 0; r < B; ++r) {
        w.band_idx[r] = scratch[K - B + r];
        // Sentinel / out of range in the survivor: every finite value is already selected.
        if (w.band_idx[r] >= search_len) {
            return false;
        }
    }
    for (uint32_t r = 0; r < B; ++r) {
        noc.async_read(
            input,
            CoreLocalMem<uint8_t>(values_base + r * boundary_read_bytes),
            boundary_read_bytes,
            {.page_id = row, .offset_bytes = (w.band_idx[r] * sizeof(uint32_t)) & ~(boundary_read_bytes - 1)},
            {});
    }
    // The row's first block streams in while the window is sorted.
    const uint32_t row_bytes = search_len * sizeof(uint32_t);
    const uint32_t num_blocks = (search_len + rescan_block_elements - 1) / rescan_block_elements;
    noc.async_read(
        input,
        CoreLocalMem<uint8_t>(blocks_base),
        row_bytes < rescan_block_bytes ? row_bytes : rescan_block_bytes,
        {.page_id = row, .offset_bytes = 0},
        {});
    noc.async_read_barrier();
    invalidate_l1_cache();
    asm volatile("" ::: "memory");

    uint32_t band_key[B];
    for (uint32_t r = 0; r < B; ++r) {
        const uint32_t u = *reinterpret_cast<volatile uint32_t*>(
            values_base + r * boundary_read_bytes + ((w.band_idx[r] * sizeof(uint32_t)) & (boundary_read_bytes - 1)));
        if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
            return false;  // NaN: leave the row exactly as ranked by the key
        }
        band_key[r] = ordered_image(u);
    }
    // Window = ranks [K - W, K), stable insertion sort, descending: equal values keep the key order.
    for (uint32_t j = 0; j < W; ++j) {
        const uint32_t kj = band_key[1 + j];
        const uint32_t ij = w.band_idx[1 + j];
        uint32_t p = j;
        while (p > 0 && w.key[p - 1] < kj) {
            w.key[p] = w.key[p - 1];
            w.idx[p] = w.idx[p - 1];
            --p;
        }
        w.key[p] = kj;
        w.idx[p] = ij;
    }
    w.upper = band_key[0] > w.key[0] ? band_key[0] : w.key[0];
    w.lo = w.key[W - 1] + 1;
    w.len = w.upper - w.key[W - 1];
    w.row_checks = 0;

    bool go = w.len != 0;
    uint32_t own_blocks = num_blocks;
    bool posted = false;
    if (mbox != nullptr && go && num_blocks >= 2) {
        own_blocks = (num_blocks + 1) / 2;
        mbox[rl::done] = 0;
        mbox[rl::lo] = w.lo;
        mbox[rl::len] = w.len;
        mbox[rl::positive] = w.key[W - 1] >= 0x80000000u ? 1 : 0;
        mbox[rl::first_block] = own_blocks;
        mbox[rl::end_block] = num_blocks;
        mbox[rl::deadline] = deadline;
        mbox[rl::row] = row;
        mbox[rl::search_len] = search_len;
        asm volatile("fence" ::: "memory");
        mbox[rl::go] = rl::go_job;
        posted = true;
    }
    if (go) {
        scan_cache_on();
    }
    for (uint32_t b = 0; b < own_blocks && go; ++b) {
        const uint32_t cur = blocks_base + (b & 1) * rescan_block_bytes;
        const bool more = b + 1 < own_blocks;
        if (more) {
            const uint32_t off = (b + 1) * rescan_block_bytes;
            const uint32_t bytes = row_bytes - off < rescan_block_bytes ? row_bytes - off : rescan_block_bytes;
            noc.async_read(
                input,
                CoreLocalMem<uint8_t>(blocks_base + ((b + 1) & 1) * rescan_block_bytes),
                bytes,
                {.page_id = row, .offset_bytes = off},
                {});
        }
        const uint32_t base_j = b * rescan_block_elements;
        const uint32_t count =
            search_len - base_j < rescan_block_elements ? search_len - base_j : rescan_block_elements;
        const uint32_t* vals = reinterpret_cast<const uint32_t*>(cur);
        for (uint32_t s = 0; s < count && go; s += rl::poll_elements) {
            const uint32_t n = count - s < rl::poll_elements ? count - s : rl::poll_elements;
            go = w.key[W - 1] >= 0x80000000u ? rescan_block<K, true>(w, scratch, vals + s, n, base_j + s)
                                             : rescan_block<K, false>(w, scratch, vals + s, n, base_j + s);
            go = go && !stop();
        }
        if (more && go) {
            noc.async_read_barrier();
            invalidate_l1_cache();
            asm volatile("" ::: "memory");
        }
    }
    // A stop can leave the next block's read in flight: it is drained by the next read barrier (the next row's
    // boundary-value reads, or the barrier at kernel end) before its slot is reused.
    scan_cache_off();
    if (posted) {
        // The reader stops by the same deadline; its hits went through the band filter of the initial window,
        // rescan_offer re-checks them against the current one.
        while (true) {
            invalidate_l1_cache();
            if (mbox[rl::done] != 0) {
                break;
            }
            riscv_wait(32);
        }
        invalidate_l1_cache();
        const uint32_t n = mbox[rl::hit_count];
        const volatile uint32_t* hits = mbox + (rl::hits - rl::mailbox) / sizeof(uint32_t);
        for (uint32_t i = 0; i < n && go; ++i) {
            go = rescan_offer<K>(w, scratch, hits[2 * i + 1], hits[2 * i]);
        }
    }
    for (uint32_t j = 0; j < W; ++j) {
        scratch[K - W + j] = w.idx[j];
    }
    return posted;
}

template <
    uint32_t source_slices_per_row,
    uint32_t output_slices_per_row,
    uint32_t slice_bytes,
    typename TensorAccessorT,
    typename BeforePopT>
FORCE_INLINE void issue_reordered_row_write(
    CircularBuffer& src_cb,
    CircularBuffer& scratch_cb,
    const Noc& noc,
    const TensorAccessorT& tensor,
    uint32_t row,
    uint32_t row_bytes,
    BeforePopT&& before_pop) {
    src_cb.wait_front(1);
    scratch_cb.reserve_back(1);
    copy_row_to_scratch<source_slices_per_row, output_slices_per_row, slice_bytes>(src_cb, scratch_cb, noc);
    before_pop();
    src_cb.pop_front(1);

    scratch_cb.push_back(1);
    scratch_cb.wait_front(1);
    noc.async_write(scratch_cb, tensor, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
}

template <typename TensorAccessorT>
FORCE_INLINE void issue_contiguous_row_write(
    CircularBuffer& src_cb, const Noc& noc, const TensorAccessorT& tensor, uint32_t row, uint32_t row_bytes) {
    src_cb.wait_front(1);
    noc.async_write(src_cb, tensor, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
}

}  // namespace

void kernel_main() {
    const uint32_t indices_addr = get_common_arg_val<uint32_t>(topk_common_args::output_address);
    const uint32_t start_row = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_indices = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t indices_page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t source_slices_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t output_slices_per_row = get_compile_time_arg_val(4);
    constexpr uint32_t indices_slice_bytes = get_compile_time_arg_val(5);
    constexpr auto indices_args = TensorAccessorArgs<6>();
    // Exact boundary re-rank (FP32 input, k < K only): values CB and the input row accessor.
    constexpr uint32_t rerank_args_base = indices_args.next_compile_time_args_offset();
    constexpr bool rerank_boundary = get_compile_time_arg_val(rerank_args_base) != 0;
    constexpr uint32_t cb_boundary_values = get_compile_time_arg_val(rerank_args_base + 1);
    constexpr auto input_args = TensorAccessorArgs<rerank_args_base + 2>();
    // Exact row rescan (FP32 input, k == K, scalar search length only): double-buffered row blocks CB.
    constexpr uint32_t rescan_args_base = input_args.next_compile_time_args_offset();
    constexpr bool rescan_rows = get_compile_time_arg_val(rescan_args_base) != 0;
    constexpr uint32_t cb_rescan = get_compile_time_arg_val(rescan_args_base + 1);

    const auto indices = TensorAccessor(indices_args, indices_addr, indices_page_bytes);
    CircularBuffer indices_cb(cb_indices);
    Noc noc;

    if constexpr (source_slices_per_row == 32) {
        for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
            const uint32_t row = start_row + local_row;
            issue_contiguous_row_write(indices_cb, noc, indices, row, indices_page_bytes);
            noc.async_writes_flushed();
            indices_cb.pop_front(1);
        }
    } else {
        CircularBuffer indices_scratch_cb(cb_indices_scratch);
        if constexpr (rescan_rows) {
            // Deadlines come from this core's row-0 arrival time t0 (one row of compute plus start-up), which no
            // scan can inflate. Row r's scan stops by ~0.9 * t0 * (r + 2), before row r + 1 arrives. The last row
            // of a core with fewer rows stops by ~0.9 * t0 * max_rows, before the busiest core ends. A busiest
            // core's last row ends the kernel, so it is never scanned. On a one-row core of a long-row case the
            // last-row deadline is ~0.95 * t0 * max_rows (see below).
            const uint32_t t_start = get_timestamp_32b();
            const uint32_t max_rows = get_arg_val<uint32_t>(2);
            const uint32_t input_row_bytes = get_common_arg_val<uint32_t>(topk_common_args::writer_input_row_bytes);
            const uint32_t search_len = get_common_arg_val<uint32_t>(topk_common_args::writer_search_length);
            const auto input = TensorAccessor(
                input_args, get_common_arg_val<uint32_t>(topk_common_args::writer_input_address), input_row_bytes);
            CircularBuffer values_cb(cb_boundary_values);
            const uint32_t values_base =
                (values_cb.get_write_ptr() + boundary_read_bytes - 1) & ~(boundary_read_bytes - 1);
            CircularBuffer rescan_cb(cb_rescan);
            const uint32_t rescan_base = (rescan_cb.get_write_ptr() + 63) & ~63u;
            const uint32_t blocks_base = rescan_base + topk_rescan_layout::writer_blocks;
            volatile uint32_t* mbox = reinterpret_cast<volatile uint32_t*>(rescan_base + topk_rescan_layout::mailbox);
            constexpr uint32_t K = source_slices_per_row * (indices_slice_bytes / sizeof(uint32_t));
            uint32_t t0 = 0;
            for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
                const uint32_t row = start_row + local_row;
                indices_cb.wait_front(1);
                if (local_row == 0) {
                    t0 = get_timestamp_32b() - t_start;
                }
                indices_scratch_cb.reserve_back(1);
                copy_row_to_scratch<source_slices_per_row, output_slices_per_row, indices_slice_bytes>(
                    indices_cb, indices_scratch_cb, noc);
                indices_cb.pop_front(1);
                const bool last_row = local_row + 1 == num_rows;
                const bool ends_kernel = num_rows >= max_rows && last_row;
                bool posted = false;
                if (ends_kernel) {
                    mbox[topk_rescan_layout::go] = topk_rescan_layout::go_skip;
                } else {
                    const uint32_t horizon =
                        local_row + 1 == num_rows || local_row + 2 >= max_rows ? max_rows : local_row + 2;
                    // A one-row core's only row: E ~ s + max_rows * R + tail and t0 = s + R, so for rows long enough
                    // that R >> s (more than 8 windows) 61/64 * max_rows * t0 still lands ~0.05 * max_rows * R before
                    // the busiest core's end. The last-row scan polls it every span, so the overshoot is ~1 us.
                    const uint32_t frac = num_rows == 1 && search_len > 8 * K ? 61 : 58;
                    const uint32_t deadline =
                        t_start + static_cast<uint32_t>((static_cast<uint64_t>(t0) * horizon * frac) >> 6);
                    posted = rescan_row_band<K>(
                        reinterpret_cast<volatile uint32_t*>(indices_scratch_cb.get_write_ptr()),
                        values_base,
                        blocks_base,
                        noc,
                        input,
                        row,
                        search_len,
                        deadline,
                        // The reader split only on rows of > 8 windows: on short rows its post-deadline drain and
                        // the writer's merge of its hits have no slack before the busiest core's end.
                        last_row && search_len > 8 * K ? mbox : nullptr,
                        last_row ? nullptr : &indices_cb);
                    if (last_row && !posted) {
                        mbox[topk_rescan_layout::go] = topk_rescan_layout::go_skip;
                    }
                }
                indices_scratch_cb.push_back(1);
                indices_scratch_cb.wait_front(1);
                noc.async_write(
                    indices_scratch_cb,
                    indices,
                    indices_page_bytes,
                    {.offset_bytes = 0},
                    {.page_id = row, .offset_bytes = 0});
                noc.async_writes_flushed();
                indices_scratch_cb.pop_front(1);
            }
            noc.async_read_barrier();
            noc.async_write_barrier();
            return;
        }
        if constexpr (rerank_boundary) {
            const uint32_t input_row_bytes = get_common_arg_val<uint32_t>(topk_common_args::writer_input_row_bytes);
            const auto input = TensorAccessor(
                input_args, get_common_arg_val<uint32_t>(topk_common_args::writer_input_address), input_row_bytes);
            const uint32_t input_width = input_row_bytes / sizeof(float);
            CircularBuffer values_cb(cb_boundary_values);
            const uint32_t values_base =
                (values_cb.get_write_ptr() + boundary_read_bytes - 1) & ~(boundary_read_bytes - 1);
            for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
                const uint32_t row = start_row + local_row;
                issue_reordered_row_write<source_slices_per_row, output_slices_per_row, indices_slice_bytes>(
                    indices_cb, indices_scratch_cb, noc, indices, row, indices_page_bytes, [&]() {
                        rerank_boundary_exact<source_slices_per_row, output_slices_per_row, indices_slice_bytes>(
                            indices_cb, indices_scratch_cb, values_base, noc, input, row, input_width);
                    });

                noc.async_writes_flushed();
                indices_scratch_cb.pop_front(1);
            }
            noc.async_write_barrier();
            return;
        }
        for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
            const uint32_t row = start_row + local_row;
            issue_reordered_row_write<source_slices_per_row, output_slices_per_row, indices_slice_bytes>(
                indices_cb, indices_scratch_cb, noc, indices, row, indices_page_bytes, []() {});

            noc.async_writes_flushed();
            indices_scratch_cb.pop_front(1);
        }
    }

    noc.async_write_barrier();
}
