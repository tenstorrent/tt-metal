// ONLY-USES: MODE_SEQUENCED. The rest of this file is unused shared infra for other ports.
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified stick-level reader for interleaved RM tensors.
//
// Reads sticks (RM pages) from DRAM using noc_async_read.
//
// Named CT args: mode, cb_id, stick_bytes, aligned_page_size, seq_id
// Positional CT args: TensorAccessorArgs (index 0)
//
// Modes:
//   MODE_SEQUENTIAL (0): Coalesced stick reads, num_reads work units of
//     sticks_per_read sticks each. (reshape)
//   MODE_TILEROW (1): TILE_H sticks per tile-row, column chunking. (tilize)
//   MODE_NONALIGNED (2): 64B-aligned scratch reads. (reshape BH)
//   MODE_LASTDIM_REPEAT (3): Read stick, replicate N times in L1. (repeat W)
//   MODE_SEQUENCED (4): Batched stick reads with seq_id-dispatched address.
//     seq_id selects the address sequencer (same IDs as tile reader).
//   MODE_TILEROW_PAD (5): Pad-aware tile-row reader. NOC-reads the valid prefix
//     of each stick, fills column-pad bytes from a packed pad value, and emits
//     whole pad tile-rows for height/outer padding. (tilize_with_val_padding)
//   MODE_PARTIAL_READ (6): Read one page's aligned partial slab, then spread
//     each stick_bytes element onto an L1_ALIGN-aligned slot for scatter
//     writeback. (reshape scatter parallelization)
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "sequencers.h"

constexpr uint32_t MODE_SEQUENTIAL = 0;  // Legacy: coalesced stick reads (reshape)
constexpr uint32_t MODE_TILEROW = 1;
constexpr uint32_t MODE_NONALIGNED = 2;
constexpr uint32_t MODE_LASTDIM_REPEAT = 3;
constexpr uint32_t MODE_SEQUENCED = 4;     // Seq_id dispatched stick reads
constexpr uint32_t MODE_TILEROW_PAD = 5;   // Pad-aware tile-row reader (tilize_with_val_padding)
constexpr uint32_t MODE_PARTIAL_READ = 6;  // Partial page read at aligned byte offset (reshape scatter)

// fill_with_val: write n_bytes of a packed value into L1 starting at start_addr.
// Ported from tt-metal reader_unary_pad_dims_split_rows_multicore.cpp. Writes 4
// bytes at a time and handles the unaligned head/tail when val_size < 4.
template <uint32_t val_size>
FORCE_INLINE void fill_with_val(uint32_t start_addr, uint32_t n_bytes, uint32_t val) {
    static_assert(val_size == 2 || val_size == 4, "Unsupported val_size");
    using IntType = std::conditional_t<(val_size == 2), uint16_t, uint32_t>;

    uint32_t end_addr = start_addr + n_bytes;
    uint32_t start_addr_4B = (start_addr + 0x3) & 0xFFFFFFFC;  // ceil aligned to 4B
    uint32_t end_addr_4B = end_addr & 0xFFFFFFFC;              // floor aligned to 4B

    {
        auto* start_ptr_4B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_addr_4B);
        auto* end_ptr_4B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(end_addr_4B);
        for (auto* ptr = start_ptr_4B; ptr < end_ptr_4B; ++ptr) {
            *ptr = val;
        }
    }

    if constexpr (val_size < 4) {
        auto* start_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(start_addr);
        auto* end_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(end_addr);
        auto* start_ptr_4B = reinterpret_cast<volatile tt_l1_ptr IntType*>(start_addr_4B);
        auto* end_ptr_4B = reinterpret_cast<volatile tt_l1_ptr IntType*>(end_addr_4B);
        IntType val_ = static_cast<IntType>(val);
        for (auto* ptr = start_ptr; ptr < start_ptr_4B; ++ptr) {
            *ptr = val_;
        }
        for (auto* ptr = end_ptr_4B; ptr < end_ptr; ++ptr) {
            *ptr = val_;
        }
    }
}

// ── RT arg structs ──────────────────────────────────────────────────

// MODE_SEQUENCED: base header, sequencer-specific fields follow
struct ArgsStickBase {
    uint32_t src_addr;
    uint32_t num_sticks;
    uint32_t start_id;
    // [3..] sequencer-specific (same as tile reader)
};

struct ArgsStickRepeat : ArgsStickBase {
    uint32_t num_repeats;
    uint32_t lower_pages;
    uint32_t rep_dim_pages;
};

struct ArgsStickSlice : ArgsStickBase {
    uint32_t num_dims;
    // followed by: num_unpadded[D], num_padded[D], id_per_dim[D]
};

struct ArgsStickHc : ArgsStickBase {
    uint32_t curr_c;
    uint32_t curr_h;
    uint32_t curr_n;
    uint32_t C;
    uint32_t H;
};

struct ArgsStickCn : ArgsStickBase {
    uint32_t N;
    uint32_t HtWt;
    uint32_t batch_step;
    uint32_t channel_step;
    uint32_t hw;
    uint32_t n;
};

struct ArgsStickPad : ArgsStickBase {
    uint32_t start_h;
    uint32_t start_c;
    uint32_t start_n;
    uint32_t H_in;
    uint32_t C_in;
    uint32_t N_in;
    uint32_t H_out;
    uint32_t C_out;
    uint32_t N_out;
    uint32_t in_stick_bytes;
    uint32_t out_stick_bytes;
    uint32_t out_stick_aligned;
    uint32_t pad_w_back_bytes;
    uint32_t packed_pad_val;
    uint32_t cb_pad;
    uint32_t front_pad_w_bytes;
    uint32_t front_h;
    uint32_t front_c;
    uint32_t front_n;
};

struct ArgsStickConcat : ArgsStickBase {
    uint32_t src_addr_1;
    uint32_t start_tensor;
    uint32_t start_tensor_id;
    uint32_t page_id_0;
    uint32_t page_id_1;
    uint32_t ppb_0;
    uint32_t ppb_1;
};

// MODE_TILEROW
struct ArgsStickTilerow {
    uint32_t src_addr;
    uint32_t num_tile_rows;
    uint32_t start_stick;
    uint32_t H_per_tile;
    uint32_t chunk_Wt;
    uint32_t num_col_chunks;
    uint32_t elem_w_bytes;
};

// MODE_TILEROW_PAD: header + tightly-packed block-rep schedule.
// Layout mirrors tt-metal reader_unary_pad_dims_split_rows_multicore.cpp:
//   [src_addr, padded_X_size, packed_pad_value, start_page_id, n_block_reps]
//   then n_block_reps * {n_data, n_mixed, n_pads, times, repeat_count}.
struct ArgsStickTilerowPad {
    uint32_t src_addr;
    uint32_t padded_X_size;  // padded last-dim bytes (= padded_X * elem_size)
    uint32_t packed_pad_value;
    uint32_t start_page_id;
    uint32_t n_block_reps;
    // followed by n_block_reps * 5 uint32_t {n_data, n_mixed, n_pads, times, repeat_count}
};

// MODE_NONALIGNED
struct ArgsStickNonaligned {
    uint32_t src_addr;
    uint32_t num_reads;
    uint32_t sticks_per_read;
    uint32_t start_stick;
    uint32_t old_stick_size;
    uint32_t new_stick_size;
    uint32_t ratio;
    uint32_t split;
    uint32_t cb_scratch_id;
};

// MODE_LASTDIM_REPEAT
struct ArgsStickLastdimRepeat {
    uint32_t src_addr;
    uint32_t num_pages;
    uint32_t start_page;
    uint32_t in_stick_size;
    uint32_t num_repeats;
};

// ── Shared transport loop for simple stick sequencers ───────────────

template <typename Accessor, typename State, typename NextFn>
FORCE_INLINE void read_sticks(
    uint32_t cb_id,
    uint32_t stick_bytes,
    uint32_t batch,
    const Accessor& accessor,
    uint32_t num_sticks,
    State& state,
    NextFn next_fn) {
    uint32_t left = num_sticks;
    while (left > 0) {
        uint32_t b = (left < batch) ? left : batch;
        cb_reserve_back(cb_id, b);
        uint32_t l1 = get_write_ptr(cb_id);
        for (uint32_t t = 0; t < b; t++) {
            uint64_t noc_addr = get_noc_addr(next_fn(state), accessor);
            noc_async_read(noc_addr, l1, stick_bytes);
            l1 += stick_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id, b);
        left -= b;
    }
}

// ── Main ────────────────────────────────────────────────────────────

void kernel_main() {
    constexpr uint32_t MODE = get_named_compile_time_arg_val("mode");
    constexpr uint32_t cb_id = get_named_compile_time_arg_val("cb_id");
    constexpr uint32_t stick_bytes = get_named_compile_time_arg_val("stick_bytes");
    constexpr uint32_t aligned_page_size = get_named_compile_time_arg_val("aligned_page_size");
    constexpr auto src_args = TensorAccessorArgs<0>();

    // ── MODE_SEQUENTIAL: coalesced stick reads (reshape) ───────────
    // Reads num_reads work units of sticks_per_read sticks each.
    // CB push count can differ from read count (coalescing ratio).
    if constexpr (MODE == MODE_SEQUENTIAL) {
        struct ArgsSeq {
            uint32_t src_addr;
            uint32_t num_reads;
            uint32_t sticks_per_read;
            uint32_t sticks_per_cb_push;
            uint32_t start_stick;
        };
        const auto* a = reinterpret_cast<const ArgsSeq*>(get_arg_addr(0));
        const auto s = TensorAccessor(src_args, a->src_addr, aligned_page_size);
        uint32_t i_stick = a->start_stick;
        for (uint32_t iter = 0; iter < a->num_reads; ++iter) {
            cb_reserve_back(cb_id, a->sticks_per_cb_push);
            uint32_t l1_addr = get_write_ptr(cb_id);
            for (uint32_t i = 0; i < a->sticks_per_read; ++i) {
                uint64_t noc_addr = get_noc_addr(i_stick, s);
                noc_async_read(noc_addr, l1_addr, stick_bytes);
                l1_addr += stick_bytes;
                i_stick++;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id, a->sticks_per_cb_push);
        }
    }

    // ── MODE_SEQUENCED: batched stick reads with sequencer dispatch ──
    else if constexpr (MODE == MODE_SEQUENCED) {
        constexpr uint32_t SEQ_ID = get_named_compile_time_arg_val("seq_id");
        constexpr uint32_t BATCH = get_named_compile_time_arg_val("batch");
        const auto* base = reinterpret_cast<const ArgsStickBase*>(get_arg_addr(0));
        const auto s = TensorAccessor(src_args, base->src_addr, aligned_page_size);

        if constexpr (SEQ_ID == SEQ_IDENTITY) {
            auto st = seq_identity_init(base->start_id);
            read_sticks(cb_id, stick_bytes, BATCH, s, base->num_sticks, st, seq_identity_next);
        } else if constexpr (SEQ_ID == SEQ_REPEAT) {
            const auto* a = reinterpret_cast<const ArgsStickRepeat*>(get_arg_addr(0));
            auto st = seq_repeat_init(a->start_id, a->num_repeats, a->lower_pages, a->rep_dim_pages);
            read_sticks(cb_id, stick_bytes, BATCH, s, a->num_sticks, st, seq_repeat_next);
        } else if constexpr (SEQ_ID == SEQ_SLICE) {
            const auto* a = reinterpret_cast<const ArgsStickSlice*>(get_arg_addr(0));
            tt_l1_ptr uint32_t* num_unpadded = (tt_l1_ptr uint32_t*)(&a->num_dims + 1);
            tt_l1_ptr uint32_t* num_padded = num_unpadded + a->num_dims;
            tt_l1_ptr uint32_t* id_per_dim = num_padded + a->num_dims;
            auto st = seq_slice_init(a->start_id, a->num_dims, num_unpadded, num_padded, id_per_dim);
            read_sticks(cb_id, stick_bytes, BATCH, s, a->num_sticks, st, seq_slice_next);
        } else if constexpr (SEQ_ID == SEQ_HC) {
            const auto* a = reinterpret_cast<const ArgsStickHc*>(get_arg_addr(0));
            auto st = seq_hc_init(a->start_id, a->curr_c, a->curr_h, a->curr_n, a->C, a->H);
            read_sticks(cb_id, stick_bytes, BATCH, s, a->num_sticks, st, seq_hc_next);
        } else if constexpr (SEQ_ID == SEQ_CN) {
            const auto* a = reinterpret_cast<const ArgsStickCn*>(get_arg_addr(0));
            auto st = seq_cn_init(a->start_id, a->hw, a->n, a->N, a->HtWt, a->batch_step, a->channel_step);
            read_sticks(cb_id, stick_bytes, BATCH, s, a->num_sticks, st, seq_cn_next);
        } else if constexpr (SEQ_ID == SEQ_PAD) {
            // PAD: conditional read (source vs fill) — custom loop
            const auto* a = reinterpret_cast<const ArgsStickPad*>(get_arg_addr(0));
            // Fill pad stick in scratch CB
            {
                volatile tt_l1_ptr uint32_t* ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(a->cb_pad));
                for (uint32_t i = 0; i < a->out_stick_bytes / 4 + 1; ++i) {
                    ptr[i] = a->packed_pad_val;
                }
            }
            uint64_t pad_noc_addr = get_noc_addr(get_read_ptr(a->cb_pad));

            // front_* = 0 reproduces this legacy stick-pad branch's original
            // back-only is_data exactly (W front pad is applied via
            // front_pad_w_bytes in the loop below, not the tile-coord window).
            // The production RM pad path uses reader_pad_rm_interleaved.cpp.
            auto pad_st = seq_pad_init(
                a->start_id,
                0,
                a->start_h,
                a->start_c,
                a->start_n,
                0,
                a->H_in,
                a->C_in,
                a->N_in,
                0,
                a->H_out,
                a->C_out,
                a->N_out,
                0,
                0,
                0,
                0);

            uint32_t src_stick = a->start_id;
            uint32_t left = a->num_sticks;
            while (left > 0) {
                uint32_t b = (left < BATCH) ? left : BATCH;
                cb_reserve_back(cb_id, b);
                uint32_t l1 = get_write_ptr(cb_id);
                for (uint32_t t = 0; t < b; t++) {
                    seq_pad_next(pad_st);
                    if (pad_st.is_data) {
                        // Data stick: read from DRAM, pad W dimension in L1
                        uint64_t noc_addr = get_noc_addr(src_stick, s);
                        // Write front pad
                        if (a->front_pad_w_bytes > 0) {
                            noc_async_read(pad_noc_addr, l1, a->front_pad_w_bytes);
                        }
                        // Write data
                        noc_async_read(noc_addr, l1 + a->front_pad_w_bytes, a->in_stick_bytes);
                        // Write back pad
                        if (a->pad_w_back_bytes > 0) {
                            noc_async_read(
                                pad_noc_addr, l1 + a->front_pad_w_bytes + a->in_stick_bytes, a->pad_w_back_bytes);
                        }
                        src_stick++;
                    } else {
                        // Full pad stick
                        noc_async_read(pad_noc_addr, l1, a->out_stick_bytes);
                    }
                    l1 += a->out_stick_aligned;
                }
                noc_async_read_barrier();
                cb_push_back(cb_id, b);
                left -= b;
            }
        } else if constexpr (SEQ_ID == SEQ_CONCAT) {
            // CONCAT: dual accessor — custom loop
            const auto* a = reinterpret_cast<const ArgsStickConcat*>(get_arg_addr(0));
            const auto s1 = TensorAccessor(src_args, a->src_addr_1, aligned_page_size);
            auto st =
                seq_concat_init(a->start_tensor, a->start_tensor_id, a->page_id_0, a->page_id_1, a->ppb_0, a->ppb_1);
            uint32_t left = a->num_sticks;
            while (left > 0) {
                uint32_t b = (left < BATCH) ? left : BATCH;
                cb_reserve_back(cb_id, b);
                uint32_t l1 = get_write_ptr(cb_id);
                for (uint32_t t = 0; t < b; t++) {
                    uint32_t read_tensor = st.curr_tensor;
                    uint32_t src_page = seq_concat_next(st);
                    if (read_tensor == 0) {
                        uint64_t noc_addr = get_noc_addr(src_page, s);
                        noc_async_read(noc_addr, l1, stick_bytes);
                    } else {
                        uint64_t noc_addr = get_noc_addr(src_page, s1);
                        noc_async_read(noc_addr, l1, stick_bytes);
                    }
                    l1 += stick_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_id, b);
                left -= b;
            }
        }
    }

    // ── MODE_TILEROW: TILE_H sticks per tile-row, optional col chunks ─
    else if constexpr (MODE == MODE_TILEROW) {
        const auto* a = reinterpret_cast<const ArgsStickTilerow*>(get_arg_addr(0));
        const auto s = TensorAccessor(src_args, a->src_addr, aligned_page_size);
        const uint32_t chunk_read_bytes = a->chunk_Wt * a->elem_w_bytes;
        uint32_t i_stick = a->start_stick;

        if (a->num_col_chunks == 1) {
            for (uint32_t tr = 0; tr < a->num_tile_rows; ++tr) {
                cb_reserve_back(cb_id, a->chunk_Wt);
                uint32_t l1_addr = get_write_ptr(cb_id);
                for (uint32_t h = 0; h < a->H_per_tile; ++h) {
                    uint64_t noc_addr = get_noc_addr(i_stick, s);
                    noc_async_read(noc_addr, l1_addr, chunk_read_bytes);
                    l1_addr += chunk_read_bytes;
                    i_stick++;
                }
                noc_async_read_barrier();
                cb_push_back(cb_id, a->chunk_Wt);
            }
        } else {
            for (uint32_t tr = 0; tr < a->num_tile_rows; ++tr) {
                uint64_t base_noc[32];
                for (uint32_t h = 0; h < a->H_per_tile; ++h) {
                    base_noc[h] = get_noc_addr(i_stick + h, s);
                }
                for (uint32_t c = 0; c < a->num_col_chunks; ++c) {
                    cb_reserve_back(cb_id, a->chunk_Wt);
                    uint32_t l1_addr = get_write_ptr(cb_id);
                    for (uint32_t h = 0; h < a->H_per_tile; ++h) {
                        noc_async_read(base_noc[h], l1_addr, chunk_read_bytes);
                        l1_addr += chunk_read_bytes;
                        base_noc[h] += chunk_read_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_id, a->chunk_Wt);
                }
                i_stick += a->H_per_tile;
            }
        }
    }

    // ── MODE_NONALIGNED: scratch CB + 64B alignment extraction ───────
    // Batched: issue up to NABATCH reads to separate scratch slots,
    // single barrier, then extract all. Reduces barrier count by NABATCH×.
    else if constexpr (MODE == MODE_NONALIGNED) {
        constexpr uint32_t NABATCH = get_named_compile_time_arg_val("nabatch");
        const auto* a = reinterpret_cast<const ArgsStickNonaligned*>(get_arg_addr(0));
        const auto s = TensorAccessor(src_args, a->src_addr, aligned_page_size);

        cb_reserve_back(a->cb_scratch_id, 1);
        uint32_t scratch_addr = get_write_ptr(a->cb_scratch_id);
        uint32_t i_stick = a->start_stick;
        const uint32_t read_size = aligned_page_size + 64;

        if (a->split) {
            // old > new: each old stick produces ratio new sticks.
            // Batch CB reserves by ratio to reduce semaphore overhead.
            uint32_t remaining = a->num_reads;
            const uint32_t new_half = a->new_stick_size / 2;
            const uint32_t cb_page_bytes = (a->new_stick_size + 15) & ~15u;  // L1-aligned
            while (remaining > 0) {
                uint32_t batch = (remaining < NABATCH) ? remaining : NABATCH;
                uint32_t byte_offsets[NABATCH];
                for (uint32_t b = 0; b < batch; ++b) {
                    uint64_t noc_addr = get_noc_addr(i_stick + b, s);
                    byte_offsets[b] = noc_addr & 63;
                    noc_async_read(noc_addr & ~uint64_t(63), scratch_addr + b * read_size, read_size);
                }
                noc_async_read_barrier();
                for (uint32_t b = 0; b < batch; ++b) {
                    // Reserve all ratio pages at once (1 semaphore check vs ratio)
                    cb_reserve_back(cb_id, a->ratio);
                    uint32_t cb_base = get_write_ptr(cb_id);
                    uint16_t* src_ptr = reinterpret_cast<uint16_t*>(scratch_addr + b * read_size + byte_offsets[b]);
                    for (uint32_t r = 0; r < a->ratio; ++r) {
                        uint16_t* dst_ptr = reinterpret_cast<uint16_t*>(cb_base + r * cb_page_bytes);
                        uint32_t off = r * new_half;
                        for (uint32_t w = 0; w < new_half; ++w) {
                            dst_ptr[w] = src_ptr[off + w];
                        }
                    }
                    cb_push_back(cb_id, a->ratio);
                }
                i_stick += batch;
                remaining -= batch;
            }
        } else {
            // new >= old: ratio old sticks pack into one new stick.
            const uint32_t old_half = a->old_stick_size / 2;
            for (uint32_t iter = 0; iter < a->num_reads; ++iter) {
                cb_reserve_back(cb_id, 1);
                uint16_t* cb_dst = reinterpret_cast<uint16_t*>(get_write_ptr(cb_id));
                uint32_t dst_off = 0;
                uint32_t spr = a->sticks_per_read;
                uint32_t done = 0;
                while (done < spr) {
                    uint32_t batch = ((spr - done) < NABATCH) ? (spr - done) : NABATCH;
                    uint32_t byte_offsets[NABATCH];
                    for (uint32_t b = 0; b < batch; ++b) {
                        uint64_t noc_addr = get_noc_addr(i_stick + b, s);
                        byte_offsets[b] = noc_addr & 63;
                        noc_async_read(noc_addr & ~uint64_t(63), scratch_addr + b * read_size, read_size);
                    }
                    noc_async_read_barrier();
                    for (uint32_t b = 0; b < batch; ++b) {
                        uint16_t* src_ptr = reinterpret_cast<uint16_t*>(scratch_addr + b * read_size + byte_offsets[b]);
                        for (uint32_t w = 0; w < old_half; ++w) {
                            cb_dst[dst_off + w] = src_ptr[w];
                        }
                        dst_off += old_half;
                    }
                    i_stick += batch;
                    done += batch;
                }
                cb_push_back(cb_id, 1);
            }
        }
    }

    // ── MODE_LASTDIM_REPEAT: read stick, replicate N times in L1 ────
    else if constexpr (MODE == MODE_LASTDIM_REPEAT) {
        constexpr uint32_t BATCH = get_named_compile_time_arg_val("batch");
        const auto* a = reinterpret_cast<const ArgsStickLastdimRepeat*>(get_arg_addr(0));
        const auto s = TensorAccessor(src_args, a->src_addr, aligned_page_size);

        // CB page = in_stick_size * num_repeats (full output stick)
        const uint32_t out_stick_size = a->in_stick_size * a->num_repeats;
        uint32_t src_page = a->start_page;
        uint32_t pages_left = a->num_pages;

        while (pages_left > 0) {
            uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
            cb_reserve_back(cb_id, batch);
            uint32_t l1_base = get_write_ptr(cb_id);

            // Read original sticks from DRAM
            for (uint32_t t = 0; t < batch; t++) {
                uint64_t noc_addr = get_noc_addr(src_page, s);
                noc_async_read(noc_addr, l1_base + t * out_stick_size, a->in_stick_size);
                src_page++;
            }
            noc_async_read_barrier();

            // Replicate each stick N-1 times via L1-to-L1 NOC copies
            if (a->num_repeats > 1) {
                for (uint32_t t = 0; t < batch; t++) {
                    uint32_t l1_addr = l1_base + t * out_stick_size;
                    uint64_t src_noc = get_noc_addr(l1_addr);
                    for (uint32_t r = 1; r < a->num_repeats; r++) {
                        noc_async_read(src_noc, l1_addr + r * a->in_stick_size, a->in_stick_size);
                    }
                }
                noc_async_read_barrier();
            }

            cb_push_back(cb_id, batch);
            pages_left -= batch;
        }
    }

    // ── MODE_TILEROW_PAD: pad-aware tile-row reader ─────────────────
    // Ports tt-metal reader_unary_pad_dims_split_rows_multicore.cpp.
    // Reads the valid prefix of each stick, fills column padding from a packed
    // pad value, and emits whole pad tile-rows for height/outer padding.
    else if constexpr (MODE == MODE_TILEROW_PAD) {
        constexpr uint32_t tile_height = get_named_compile_time_arg_val("tile_height");
        constexpr uint32_t tile_row_shift_bits = get_named_compile_time_arg_val("tile_row_shift_bits");
        constexpr uint32_t num_pages_in_row = get_named_compile_time_arg_val("num_pages_in_row");
        constexpr uint32_t unpadded_X_size = get_named_compile_time_arg_val("unpadded_X_bytes");
        constexpr uint32_t valid_last_page_bytes = get_named_compile_time_arg_val("valid_last_page_bytes");
        // page_size for the TensorAccessor: full padded/unaligned input page size.
        constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");

        const auto* a = reinterpret_cast<const ArgsStickTilerowPad*>(get_arg_addr(0));
        const auto s = TensorAccessor(src_args, a->src_addr, page_size);

        const uint32_t padded_X_size = a->padded_X_size;
        const uint32_t pad_value = a->packed_pad_value;
        const uint32_t num_tiles_per_row = padded_X_size >> tile_row_shift_bits;

        // pad_blocks: emit `num_blocks` whole pad tile-rows.
        auto pad_blocks = [&](uint32_t num_blocks) {
            for (uint32_t i = 0; i < num_blocks; i++) {
                cb_reserve_back(cb_id, num_tiles_per_row);
                uint32_t l1_write_addr = get_write_ptr(cb_id);
                // MODE_TILEROW_PAD is unreachable for this op (permute always sets mode =
                // MODE_SEQUENCED); the toolchain still fully compiles this branch (see the
                // factory comment on named_compile_time_args), and fill_with_val<elem_size>
                // fails template-argument substitution here, so the val_size is hardcoded.
                fill_with_val<4>(l1_write_addr, padded_X_size << 5, pad_value);
                cb_push_back(cb_id, num_tiles_per_row);
            }
        };

        // read_block: read `num_rows` data sticks into one tile-row, padding the
        // column tail of each row and any remaining (mixed) rows with pad_value.
        auto read_block = [&](uint32_t base_page_id, uint32_t num_rows) {
            uint32_t padding_rows = (tile_height - num_rows) & 31;
            bool has_rows = (num_rows + padding_rows) > 0;

            cb_reserve_back(cb_id, num_tiles_per_row * has_rows);
            uint32_t l1_write_addr = get_write_ptr(cb_id);
            for (uint32_t k = 0; k < num_rows; k++) {
                uint32_t start_of_row_l1_write_addr = l1_write_addr;
                for (uint32_t i = 0; i < num_pages_in_row - 1; i++) {
                    uint64_t noc_addr = get_noc_addr(base_page_id + k * num_pages_in_row + i, s);
                    noc_async_read(noc_addr, l1_write_addr, page_size);
                    l1_write_addr += page_size;
                }
                // Last page in the row may carry column padding.
                uint64_t noc_addr = get_noc_addr(base_page_id + k * num_pages_in_row + num_pages_in_row - 1, s);
                noc_async_read(noc_addr, l1_write_addr, valid_last_page_bytes);
                uint32_t size_of_padding_columns = padded_X_size - unpadded_X_size;
                fill_with_val<4>(start_of_row_l1_write_addr + unpadded_X_size, size_of_padding_columns, pad_value);
                l1_write_addr += valid_last_page_bytes + size_of_padding_columns;
            }
            fill_with_val<4>(l1_write_addr, padding_rows * padded_X_size, pad_value);
            noc_async_read_barrier();
            cb_push_back(cb_id, num_tiles_per_row * has_rows);
        };

        uint32_t page_id = a->start_page_id;
        const uint32_t n_block_reps = a->n_block_reps;
        const tt_l1_ptr uint32_t* reps = reinterpret_cast<const tt_l1_ptr uint32_t*>(&a->n_block_reps + 1);

        constexpr uint32_t N_DATA_IDX = 0;
        constexpr uint32_t N_MIXED_IDX = 1;
        constexpr uint32_t N_PADS_IDX = 2;
        constexpr uint32_t TIMES_IDX = 3;
        constexpr uint32_t REPEAT_CT_IDX = 4;
        constexpr uint32_t NUM_RT = 5;

        uint32_t rep_off = 0;
        uint32_t count = 1;
        for (uint32_t block_rep_idx = 0; block_rep_idx < n_block_reps; ++block_rep_idx) {
            const uint32_t repeat_count = reps[rep_off + REPEAT_CT_IDX];
            const uint32_t n_data = reps[rep_off + N_DATA_IDX];
            const uint32_t n_mixed = reps[rep_off + N_MIXED_IDX];
            const uint32_t n_pads = reps[rep_off + N_PADS_IDX];
            const uint32_t times = reps[rep_off + TIMES_IDX];
            if (count == repeat_count) {
                rep_off += NUM_RT;
                count = 1;
            } else {
                count++;
            }
            for (uint32_t t = 0; t < times; ++t) {
                for (uint32_t y_t = 0; y_t < n_data; y_t++) {
                    read_block(page_id, tile_height);
                    page_id += tile_height * num_pages_in_row;
                }
                read_block(page_id, n_mixed);
                page_id += n_mixed * num_pages_in_row;
                pad_blocks(n_pads);
            }
        }
    }

    // ── MODE_PARTIAL_READ: read ONE page's aligned partial slab into the CB ──
    // (reshape scatter parallelization: each core reads its group's byte-range of
    // the single source page at an aligned col_off; the normal writer then scatters
    // that slab into the group's output sticks). src_page + col_off + nbytes are all
    // DRAM-aligned by construction in build_reshape_rm_factory.
    else if constexpr (MODE == MODE_PARTIAL_READ) {
        // Degenerate SCATTER: read this core's aligned `nbytes` slab from ONE
        // source page at `col_off`, then SPREAD each `stick_bytes` element onto an
        // L1_ALIGN-aligned slot. The spread is mandatory: the writer scatters each
        // element to a distinct 16B-aligned output page, and noc_async_write
        // corrupts unless (src_L1 % L1_ALIGN) == (dst % L1_ALIGN) — a packed slab
        // (stride stick_bytes < L1_ALIGN) would floor the L1 source to the 16B
        // boundary and replicate every element L1_ALIGN/stick_bytes times.
        constexpr uint32_t L1_ALIGN = 16;
        struct ArgsPartial {
            uint32_t src_addr;
            uint32_t src_page;
            uint32_t col_off;
            uint32_t nbytes;
        };
        const auto* a = reinterpret_cast<const ArgsPartial*>(get_arg_addr(0));
        const auto s = TensorAccessor(src_args, a->src_addr, aligned_page_size);
        cb_reserve_back(cb_id, 1);
        uint32_t l1 = get_write_ptr(cb_id);
        uint64_t noc = get_noc_addr(a->src_page, s) + a->col_off;
        noc_async_read(noc, l1, a->nbytes);
        noc_async_read_barrier();
        // Spread packed -> 16B-strided, in place, HIGH index first so the write of
        // slot i (offset i*L1_ALIGN) never clobbers a not-yet-read packed source
        // (offset i'*stick_bytes for i' < i, since i*L1_ALIGN > i'*stick_bytes).
        const uint32_t n_elems = a->nbytes / stick_bytes;
        for (uint32_t i = n_elems; i-- > 0;) {
            volatile tt_l1_ptr uint8_t* dst = (volatile tt_l1_ptr uint8_t*)(l1 + i * L1_ALIGN);
            volatile tt_l1_ptr uint8_t* srcp = (volatile tt_l1_ptr uint8_t*)(l1 + i * stick_bytes);
            for (uint32_t b = 0; b < stick_bytes; ++b) {
                dst[b] = srcp[b];
            }
        }
        cb_push_back(cb_id, 1);
    }
}
