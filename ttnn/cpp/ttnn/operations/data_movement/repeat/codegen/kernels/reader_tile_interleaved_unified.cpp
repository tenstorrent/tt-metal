// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified batched tile reader for interleaved tensors.
//
// One transport loop, pluggable address sequencer selected by "seq_id".
// The sequencer maps output_page -> source_page. All sequencer logic
// lives in sequencers.h as FORCE_INLINE functions.
//
// CT args:
//   Named:      seq_id, cb_id, batch
//   Positional: TensorAccessorArgs (starts at index 0)
//
// RT args (flat uint32_t array, read as struct via get_arg_addr):
//   [0] src_addr    — source buffer address
//   [1] num_pages   — pages this core reads
//   [2] start_id    — starting page ID (meaning varies by sequencer)
//   [3..] sequencer-specific params (see SeqArgs structs in sequencers.h)
#include "api/dataflow/dataflow_api.h"
#include "sequencers.h"

// ── RT arg structs ──────────────────────────────────────────────────
// Laid out to match the Python builder's RT arg packing exactly.
// All structs share a common 3-field header.

struct ArgsBase {
    uint32_t src_addr;
    uint32_t num_pages;
    uint32_t start_id;
};

struct ArgsRepeat : ArgsBase {
    uint32_t num_repeats;
    uint32_t lower_pages;
    uint32_t rep_dim_pages;
};

struct ArgsSlice : ArgsBase {
    uint32_t num_dims;
    // followed by: num_unpadded[D], num_padded[D], id_per_dim[D]
    // (variable length — access via pointer arithmetic from &num_dims + 1)
};

struct ArgsPermute : ArgsBase {
    uint32_t num_dims;
    // followed by: src_strides[D], out_shape[D], inv_perm[D], id_per_dim[D]
};

struct ArgsTransposeWh : ArgsBase {
    uint32_t start_ht;
    uint32_t start_wt;
    uint32_t Ht;
    uint32_t Wt;
    uint32_t HtWt;
};

struct ArgsPad : ArgsBase {
    uint32_t start_wt;
    uint32_t start_ht;
    uint32_t start_c;
    uint32_t start_n;
    uint32_t Wt_in;
    uint32_t Ht_in;
    uint32_t C_in;
    uint32_t N_in;
    uint32_t Wt_out;
    uint32_t Ht_out;
    uint32_t C_out;
    uint32_t N_out;
    uint32_t tile_bytes;
    uint32_t packed_pad_val;
    uint32_t cb_pad;
    uint32_t front_wt;  // tile-aligned leading pad, W (tiles)
    uint32_t front_ht;  // tile-aligned leading pad, H (tiles)
    uint32_t front_c;   // leading pad, C
    uint32_t front_n;   // leading pad, N
};

struct ArgsConcat : ArgsBase {
    uint32_t src_addr_1;
    uint32_t start_tensor;
    uint32_t start_tensor_id;
    uint32_t page_id_0;
    uint32_t page_id_1;
    uint32_t ppb_0;
    uint32_t ppb_1;
};

// ── Transport loop (written ONCE) ───────────────────────────────────
// For simple sequencers (identity, repeat, slice, permute, transpose_wh):
//   each page = noc_async_read_tile(next_page, accessor, l1)
// PAD and CONCAT have different loop bodies (conditional reads / 2 accessors).

template <typename Accessor, typename State, typename NextFn>
FORCE_INLINE void read_pages(
    uint32_t cb_id,
    uint32_t BATCH,
    uint32_t page_size,
    const Accessor& accessor,
    uint32_t num_pages,
    State& state,
    NextFn next_fn) {
    uint32_t pages_left = num_pages;
    while (pages_left > 0) {
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        cb_reserve_back(cb_id, batch);
        uint32_t l1_addr = get_write_ptr(cb_id);
        for (uint32_t t = 0; t < batch; t++) {
            noc_async_read_tile(next_fn(state), accessor, l1_addr);
            l1_addr += page_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id, batch);
        pages_left -= batch;
    }
}

// ── Main ────────────────────────────────────────────────────────────

void kernel_main() {
    // Named CT args
    constexpr uint32_t SEQ_ID = get_named_compile_time_arg_val("seq_id");
    constexpr uint32_t cb_id = get_named_compile_time_arg_val("cb_id");
    constexpr uint32_t BATCH = get_named_compile_time_arg_val("batch");

    // Positional CT args: TensorAccessorArgs only (index 0)
    constexpr auto src_args = TensorAccessorArgs<0>();

    // RT args base (common header)
    const auto* base = reinterpret_cast<const ArgsBase*>(get_arg_addr(0));
    const uint32_t page_size = get_tile_size(cb_id);
    const auto s = TensorAccessor(src_args, base->src_addr, page_size);

    // ── IDENTITY ────────────────────────────────────────────────────
    if constexpr (SEQ_ID == SEQ_IDENTITY) {
        auto st = seq_identity_init(base->start_id);
        read_pages(cb_id, BATCH, page_size, s, base->num_pages, st, seq_identity_next);
    }

    // ── REPEAT ──────────────────────────────────────────────────────
    else if constexpr (SEQ_ID == SEQ_REPEAT) {
        const auto* a = reinterpret_cast<const ArgsRepeat*>(get_arg_addr(0));
        auto st = seq_repeat_init(a->start_id, a->num_repeats, a->lower_pages, a->rep_dim_pages);
        read_pages(cb_id, BATCH, page_size, s, a->num_pages, st, seq_repeat_next);
    }

    // ── REPEAT_INTERLEAVE (per-element AABB replication) ─────────────
    // Reuses ArgsRepeat — only the addressing function differs.
    else if constexpr (SEQ_ID == SEQ_REPEAT_INTERLEAVE) {
        const auto* a = reinterpret_cast<const ArgsRepeat*>(get_arg_addr(0));
        auto st = seq_repeat_interleave_init(a->start_id, a->num_repeats, a->lower_pages, a->rep_dim_pages);
        read_pages(cb_id, BATCH, page_size, s, a->num_pages, st, seq_repeat_interleave_next);
    }

    // ── SLICE ───────────────────────────────────────────────────────
    else if constexpr (SEQ_ID == SEQ_SLICE) {
        const auto* a = reinterpret_cast<const ArgsSlice*>(get_arg_addr(0));
        const uint32_t nd = a->num_dims;
        // Variable-length arrays follow num_dims in RT arg memory
        tt_l1_ptr uint32_t* num_unpadded = (tt_l1_ptr uint32_t*)(&a->num_dims + 1);
        tt_l1_ptr uint32_t* num_padded = num_unpadded + nd;
        tt_l1_ptr uint32_t* id_per_dim = num_padded + nd;

        auto st = seq_slice_init(a->start_id, nd, num_unpadded, num_padded, id_per_dim);
        read_pages(cb_id, BATCH, page_size, s, a->num_pages, st, seq_slice_next);
    }

    // ── PERMUTE ─────────────────────────────────────────────────────
    else if constexpr (SEQ_ID == SEQ_PERMUTE) {
        const auto* a = reinterpret_cast<const ArgsPermute*>(get_arg_addr(0));
        // src_strides[D], out_shape[D], inv_perm[D], id_per_dim[D] follow num_dims
        // seq_permute_init reads them from RT arg indices starting at &num_dims + 1
        uint32_t rt_data_start = 4;  // index of first element after ArgsPermute header
        auto st = seq_permute_init(a->num_dims, rt_data_start);
        read_pages(cb_id, BATCH, page_size, s, a->num_pages, st, seq_permute_next);
    }

    // ── TRANSPOSE_WH ────────────────────────────────────────────────
    else if constexpr (SEQ_ID == SEQ_TRANSPOSE_WH) {
        const auto* a = reinterpret_cast<const ArgsTransposeWh*>(get_arg_addr(0));
        auto st = seq_transpose_wh_init(a->start_id, a->start_ht, a->start_wt, a->Ht, a->Wt, a->HtWt);
        read_pages(cb_id, BATCH, page_size, s, a->num_pages, st, seq_transpose_wh_next);
    }

    // ── PAD (conditional source / fill) ─────────────────────────────
    // Different loop body: branches on is_data per page.
    else if constexpr (SEQ_ID == SEQ_PAD) {
        const auto* a = reinterpret_cast<const ArgsPad*>(get_arg_addr(0));

        // Fill pad tile in scratch CB
        {
            volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(a->cb_pad));
            for (uint32_t i = 0; i < a->tile_bytes / 4 + 1; ++i) {
                ptr[i] = a->packed_pad_val;
            }
        }
        uint64_t pad_noc_addr = get_noc_addr(get_read_ptr(a->cb_pad));

        auto st = seq_pad_init(
            a->start_id,
            a->start_wt,
            a->start_ht,
            a->start_c,
            a->start_n,
            a->Wt_in,
            a->Ht_in,
            a->C_in,
            a->N_in,
            a->Wt_out,
            a->Ht_out,
            a->C_out,
            a->N_out,
            a->front_wt,
            a->front_ht,
            a->front_c,
            a->front_n);

        uint32_t pages_left = a->num_pages;
        while (pages_left > 0) {
            uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
            cb_reserve_back(cb_id, batch);
            uint32_t l1_addr = get_write_ptr(cb_id);
            for (uint32_t t = 0; t < batch; t++) {
                uint32_t src_page = seq_pad_next(st);
                if (st.is_data) {
                    noc_async_read_tile(src_page, s, l1_addr);
                } else {
                    noc_async_read(pad_noc_addr, l1_addr, a->tile_bytes);
                }
                l1_addr += a->tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id, batch);
            pages_left -= batch;
        }
    }

    // ── CONCAT (2-tensor interleave) ────────────────────────────────
    // Different loop body: two TensorAccessors.
    else if constexpr (SEQ_ID == SEQ_CONCAT) {
        const auto* a = reinterpret_cast<const ArgsConcat*>(get_arg_addr(0));
        const auto s1 = TensorAccessor(src_args, a->src_addr_1, page_size);

        auto st = seq_concat_init(a->start_tensor, a->start_tensor_id, a->page_id_0, a->page_id_1, a->ppb_0, a->ppb_1);

        uint32_t pages_left = a->num_pages;
        while (pages_left > 0) {
            uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
            cb_reserve_back(cb_id, batch);
            uint32_t l1_addr = get_write_ptr(cb_id);
            for (uint32_t t = 0; t < batch; t++) {
                uint32_t read_tensor = st.curr_tensor;
                uint32_t src_page = seq_concat_next(st);
                if (read_tensor == 0) {
                    noc_async_read_tile(src_page, s, l1_addr);
                } else {
                    noc_async_read_tile(src_page, s1, l1_addr);
                }
                l1_addr += page_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id, batch);
            pages_left -= batch;
        }
    }
}
