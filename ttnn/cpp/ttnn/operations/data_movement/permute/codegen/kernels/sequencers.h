// ONLY-USES: SEQ_IDENTITY. The rest of this file is unused shared infra for other ports.
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified address sequencers for data movement ops.
//
// Each sequencer maps output_page -> source_page. The transport loop
// (reader_tile_interleaved_unified.cpp) calls sequencer_next<SEQ_ID>()
// per page. All functions are FORCE_INLINE — the compiler eliminates
// dead code via if constexpr on SEQ_ID. Zero overhead vs hand-written.
//
// Sequencer IDs (passed as CT arg SEQ_ID):
//   0 = IDENTITY   : src = sequential page (copy, no remapping)
//   1 = REPEAT     : modular block arithmetic for higher-dim repeat
//   2 = SLICE      : N-dim skip with padding offsets
//   3 = PERMUTE    : inverse permutation with strides
//   4 = TRANSPOSE_WH: stride-permuted tile index (W↔H swap)
//   5 = PAD        : conditional in-bounds/fill
//   6 = CONCAT     : multi-tensor page interleave (2 tensors)
//   7 = HC         : height-channel transpose stick mapper
//   8 = CN         : channel-batch transpose page mapper
//   9 = REPEAT_INTERLEAVE : per-element (AABB) replication for repeat_interleave
#pragma once

#include "api/dataflow/dataflow_api.h"

// --- Sequencer ID constants (match CT arg SEQ_ID) ---
constexpr uint32_t SEQ_IDENTITY = 0;
constexpr uint32_t SEQ_REPEAT = 1;
constexpr uint32_t SEQ_SLICE = 2;
constexpr uint32_t SEQ_PERMUTE = 3;
constexpr uint32_t SEQ_TRANSPOSE_WH = 4;
constexpr uint32_t SEQ_PAD = 5;
constexpr uint32_t SEQ_CONCAT = 6;
constexpr uint32_t SEQ_HC = 7;
constexpr uint32_t SEQ_CN = 8;
constexpr uint32_t SEQ_REPEAT_INTERLEAVE = 9;

// Max supported dimensions for N-dim sequencers (slice, permute)
constexpr uint32_t SEQ_MAX_DIMS = 8;

// =====================================================================
// Sequencer state structs
// =====================================================================

struct SeqIdentityState {
    uint32_t page_id;
};

struct SeqRepeatState {
    uint32_t out_page;
    uint32_t src_lower;  // REP_DIM_PAGES * LOWER_PAGES
    uint32_t dst_lower;  // NUM_REPEATS * src_lower
};

struct SeqRepeatInterleaveState {
    uint32_t out_page;
    uint32_t src_lower;    // REP_DIM_PAGES * LOWER_PAGES
    uint32_t dst_lower;    // NUM_REPEATS * src_lower
    uint32_t num_repeats;  // per-element replication factor
    uint32_t lower_pages;  // pages below the repeat dim
};

struct SeqSliceState {
    uint32_t src_tile_id;
    uint32_t num_dims;
    // These point into RT arg memory (L1) — no copy needed
    tt_l1_ptr uint32_t* num_unpadded;
    tt_l1_ptr uint32_t* num_padded;
    tt_l1_ptr uint32_t* id_per_dim;
};

struct SeqPermuteState {
    uint32_t num_dims;
    uint32_t src_strides[SEQ_MAX_DIMS];
    uint32_t out_shape[SEQ_MAX_DIMS];
    uint32_t inv_perm[SEQ_MAX_DIMS];
    uint32_t id_per_dim[SEQ_MAX_DIMS];
};

struct SeqTransposeWhState {
    uint32_t i_tile;
    uint32_t ht;
    uint32_t wt;
    uint32_t Ht;
    uint32_t Wt;
    uint32_t HtWt;
};

struct SeqPadState {
    uint32_t src_tile;
    uint32_t curr_wt;
    uint32_t curr_ht;
    uint32_t curr_c;
    uint32_t curr_n;
    uint32_t Wt_in;
    uint32_t Ht_in;
    uint32_t C_in;
    uint32_t N_in;
    uint32_t Wt_out;
    uint32_t Ht_out;
    uint32_t C_out;
    uint32_t N_out;
    uint32_t front_wt;  // tile-aligned leading pad, W (tiles)
    uint32_t front_ht;  // tile-aligned leading pad, H (tiles)
    uint32_t front_c;   // leading pad, C
    uint32_t front_n;   // leading pad, N
    bool is_data;       // set by next(), read by transport loop
};

struct SeqConcatState {
    uint32_t curr_tensor;
    uint32_t curr_tensor_id;
    uint32_t page_id_0;
    uint32_t page_id_1;
    uint32_t ppb_0;  // pages per block, tensor 0
    uint32_t ppb_1;  // pages per block, tensor 1
};

// =====================================================================
// IDENTITY sequencer
// =====================================================================

inline __attribute__((always_inline)) SeqIdentityState seq_identity_init(uint32_t start_id) { return {start_id}; }

inline __attribute__((always_inline)) uint32_t seq_identity_next(SeqIdentityState& st) { return st.page_id++; }

// =====================================================================
// REPEAT sequencer
// =====================================================================

inline __attribute__((always_inline)) SeqRepeatState
seq_repeat_init(uint32_t out_start_page, uint32_t num_repeats, uint32_t lower_pages, uint32_t rep_dim_pages) {
    uint32_t src_lower = rep_dim_pages * lower_pages;
    uint32_t dst_lower = num_repeats * src_lower;
    return {out_start_page, src_lower, dst_lower};
}

inline __attribute__((always_inline)) uint32_t seq_repeat_next(SeqRepeatState& st) {
    uint32_t block = st.out_page / st.dst_lower;
    uint32_t within = st.out_page % st.dst_lower;
    uint32_t lower_in_rep = within % st.src_lower;
    uint32_t src_page = block * st.src_lower + lower_in_rep;
    st.out_page++;
    return src_page;
}

// =====================================================================
// REPEAT_INTERLEAVE sequencer (per-element / AABB replication)
//
// Same modular-block family as REPEAT, but the within-block offset uses
// integer division by num_repeats instead of modulo by src_lower:
//   src_lower = rep_dim_pages * lower_pages
//   dst_lower = num_repeats * src_lower
//   block     = out_page / dst_lower            (everything above rep_dim)
//   within    = out_page % dst_lower
//   lo        = within % lower_pages            (offset below rep_dim)
//   out_rep   = within / lower_pages            (replicated rep_dim index)
//   in_rep    = out_rep / num_repeats           (AABB collapse, per-element)
//   src = block*src_lower + in_rep*lower_pages + lo
//
// When lower_pages == 1 this reduces to the documented short form
//   src = block*src_lower + (within / num_repeats)
// repeat (ABAB) instead uses within % src_lower, replicating whole blocks.
// =====================================================================

inline __attribute__((always_inline)) SeqRepeatInterleaveState seq_repeat_interleave_init(
    uint32_t out_start_page, uint32_t num_repeats, uint32_t lower_pages, uint32_t rep_dim_pages) {
    uint32_t src_lower = rep_dim_pages * lower_pages;
    uint32_t dst_lower = num_repeats * src_lower;
    return {out_start_page, src_lower, dst_lower, num_repeats, lower_pages};
}

inline __attribute__((always_inline)) uint32_t seq_repeat_interleave_next(SeqRepeatInterleaveState& st) {
    uint32_t block = st.out_page / st.dst_lower;
    uint32_t within = st.out_page % st.dst_lower;
    uint32_t lo = within % st.lower_pages;
    uint32_t out_rep = within / st.lower_pages;
    uint32_t in_rep = out_rep / st.num_repeats;
    uint32_t src_page = block * st.src_lower + in_rep * st.lower_pages + lo;
    st.out_page++;
    return src_page;
}

// =====================================================================
// SLICE sequencer (N-dimensional skip)
// =====================================================================

inline __attribute__((always_inline)) SeqSliceState seq_slice_init(
    uint32_t start_id,
    uint32_t num_dims,
    tt_l1_ptr uint32_t* num_unpadded,
    tt_l1_ptr uint32_t* num_padded,
    tt_l1_ptr uint32_t* id_per_dim) {
    return {start_id, num_dims, num_unpadded, num_padded, id_per_dim};
}

inline __attribute__((always_inline)) uint32_t seq_slice_next(SeqSliceState& st) {
    uint32_t result = st.src_tile_id;
    st.src_tile_id++;
    for (uint32_t j = 0; j < st.num_dims; ++j) {
        st.id_per_dim[j]++;
        if (st.id_per_dim[j] == st.num_unpadded[j]) {
            st.id_per_dim[j] = 0;
            st.src_tile_id += st.num_padded[j];
        } else {
            break;
        }
    }
    return result;
}

// =====================================================================
// PERMUTE sequencer (inverse permutation with strides)
// =====================================================================

inline __attribute__((always_inline)) SeqPermuteState seq_permute_init(
    uint32_t num_dims,
    uint32_t rt_start_idx  // RT arg index where src_strides[D] begins
) {
    SeqPermuteState st;
    st.num_dims = num_dims;
    uint32_t idx = rt_start_idx;
    for (uint32_t d = 0; d < num_dims; d++) {
        st.src_strides[d] = get_arg_val<uint32_t>(idx++);
    }
    for (uint32_t d = 0; d < num_dims; d++) {
        st.out_shape[d] = get_arg_val<uint32_t>(idx++);
    }
    for (uint32_t d = 0; d < num_dims; d++) {
        st.inv_perm[d] = get_arg_val<uint32_t>(idx++);
    }
    for (uint32_t d = 0; d < num_dims; d++) {
        st.id_per_dim[d] = get_arg_val<uint32_t>(idx++);
    }
    return st;
}

inline __attribute__((always_inline)) uint32_t seq_permute_next(SeqPermuteState& st) {
    // Compute source tile via inverse permutation
    uint32_t src_tile_id = 0;
    for (uint32_t d = 0; d < st.num_dims; d++) {
        src_tile_id += st.id_per_dim[st.inv_perm[d]] * st.src_strides[d];
    }
    // Advance N-dim output position (innermost last)
    for (uint32_t d = st.num_dims; d > 0; d--) {
        st.id_per_dim[d - 1]++;
        if (st.id_per_dim[d - 1] < st.out_shape[d - 1]) {
            break;
        }
        st.id_per_dim[d - 1] = 0;
    }
    return src_tile_id;
}

// =====================================================================
// TRANSPOSE_WH sequencer (W↔H tile swap via stride arithmetic)
// =====================================================================

inline __attribute__((always_inline)) SeqTransposeWhState seq_transpose_wh_init(
    uint32_t start_id, uint32_t start_ht, uint32_t start_wt, uint32_t Ht, uint32_t Wt, uint32_t HtWt) {
    return {start_id, start_ht, start_wt, Ht, Wt, HtWt};
}

inline __attribute__((always_inline)) uint32_t seq_transpose_wh_next(SeqTransposeWhState& st) {
    uint32_t result = st.i_tile;
    st.i_tile += st.Wt;
    st.ht++;
    if (st.ht == st.Ht) {
        st.ht = 0;
        st.i_tile++;
        st.wt++;
        if (st.wt == st.Wt) {
            st.wt = 0;
            st.i_tile -= st.Wt;
        } else {
            st.i_tile -= st.HtWt;
        }
    }
    return result;
}

// =====================================================================
// PAD sequencer (conditional in-bounds vs fill tile)
//
// After calling seq_pad_next(), check st.is_data:
//   true  -> returned page is a valid source tile ID, use noc_async_read_tile
//   false -> tile is padding, use noc_async_read from pad_noc_addr
// =====================================================================

inline __attribute__((always_inline)) SeqPadState seq_pad_init(
    uint32_t start_src_tile,
    uint32_t start_wt,
    uint32_t start_ht,
    uint32_t start_c,
    uint32_t start_n,
    uint32_t Wt_in,
    uint32_t Ht_in,
    uint32_t C_in,
    uint32_t N_in,
    uint32_t Wt_out,
    uint32_t Ht_out,
    uint32_t C_out,
    uint32_t N_out,
    uint32_t front_wt,
    uint32_t front_ht,
    uint32_t front_c,
    uint32_t front_n) {
    return {
        start_src_tile,
        start_wt,
        start_ht,
        start_c,
        start_n,
        Wt_in,
        Ht_in,
        C_in,
        N_in,
        Wt_out,
        Ht_out,
        C_out,
        N_out,
        front_wt,
        front_ht,
        front_c,
        front_n,
        false};
}

inline __attribute__((always_inline)) uint32_t seq_pad_next(SeqPadState& st) {
    // Data occupies [front, front + in) in each dim; everything else is pad.
    // Back-only padding is the front_* == 0 special case.
    st.is_data = (st.curr_wt >= st.front_wt) && (st.curr_wt < st.front_wt + st.Wt_in) && (st.curr_ht >= st.front_ht) &&
                 (st.curr_ht < st.front_ht + st.Ht_in) && (st.curr_c >= st.front_c) &&
                 (st.curr_c < st.front_c + st.C_in) && (st.curr_n >= st.front_n) && (st.curr_n < st.front_n + st.N_in);

    uint32_t result = st.src_tile;
    if (st.is_data) {
        st.src_tile++;
    }

    // Advance output tile coordinates: wt (innermost), ht, c, n
    st.curr_wt++;
    if (st.curr_wt == st.Wt_out) {
        st.curr_wt = 0;
        st.curr_ht++;
        if (st.curr_ht == st.Ht_out) {
            st.curr_ht = 0;
            st.curr_c++;
            if (st.curr_c == st.C_out) {
                st.curr_c = 0;
                st.curr_n++;
            }
        }
    }
    return result;
}

// =====================================================================
// CONCAT sequencer (2-tensor page interleave)
//
// After calling seq_concat_next(), check st.curr_tensor to know
// which TensorAccessor to use for the NOC read.
// The returned value is the page_id within that tensor.
// =====================================================================

inline __attribute__((always_inline)) SeqConcatState seq_concat_init(
    uint32_t start_tensor,
    uint32_t start_tensor_id,
    uint32_t page_id_0,
    uint32_t page_id_1,
    uint32_t ppb_0,
    uint32_t ppb_1) {
    return {start_tensor, start_tensor_id, page_id_0, page_id_1, ppb_0, ppb_1};
}

inline __attribute__((always_inline)) uint32_t seq_concat_next(SeqConcatState& st) {
    uint32_t result;
    uint32_t read_tensor = st.curr_tensor;  // save for caller

    if (st.curr_tensor == 0) {
        result = st.page_id_0++;
    } else {
        result = st.page_id_1++;
    }

    // Advance block tracking
    st.curr_tensor_id++;
    if (st.curr_tensor == 0 && st.curr_tensor_id == st.ppb_0) {
        st.curr_tensor_id = 0;
        st.curr_tensor = 1;
    } else if (st.curr_tensor == 1 && st.curr_tensor_id == st.ppb_1) {
        st.curr_tensor_id = 0;
        st.curr_tensor = 0;
    }

    return result;
}

// =====================================================================
// HC sequencer (height-channel transpose: iterate C fast, then H, then N)
//
// Input layout [N,C,H,W], output [N,H,C,W]. Sticks are W-element rows.
// Reader iterates in output order: for each output stick at (n,h,c),
// reads from input stick at (n,c,h) = i_stick.
// i_stick advances by H per C step (jumping between channels).
// =====================================================================

struct SeqHcState {
    uint32_t i_stick;
    uint32_t curr_c;
    uint32_t curr_h;
    uint32_t curr_n;
    uint32_t C;
    uint32_t H;
    uint32_t CH;
};

inline __attribute__((always_inline)) SeqHcState
seq_hc_init(uint32_t start_id, uint32_t curr_c, uint32_t curr_h, uint32_t curr_n, uint32_t C, uint32_t H) {
    return {start_id, curr_c, curr_h, curr_n, C, H, C * H};
}

inline __attribute__((always_inline)) uint32_t seq_hc_next(SeqHcState& st) {
    uint32_t result = st.i_stick;
    st.curr_c++;
    st.i_stick += st.H;
    if (st.curr_c == st.C) {
        st.curr_h++;
        st.curr_c = 0;
        if (st.curr_h == st.H) {
            st.curr_n++;
            st.curr_h = 0;
            st.i_stick = st.i_stick - st.H + 1;
        } else {
            st.i_stick = st.i_stick - st.CH + 1;
        }
    }
    return result;
}

// =====================================================================
// CN sequencer (channel-batch transpose: iterate N fast, then C)
//
// Input layout [N,C,HtWt], output [C,N,HtWt]. Pages are tiles or sticks.
// Reader iterates in output order: for each (c,n,hw), reads from (n,c,hw).
// page_idx advances by 1 within an HtWt block, then jumps by batch_step
// to next N, wraps by channel_step at N boundary.
// =====================================================================

struct SeqCnState {
    uint32_t page_idx;
    uint32_t hw;
    uint32_t n;
    uint32_t N;
    uint32_t HtWt;
    uint32_t batch_step;    // C*HtWt - HtWt
    uint32_t channel_step;  // N*C*HtWt - HtWt
};

inline __attribute__((always_inline)) SeqCnState seq_cn_init(
    uint32_t start_id, uint32_t hw, uint32_t n, uint32_t N, uint32_t HtWt, uint32_t batch_step, uint32_t channel_step) {
    return {start_id, hw, n, N, HtWt, batch_step, channel_step};
}

inline __attribute__((always_inline)) uint32_t seq_cn_next(SeqCnState& st) {
    uint32_t result = st.page_idx;
    st.page_idx++;
    st.hw++;
    if (st.hw == st.HtWt) {
        st.hw = 0;
        st.n++;
        st.page_idx += st.batch_step;
        if (st.n == st.N) {
            st.n = 0;
            st.page_idx -= st.channel_step;
        }
    }
    return result;
}
