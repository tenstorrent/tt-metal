// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// deepseek_v4_prefill_indexer_select compute. Per work unit (32 queries, one split of their visible 512-entry
// blocks), per block:
//   score[s, t] = sum_h relu(q[h, s] . k[t]) * w[s, h]      (matmul with packer ReLU, then a DEST-MAC gate)
// untilized into one 512-wide row per query, then folded into the query's running top-512 (TopK XL, unfused
// values + uint32 indices) with only the query's visible entries active. Survivors live in L1 between blocks.
// The leader merges the other splits' survivors and emits the indices, invisible slots as 0xFFFFFFFF.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/experimental/indexer_mul_custom.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/deepseek_v4_prefill_indexer_select_common.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

inline void _indexer_select_mark_neginf_indices_init_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_0);
}

// K = 512 survivor [values, indices]: an index whose value is exactly -inf (an invisible or empty slot) becomes
// 0xFFFFFFFF.
inline void _indexer_select_mark_neginf_indices_() {
    constexpr uint32_t indices_offset = 64;
    constexpr uint32_t iterations = 16;

    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0xFF80);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, 0xFFFF);

    for (uint32_t i = 0; i < iterations; ++i) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

}  // namespace ckernel::sfpu
#endif

namespace {

namespace dis = deepseek_v4_prefill_indexer_select;

constexpr uint32_t num_heads = get_compile_time_arg_val(dis::compute_ct_arg::NUM_HEADS);
constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(dis::compute_ct_arg::HEAD_DIM_TILES);
constexpr uint32_t num_splits = get_compile_time_arg_val(dis::compute_ct_arg::NUM_SPLITS);
constexpr uint32_t cb_q = get_compile_time_arg_val(dis::compute_ct_arg::CB_Q);
constexpr uint32_t cb_w = get_compile_time_arg_val(dis::compute_ct_arg::CB_W);
constexpr uint32_t cb_k_rm = get_compile_time_arg_val(dis::compute_ct_arg::CB_K_RM);
constexpr uint32_t cb_k = get_compile_time_arg_val(dis::compute_ct_arg::CB_K);
constexpr uint32_t cb_qk = get_compile_time_arg_val(dis::compute_ct_arg::CB_QK);
constexpr uint32_t cb_scores = get_compile_time_arg_val(dis::compute_ct_arg::CB_SCORES);
constexpr uint32_t cb_neginf = get_compile_time_arg_val(dis::compute_ct_arg::CB_NEGINF);
constexpr uint32_t cb_surv_v = get_compile_time_arg_val(dis::compute_ct_arg::CB_SURV_V);
constexpr uint32_t cb_surv_i = get_compile_time_arg_val(dis::compute_ct_arg::CB_SURV_I);
constexpr uint32_t cb_peer = get_compile_time_arg_val(dis::compute_ct_arg::CB_PEER);
constexpr uint32_t cb_out = get_compile_time_arg_val(dis::compute_ct_arg::CB_OUT);
constexpr uint32_t cb_meta = get_compile_time_arg_val(dis::compute_ct_arg::CB_META);
constexpr uint32_t cb_done = get_compile_time_arg_val(dis::compute_ct_arg::CB_DONE);

constexpr uint32_t K = dis::TOPK;
constexpr uint32_t step_tiles = dis::STEP_KEY_TILES;
constexpr uint32_t steps_per_block = K / (step_tiles * TILE_R_DIM);
constexpr uint32_t heads_per_pass = dis::HEADS_PER_PASS;
constexpr uint32_t step_qk_tiles = step_tiles * num_heads;
// Each query's score row occupies one bf16 tile slot of cb_scores (the TopK XL copy reads a K = 512 chunk from
// the top half of a tile), so the untilized rows are 32 x 32 datums (a full tile) apart.
constexpr uint32_t score_row_tiles = TILE_R_DIM;
static_assert(num_heads % heads_per_pass == 0);

// One step of step_tiles key tiles: tilize the keys, relu(q . k^T) for every head into cb_qk (head-major), then
// gate-multiply and reduce over heads in DEST and untilize the step's columns into every query's score row.
void score_step(uint32_t step) {
    compute_kernel_lib::tilize<head_dim_tiles, cb_k_rm, cb_k>(step_tiles, step_tiles * TILE_R_DIM);

    CircularBuffer k(cb_k), qk(cb_qk);
    k.wait_front(step_tiles * head_dim_tiles);
    // matmul maps in1 (k) to srcA and in0 (q) to srcB.
    reconfig_data_format(cb_k, cb_q);
    pack_reconfig_data_format(cb_qk);
    matmul_block_init(cb_q, cb_k, 1 /*transpose k*/, 1 /*ct_dim*/, heads_per_pass /*rt_dim*/, head_dim_tiles);
    pack_relu_config(ReluConfig::zero());
    qk.reserve_back(step_qk_tiles);
    for (uint32_t col = 0; col < step_tiles; ++col) {
        for (uint32_t h0 = 0; h0 < num_heads; h0 += heads_per_pass) {
            tile_regs_acquire();
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                matmul_block(
                    cb_q,
                    cb_k,
                    h0 * head_dim_tiles + d,
                    col * head_dim_tiles + d,
                    0,
                    1 /*transpose k*/,
                    1,
                    heads_per_pass,
                    head_dim_tiles);
            }
            tile_regs_commit();
            tile_regs_wait();
            // The matmul DEST layout needs the out-of-order matmul pack.
            for (uint32_t i = 0; i < heads_per_pass; ++i) {
                PACK((llk_matmul_pack<DST_ACCUM_MODE, true /*out_of_order*/, PackMode::Default>(
                    i, cb_qk, 1, (h0 + i) * step_tiles + col)));
            }
            tile_regs_release();
        }
    }
    qk.push_back(step_qk_tiles);
    k.pop_front(step_tiles * head_dim_tiles);
    pack_relu_config(ReluConfig::none());

    reconfig_data_format(cb_qk, cb_w);
    mul_bcast_cols_init_short_custom(cb_qk, cb_w);
    pack_reconfig_data_format(cb_scores);
    pack_untilize_dest_init<step_tiles, score_row_tiles>(cb_scores);
    qk.wait_front(step_qk_tiles);
    tile_regs_acquire();
    for (uint32_t h = 0; h < num_heads; ++h) {
        mul_tiles_bcast_cols_custom(cb_qk, cb_w, h * step_tiles, h, 0, step_tiles);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_untilize_dest<step_tiles, score_row_tiles>(cb_scores, 1, step);
    tile_regs_release();
    pack_untilize_uninit(cb_scores);
    qk.pop_front(step_qk_tiles);
}

// Sorts one chunk at DST dst into [values, indices] at dst, dst + 1, indices offset by entry base. The global
// index decode needs a rebuilt (fully ordered) fused sequence, so the order is set by the rebuild's mirror.
void sort_chunk(uint32_t src_cb, uint32_t src_tile, uint32_t active, uint32_t base, uint32_t dst, bool ascending) {
    topk_xl_copy_tile_init(src_cb);
    topk_xl_copy_tile<K>(src_cb, dst, src_tile, active);
    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices_rt<K>(dst, 0);
    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(dst, false);
    topk_xl_rebuild<K, true>(dst, ascending);
    topk_xl_separate_indices_row_major_global_init();
    topk_xl_separate_indices_row_major_global_base<K>(dst, base);
}

// Loads a [values, indices] fp32 pair at tiles (v_tile, i_tile) of cb into DST dst, dst + 1 (unpacked to DEST
// unchanged, so index bits survive).
void load_pair(uint32_t cb_v, uint32_t v_tile, uint32_t cb_i, uint32_t i_tile, uint32_t dst) {
    reconfig_data_format_srca(cb_v);
    copy_init(cb_v);
    copy_tile(cb_v, v_tile, dst);
    copy_tile(cb_i, i_tile, dst + 1);
}

void store_survivor(uint32_t s) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_surv_v);
    pack_tile<true>(0, cb_surv_v, s);
    pack_tile<true>(1, cb_surv_i, s);
    tile_regs_release();
}

// Survivors stay descending while the split scans and end in the order the leader's merge takes them: descending
// for the leader's own (the accumulator), ascending for a peer's, so the leader merges them as they arrive.
//
// One row against one group of blocks [group_block, group_block + count) (the group's visible chunks of the row,
// active entries each). The chunks are merged fused ([bf16 value | u16 stamp] per DEST word, the stamp holding the
// block's position in its group) as in topk_large_indices; the group's top-k is then split into [values, indices]
// once and merged into the row's survivor. The first group seeds the survivor instead (at DEST 0); later groups
// build at DEST 2, the merge's second operand.
void select_group(
    uint32_t s, uint32_t group_block, uint32_t count, const uint32_t* active, bool first, bool last, bool ascending) {
    const uint32_t slot = first ? 0 : 2;
    tile_regs_acquire();
    if (!first) {
        load_pair(cb_surv_v, s, cb_surv_i, s, 0);
    }
    for (uint32_t j = 0; j < count; ++j) {
        const uint32_t dst = j == 0 ? slot : slot + 1;
        topk_xl_copy_tile_init(cb_scores);
        topk_xl_copy_tile<K>(cb_scores, dst, j * dis::ROWS + s, active[j]);
        topk_xl_add_lsb_indices_init();
        topk_xl_add_lsb_indices_rt<K>(dst, (group_block + j) % dis::GROUP_BLOCKS);
        topk_xl_init<K, true>();
        topk_xl_local_sort<K>(dst, j != 0);
        if (j != 0) {
            topk_xl_merge<K, true>(slot);
        }
        // The last rebuild sets the group's order: final for a first (seeding) group, ascending as a merge operand.
        const bool mirror = first ? last && ascending : true;
        topk_xl_rebuild<K, true>(slot, j + 1 == count && mirror);
    }
    topk_xl_separate_indices_row_major_global_init();
    topk_xl_separate_indices_row_major_global_base<K>(slot, (group_block / dis::GROUP_BLOCKS) * dis::GROUP_BLOCKS * K);
    if (!first) {
        topk_xl_init<K, false>();
        topk_xl_merge<K, false>(0);
        topk_xl_rebuild<K, false>(0, last && ascending);
    }
    store_survivor(s);
}

// A row that sees none of the split: an all -inf survivor.
void seed_empty_row(uint32_t s) {
    tile_regs_acquire();
    sort_chunk(cb_neginf, 0, 1, 0, 0, false);
    store_survivor(s);
}

// Leader: merge the survivors of the splits that scanned a block into its own and emit. A tile that sees no entry
// emits an all -inf chunk, i.e. only sentinels.
void emit_row(uint32_t s, uint32_t tile_blocks) {
    CircularBuffer peer(cb_peer), out(cb_out);
    tile_regs_acquire();
    if (tile_blocks == 0) {
        sort_chunk(cb_neginf, 0, 1, 0, 0, false);
    } else {
        load_pair(cb_surv_v, s, cb_surv_i, s, 0);
    }
    for (uint32_t p = 1; p < num_splits; ++p) {
        if (dis::split_first_block(p + 1, tile_blocks, num_splits) ==
            dis::split_first_block(p, tile_blocks, num_splits)) {
            continue;
        }
        peer.wait_front(2);
        load_pair(cb_peer, 0, cb_peer, 1, 2);
        peer.pop_front(2);
        topk_xl_init<K, false>();
        topk_xl_merge<K, false>(0);
        topk_xl_rebuild<K, false>(0, false);
    }
    MATH((ckernel::sfpu::_indexer_select_mark_neginf_indices_init_()));
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_indexer_select_mark_neginf_indices_, 0, VectorMode::RC_custom)));
    // Indices in rank order, row-major. The transpose moves DEST through the FPU, which does not wait for the SFPU,
    // and carries each 32-bit word as two 16-bit halves through the source registers, where zero flags would flush
    // index halves that look like zeros (e.g. 256 = 0x0100).
    MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)));
    transpose_dest_init<true, false>(cb_out);
    MATH((cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1)));
    MATH((cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_dst_RMW>(1)));
    transpose_dest<true, false>(1);
    MATH((cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0)));
    MATH((cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_dst_RMW>(0)));
    tile_regs_commit();
    tile_regs_wait();
    out.reserve_back(1);
    pack_reconfig_data_format(cb_out);
    pack_untilize_dest_init<1, 1>(cb_out);
    pack_untilize_dest<1, 1>(cb_out, 1, 0, 1);
    pack_untilize_uninit(cb_out);
    out.push_back(1);
    tile_regs_release();
}

}  // namespace

void kernel_main() {
    const uint32_t num_q_tiles = get_arg_val<uint32_t>(dis::core_arg::NUM_Q_TILES);
    const uint32_t split = get_arg_val<uint32_t>(dis::core_arg::SPLIT);

    compute_kernel_hw_startup(cb_k, cb_q, cb_qk);
    CircularBuffer meta(cb_meta), q(cb_q), w(cb_w), scores(cb_scores), done(cb_done);
    CircularBuffer(cb_neginf).wait_front(1);

    for (uint32_t unit = 0; unit < num_q_tiles; ++unit) {
        meta.wait_front(1);
        const uint32_t num_blocks = read_tile_value(cb_meta, 0, dis::meta_word::NUM_BLOCKS);
        const uint32_t first_entry = read_tile_value(cb_meta, 0, dis::meta_word::FIRST_ENTRY);
        const uint32_t tile_blocks = read_tile_value(cb_meta, 0, dis::meta_word::TILE_BLOCKS);
        uint32_t limits[dis::ROWS];
        for (uint32_t s = 0; s < dis::ROWS; ++s) {
            limits[s] = read_tile_value(cb_meta, 0, dis::meta_word::LIMITS + s);
        }
        q.wait_front(num_heads * head_dim_tiles);
        w.wait_front(num_heads);

        // Groups are aligned to absolute multiples of GROUP_BLOCKS blocks, so a group's entry base is a multiple of
        // GROUP_BLOCKS * K and ORs onto the decoded in-group index.
        const uint32_t first_block = first_entry / K;
        const uint32_t end_block = first_block + num_blocks;
        for (uint32_t group = first_block; group < end_block;) {
            const uint32_t group_end_aligned = (group / dis::GROUP_BLOCKS + 1) * dis::GROUP_BLOCKS;
            const uint32_t group_end = group_end_aligned < end_block ? group_end_aligned : end_block;
            const uint32_t group_blocks = group_end - group;
            for (uint32_t b = 0; b < group_blocks; ++b) {
                scores.reserve_back(dis::ROWS);
                for (uint32_t step = 0; step < steps_per_block; ++step) {
                    score_step(step);
                }
                scores.push_back(dis::ROWS);
            }
            // The TopK XL copy does not wrap around the CB, so every group fills (and frees) the whole of cb_scores
            // and starts at its base; the slots of a short group's missing blocks are left unwritten.
            scores.reserve_back((dis::GROUP_BLOCKS - group_blocks) * dis::ROWS);
            scores.push_back((dis::GROUP_BLOCKS - group_blocks) * dis::ROWS);
            scores.wait_front(dis::GROUP_BLOCKS * dis::ROWS);
            for (uint32_t s = 0; s < dis::ROWS; ++s) {
                // Query s sees blocks [first_block, row_end) of the split; its last one finishes its survivor.
                const uint32_t seen_blocks = (limits[s] + K - 1) / K;
                const uint32_t row_end = seen_blocks < end_block ? seen_blocks : end_block;
                if (row_end <= first_block) {
                    if (group == first_block) {
                        seed_empty_row(s);
                    }
                    continue;
                }
                if (row_end <= group) {
                    continue;
                }
                const uint32_t count = (row_end < group_end ? row_end : group_end) - group;
                uint32_t active[dis::GROUP_BLOCKS];
                for (uint32_t j = 0; j < count; ++j) {
                    const uint32_t base = (group + j) * K;
                    active[j] = limits[s] - base < K ? limits[s] - base : K;
                }
                select_group(s, group, count, active, group == first_block, row_end <= group_end, split != 0);
            }
            scores.pop_front(dis::GROUP_BLOCKS * dis::ROWS);
            group = group_end;
        }
        if (split == 0) {
            for (uint32_t s = 0; s < dis::ROWS; ++s) {
                emit_row(s, tile_blocks);
            }
        } else if (num_blocks > 0) {
            done.reserve_back(1);
            done.push_back(1);
        }
        q.pop_front(num_heads * head_dim_tiles);
        w.pop_front(num_heads);
        meta.pop_front(1);
    }
}
