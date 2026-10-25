// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa compute: flash/online-softmax over k_chunks (compute_streaming primitives).
// Per token: tilize Q. Per chunk: tilize K to [Skt,DHt] (both matmuls read it directly, no Kᵀ/V copies),
// then for each query group of qsb<=dst_size tile-rows (so DST never caps H):
//   Phase 1: Q@Kᵀ -> cb_qk_im (held wr_ptr), boundary mask add, running row-max.
//   Phase 2: sub_exp in place (exp((s-max)*scale)) + partial row-sum (L1-acc into cur.sum), probs@V -> cur.out,
//            then the SALAD flash combine: correction exp(prev_max-cur_max) applied to prev.out/prev.sum.
// The partial row-sum is finalized once on the last chunk (normalize_row_streaming). num_active_chunks==1
// degenerates to a plain single-chunk softmax (no SALAD).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"  // add_tiles_bcast_rows (mask add)
#include "api/compute/binary_max_min.h"  // split-K merge: elementwise max of two running-max tiles
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
constexpr bool EXP_APPROX_MODE = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::MATH_APPROX_MODE) != 0;
// compute_streaming.hpp needs declarations from compute_common.hpp (LightweightMaskContext, reduce helpers,
// DEST_AUTO_LIMIT); include it first. Only compute_streaming primitives are used.
#include "compute_common.hpp"
#include "compute_streaming.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/dataflow/circular_buffer.h"  // CircularBuffer: COMPILE_FOR_TRISC-aware CB lifecycle
#include "api/tile_constants.h"            // tt::constants::TILE_HEIGHT

// Debug-only profiling zones (SPARSE_SDPA_DBG bitmask, see sparse_sdpa_common.hpp dbg::).
#if defined(SPARSE_SDPA_DBG) && (SPARSE_SDPA_DBG & 4)
#define CZONE(n) DeviceZoneScopedN(n)
#else
#define CZONE(n)
#endif
#if defined(SPARSE_SDPA_DBG) && (SPARSE_SDPA_DBG & 128)
#define CZONE_W(n) DeviceZoneScopedN(n)
#else
#define CZONE_W(n)
#endif

// In-place scores[tile] += row-broadcast of mask row 0. cb_qk_im is held (hold_wr_ptr), so re-pack at the
// absolute position. Caller has init'd the bcast-rows op + single-tile pack.
ALWI void add_bcast_row_mask_tile(uint32_t scores_cb, uint32_t mask_cb, uint32_t score_tile) {
    tile_regs_acquire();
    add_tiles_bcast_rows(scores_cb, mask_cb, score_tile, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, scores_cb, score_tile);
    tile_regs_release();
}

// Make in-place PACK writes to a held CB visible to the next UNPACK read (after the in-place mask/sub_exp
// writes to cb_qk_im and the V-matmul writes to out_cur, none of which do a cb_push_back).
ALWI void pack_to_unpack_sync() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

ALWI void swap_cb(CircularBuffer& a, CircularBuffer& b) {
    const CircularBuffer t = a;
    a = b;
    b = t;
}

// Tilize a 32-row field in a mixed-format packed slab without first copying it into a dense row-major CB.
// `physical_ct_dim` programs the unpacker's row stride; `field_ct_dim` is the number of columns actually
// emitted. The read-pointer offset is temporary and restored before the normal FIFO pop advances by 32
// physical packed pages.
template <
    uint32_t input_cb,
    uint32_t output_cb,
    uint32_t physical_ct_dim,
    uint32_t field_ct_dim,
    uint32_t field_offset_16b,
    bool manage_input_lifecycle = true>
ALWI void tilize_packed_field() {
    CircularBuffer in(input_cb), out(output_cb);
    if constexpr (manage_input_lifecycle) {
        in.wait_front(tt::constants::TILE_HEIGHT);
    }
    out.reserve_back(field_ct_dim);

    reconfig_data_format_srca(input_cb);
    pack_reconfig_data_format(output_cb);
    tilize_init(input_cb, physical_ct_dim, output_cb);

    uint32_t saved_rd_ptr = 0;
    UNPACK({
        auto& iface = get_local_cb_interface(input_cb);
        saved_rd_ptr = iface.fifo_rd_ptr;
        iface.fifo_rd_ptr = saved_rd_ptr + field_offset_16b;
    });
    tilize_block(input_cb, field_ct_dim, output_cb);
    UNPACK(get_local_cb_interface(input_cb).fifo_rd_ptr = saved_rd_ptr;);

    out.push_back(field_ct_dim);
    if constexpr (manage_input_lifecycle) {
        in.pop_front(tt::constants::TILE_HEIGHT);
    }
    tilize_uninit(input_cb, output_cb);
}

// Copy `ntiles` bf16 tiles in_cb -> out_cb through DST (split-K HEAD state hand-off to the writer).
template <uint32_t dst_batch>
ALWI void copy_state_tiles(uint32_t in_cb, uint32_t out_cb, uint32_t ntiles) {
    CircularBuffer in(in_cb), out(out_cb);
    in.wait_front(ntiles);
    out.reserve_back(ntiles);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    pack_reconfig_l1_acc(0);
    configure_single_tile_pack(out_cb);
    copy_init(in_cb);
    for (uint32_t t = 0; t < ntiles; t += dst_batch) {
        const uint32_t n = (ntiles - t) < dst_batch ? (ntiles - t) : dst_batch;
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            copy_tile(in_cb, t + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile(j, out_cb);
        }
        tile_regs_release();
    }
    out.push_back(ntiles);
    in.pop_front(ntiles);
}

void kernel_main() {
    constexpr bool sink_per_row = true;  // sparse query rows represent different heads
    constexpr bool use_attention_sink = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::USE_ATTENTION_SINK) != 0;
    constexpr uint32_t cb_attention_sink = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_ATTENTION_SINK);
    constexpr uint32_t H = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::H);
    constexpr uint32_t DHt = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::DHT);
    constexpr uint32_t vDHt = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::V_DHT);
    constexpr uint32_t Skt = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::SKT);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::SCALE);

    // CB ids from the factory (the SparseCB enum is the single source); order matches the factory's compute
    // compile-arg block (5..24). They feed the templates + format helpers; the CB objects below wrap them.
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_Q_RM);
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_Q_IN);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_K_RM);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_K_IN);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_NEGINF);
    constexpr uint32_t cb_mask_part = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_MASK_PART);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_SCALE);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_QK_IM);
    constexpr uint32_t cb_max_a = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_MAX_A);
    constexpr uint32_t cb_max_b = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_MAX_B);
    constexpr uint32_t cb_sum_a = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_SUM_A);
    constexpr uint32_t cb_sum_b = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_SUM_B);
    constexpr uint32_t cb_out_a = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_OUT_A);
    constexpr uint32_t cb_out_b = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_OUT_B);
    constexpr uint32_t cb_corr = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_CORR);
    constexpr uint32_t cb_out_im = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_OUT_IM);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_OUT_RM);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_CTRL);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_COL_IDENTITY);
    constexpr uint32_t cb_recip_scratch = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_RECIP_SCRATCH);
    constexpr bool scaled_kv = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::SCALED_KV) != 0;
    constexpr uint32_t cb_k_rope_rm = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_K_ROPE_RM);
    constexpr uint32_t cb_k_scale_bcast = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_K_SCALE_BCAST);
    constexpr uint32_t cb_k_latent_tile = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_K_LATENT_TILE);
    constexpr uint32_t cb_k_rope_tile = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_K_ROPE_TILE);
    constexpr bool split_k = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::SPLIT_K) != 0;
    constexpr uint32_t cb_send_out = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_SEND_OUT);
    constexpr uint32_t cb_send_max = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_SEND_MAX);
    constexpr uint32_t cb_send_sum = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_SEND_SUM);
    constexpr uint32_t cb_recv_out = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_RECV_OUT);
    constexpr uint32_t cb_recv_max = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_RECV_MAX);
    constexpr uint32_t cb_recv_sum = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_RECV_SUM);
    constexpr uint32_t cb_merge_ctrl = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_MERGE_CTRL);
    constexpr uint32_t cb_mmax = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_MMAX);
    constexpr uint32_t cb_corr2 = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::CB_CORR2);
    constexpr bool kdeq_batch = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::KDEQ_BATCH) != 0;
    constexpr uint32_t pv_w = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::PV_W);

    constexpr uint32_t qsb =
        get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::QUERY_SUBBLOCK);  // query tile-rows per DST group
    constexpr uint32_t packed_row_bytes = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::PACKED_ROW_BYTES);
    constexpr uint32_t Sqt = H / tt::constants::TILE_HEIGHT;  // total query tile-rows (32 heads each)
    constexpr uint32_t q_groups = Sqt / qsb;                  // DST-bound work runs in this many query-row passes
    constexpr uint32_t KT_stride = Skt;                       // cb_qk_im physical row width
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    // sub_exp packs qsb*sbw tiles into DST: full Skt width when one group fits, else one key-tile column at a time.
    constexpr uint32_t exp_sbw = (qsb * Skt <= dst_size) ? Skt : 1;

    // CB wrappers for the fixed (non-ping-pong) buffers' lifecycle verbs.
    CircularBuffer q_in_cb(cb_q_in), k_in_cb(cb_k_in), qk_cb(cb_qk_im), scale_cb(cb_scale), ctrl_cb(cb_ctrl);
    CircularBuffer neginf_cb(cb_neginf), mask_part_cb(cb_mask_part), corr_cb(cb_corr);
    CircularBuffer k_rm_cb(cb_k_rm), scale_bcast_cb(cb_k_scale_bcast);
    CircularBuffer latent_tile_cb(cb_k_latent_tile), rope_tile_cb(cb_k_rope_tile);

    const uint32_t tok_count = get_arg_val<uint32_t>(sparse_sdpa::split_rt_arg::TOK_COUNT);
    const uint32_t head_c0 = get_arg_val<uint32_t>(sparse_sdpa::split_rt_arg::HEAD_C0);
    const uint32_t tail_c1 = get_arg_val<uint32_t>(sparse_sdpa::split_rt_arg::TAIL_C1);
    const uint32_t n_segs = (head_c0 != 0 ? 1u : 0u) + tok_count + (tail_c1 != 0 ? 1u : 0u);

    // Q tilize is the first operation, so startup must configure raw-Q -> tiled-Q.
    compute_kernel_hw_startup(cb_q_rm, cb_q_in);

    scale_cb.wait_front(1);  // persistent reduce scaler; the streaming reduce assumes it is ready

    if constexpr (sparse_sdpa::dbg::MASK & sparse_sdpa::dbg::NO_COMPUTE) {
        // Gather-only timing: drain every CB the dataflow kernels fill, emit an (uninitialized) output block.
        CircularBuffer q_rm_cb(cb_q_rm), out_rm_cb(cb_out_rm);
        for (uint32_t tok = 0; tok < n_segs; ++tok) {
            q_rm_cb.wait_front(H);
            q_rm_cb.pop_front(H);
            ctrl_cb.wait_front(1);
            const uint32_t n_active = ckernel::read_tile_value(cb_ctrl, 0, sparse_sdpa::control_message::ACTIVE_CHUNKS);
            const uint32_t n_valid = ckernel::read_tile_value(cb_ctrl, 0, sparse_sdpa::control_message::VALID_KEYS);
            ctrl_cb.pop_front(1);
            for (uint32_t chunk = 0; chunk < n_active; ++chunk) {
                for (uint32_t row_tile = 0; row_tile < Skt; ++row_tile) {
                    k_rm_cb.wait_front(tt::constants::TILE_HEIGHT);
                    k_rm_cb.pop_front(tt::constants::TILE_HEIGHT);
                    if constexpr (scaled_kv) {
                        constexpr uint32_t sb = sparse_sdpa::scale_block_count(vDHt * tt::constants::TILE_WIDTH);
                        scale_bcast_cb.wait_front(sb);
                        scale_bcast_cb.pop_front(sb);
                    }
                }
                const uint32_t k_chunk = Skt * tt::constants::TILE_WIDTH;
                if (chunk == n_active - 1 && ((n_valid - chunk * k_chunk) % tt::constants::TILE_WIDTH) != 0) {
                    mask_part_cb.wait_front(1);
                    mask_part_cb.pop_front(1);
                }
            }
            out_rm_cb.reserve_back(Sqt * vDHt);
            out_rm_cb.push_back(Sqt * vDHt);
        }
        return;
    }
    for (uint32_t seg = 0; seg < n_segs; ++seg) {
        // tilize Q rows -> [Sqt, DHt]
        compute_kernel_lib::tilize<DHt, cb_q_rm, cb_q_in>(/*num_blocks=*/Sqt, /*total_input_pages=*/H);

        // Per-token control from the reader: num_active_chunks (>=1; all-sentinel chunks are skipped) and
        // num_valid_keys (last chunk's mask boundary). read_tile_value mailbox-distributes the UNPACK read so
        // all three threads see the same loop bound / mask branches.
        ctrl_cb.wait_front(1);
        const uint32_t num_active_chunks = ckernel::read_tile_value(
            cb_ctrl, /*tile=*/0, /*element_offset=*/sparse_sdpa::control_message::ACTIVE_CHUNKS);
        const uint32_t num_valid_keys =
            ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/sparse_sdpa::control_message::VALID_KEYS);
        const uint32_t chunk_base =
            ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/sparse_sdpa::control_message::CHUNK_BASE);
        const uint32_t seg_kind =
            ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/sparse_sdpa::control_message::SEG_KIND);
        ctrl_cb.pop_front(1);
        // Whole-token active chunk count (the boundary mask lives on the token's last active chunk).
        constexpr uint32_t k_chunk_keys = Skt * tt::constants::TILE_WIDTH;
        const uint32_t token_chunks = (num_valid_keys + k_chunk_keys - 1) / k_chunk_keys;
        if (num_active_chunks == 0) {  // empty split-K HEAD: nothing to compute or send
            q_in_cb.wait_front(Sqt * DHt);
            q_in_cb.pop_front(Sqt * DHt);
            continue;
        }

        // Flash running state, ping-pong. Reset every token; all buffers start empty.
        CircularBuffer max_prev(cb_max_a), max_cur(cb_max_b);
        CircularBuffer sum_prev(cb_sum_a), sum_cur(cb_sum_b);
        CircularBuffer out_prev(cb_out_a), out_cur(cb_out_b);

        for (uint32_t chunk = 0; chunk < num_active_chunks; ++chunk) {
            if constexpr (scaled_kv) {
                // Reconstruct only the scaled latent field into persistent BFP8 tiles. RoPE stays in its
                // separate BF16 tiled buffer, avoiding both a full BF16 K and the RoPE copy into it.
                constexpr uint32_t latent_DHt = vDHt;
                constexpr uint32_t rope_DHt = DHt - vDHt;
                constexpr uint32_t latent_width = latent_DHt * tt::constants::TILE_WIDTH;
                static_assert(
                    sparse_sdpa::SCALE_BLOCK_WIDTH % tt::constants::TILE_WIDTH == 0,
                    "scaled KV scale blocks must span an exact number of tiles");
                constexpr uint32_t tiles_per_scale_block = sparse_sdpa::SCALE_BLOCK_WIDTH / tt::constants::TILE_WIDTH;
                constexpr uint32_t scale_blocks = sparse_sdpa::scale_block_count(latent_width);
                constexpr uint32_t scale_bytes = sparse_sdpa::scaled_kv_scale_bytes(latent_width);
                static_assert(
                    !scaled_kv || sparse_sdpa::scaled_kv_rope_offset_is_aligned(latent_width),
                    "scaled KV scale/RoPE boundary must be addressable in 16-byte units");
                static_assert(
                    packed_row_bytes % (tt::constants::TILE_WIDTH * sizeof(uint16_t)) == 0,
                    "scaled KV physical row stride must be an exact number of BF16 tile rows");
                constexpr uint32_t latent_physical_ct = packed_row_bytes / tt::constants::TILE_WIDTH;
                constexpr uint32_t rope_physical_ct = packed_row_bytes / (tt::constants::TILE_WIDTH * sizeof(uint16_t));
                constexpr uint32_t rope_offset_16b =
                    (latent_width + scale_bytes) / sparse_sdpa::PACKED_FIELD_ADDRESS_UNIT_BYTES;

                CZONE("C-KDEQ");
                k_in_cb.reserve_back(Skt * vDHt);
                if constexpr (kdeq_batch) {
                    // Whole chunk at once: one tilize init per field and one multiply init per chunk instead of
                    // per 32-row slab (needs cb_k_rm to hold the chunk's Skt slabs contiguously).
                    constexpr uint32_t slab_16b =
                        tt::constants::TILE_HEIGHT * packed_row_bytes / sparse_sdpa::PACKED_FIELD_ADDRESS_UNIT_BYTES;
                    {
                        CZONE_W("C-KW");
                        k_rm_cb.wait_front(tt::constants::TILE_HEIGHT * Skt);
                    }
                    uint32_t saved_rd_ptr = 0;
                    UNPACK(saved_rd_ptr = get_local_cb_interface(cb_k_rm).fifo_rd_ptr;);
                    latent_tile_cb.reserve_back(Skt * latent_DHt);
                    reconfig_data_format_srca(cb_k_rm);
                    pack_reconfig_data_format(cb_k_latent_tile);
                    tilize_init(cb_k_rm, latent_physical_ct, cb_k_latent_tile);
                    for (uint32_t r = 0; r < Skt; ++r) {
                        UNPACK(get_local_cb_interface(cb_k_rm).fifo_rd_ptr = saved_rd_ptr + r * slab_16b;);
                        tilize_block(cb_k_rm, latent_DHt, cb_k_latent_tile, 0, r * latent_DHt);
                    }
                    UNPACK(get_local_cb_interface(cb_k_rm).fifo_rd_ptr = saved_rd_ptr;);
                    latent_tile_cb.push_back(Skt * latent_DHt);
                    tilize_uninit(cb_k_rm, cb_k_latent_tile);

                    latent_tile_cb.wait_front(Skt * latent_DHt);
                    scale_bcast_cb.wait_front(Skt * scale_blocks);
                    reconfig_data_format(cb_k_latent_tile, cb_k_scale_bcast);
                    pack_reconfig_data_format(cb_k_in);
                    mul_bcast_cols_init(cb_k_latent_tile, cb_k_scale_bcast);
                    for (uint32_t r = 0; r < Skt; ++r) {
                        for (uint32_t block = 0; block < scale_blocks; ++block) {
                            tile_regs_acquire();
                            for (uint32_t t = 0; t < tiles_per_scale_block; ++t) {
                                mul_tiles_bcast_cols(
                                    cb_k_latent_tile,
                                    cb_k_scale_bcast,
                                    r * vDHt + block * tiles_per_scale_block + t,
                                    r * scale_blocks + block,
                                    t);
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            for (uint32_t t = 0; t < tiles_per_scale_block; ++t) {
                                pack_tile<true>(t, cb_k_in, r * vDHt + block * tiles_per_scale_block + t);
                            }
                            tile_regs_release();
                        }
                    }
                    latent_tile_cb.pop_front(Skt * latent_DHt);
                    scale_bcast_cb.pop_front(Skt * scale_blocks);

                    rope_tile_cb.reserve_back(Skt * rope_DHt);
                    reconfig_data_format_srca(cb_k_rope_rm);
                    pack_reconfig_data_format(cb_k_rope_tile);
                    tilize_init(cb_k_rope_rm, rope_physical_ct, cb_k_rope_tile);
                    for (uint32_t r = 0; r < Skt; ++r) {
                        UNPACK(get_local_cb_interface(cb_k_rope_rm).fifo_rd_ptr =
                                   saved_rd_ptr + r * slab_16b + rope_offset_16b;);
                        tilize_block(cb_k_rope_rm, rope_DHt, cb_k_rope_tile, 0, r * rope_DHt);
                    }
                    rope_tile_cb.push_back(Skt * rope_DHt);
                    tilize_uninit(cb_k_rope_rm, cb_k_rope_tile);
                    k_rm_cb.pop_front(tt::constants::TILE_HEIGHT * Skt);
                }
                for (uint32_t row_tile = 0; row_tile < (kdeq_batch ? 0u : Skt); ++row_tile) {
                    {
                        CZONE_W("C-KW");
                        k_rm_cb.wait_front(tt::constants::TILE_HEIGHT);
                    }
                    if constexpr (sparse_sdpa::dbg::MASK & sparse_sdpa::dbg::NO_KTILIZE) {
                        latent_tile_cb.reserve_back(latent_DHt);
                        latent_tile_cb.push_back(latent_DHt);
                    } else {
                        tilize_packed_field<
                            cb_k_rm,
                            cb_k_latent_tile,
                            latent_physical_ct,
                            latent_DHt,
                            /*field_offset_16b=*/0,
                            /*manage_input_lifecycle=*/false>();
                    }

                    if constexpr (sparse_sdpa::dbg::MASK & sparse_sdpa::dbg::NO_DEQUANT_MUL) {
                        latent_tile_cb.wait_front(latent_DHt);
                        scale_bcast_cb.wait_front(scale_blocks);
                        latent_tile_cb.pop_front(latent_DHt);
                        scale_bcast_cb.pop_front(scale_blocks);
                    } else {
                        latent_tile_cb.wait_front(latent_DHt);
                        scale_bcast_cb.wait_front(scale_blocks);
                        reconfig_data_format(cb_k_latent_tile, cb_k_scale_bcast);
                        pack_reconfig_data_format(cb_k_in);
                        mul_bcast_cols_init(cb_k_latent_tile, cb_k_scale_bcast);
                        for (uint32_t block = 0; block < scale_blocks; ++block) {
                            for (uint32_t tile = 0; tile < tiles_per_scale_block; ++tile) {
                                tile_regs_acquire();
                                mul_tiles_bcast_cols(
                                    cb_k_latent_tile, cb_k_scale_bcast, block * tiles_per_scale_block + tile, block, 0);
                                tile_regs_commit();
                                tile_regs_wait();
                                pack_tile<true>(0, cb_k_in, row_tile * vDHt + block * tiles_per_scale_block + tile);
                                tile_regs_release();
                            }
                        }
                        latent_tile_cb.pop_front(latent_DHt);
                        scale_bcast_cb.pop_front(scale_blocks);
                    }

                    if constexpr (sparse_sdpa::dbg::MASK & sparse_sdpa::dbg::NO_KTILIZE) {
                        rope_tile_cb.reserve_back(DHt - vDHt);
                        rope_tile_cb.push_back(DHt - vDHt);
                    } else {
                        // cb_k_rope_rm is a format-only alias: synchronize its UNPACK pointer to the owning
                        // packed FIFO for this slab. It never participates in wait/pop lifecycle accounting.
                        UNPACK(get_local_cb_interface(cb_k_rope_rm).fifo_rd_ptr =
                                   get_local_cb_interface(cb_k_rm).fifo_rd_ptr;);
                        tilize_packed_field<
                            cb_k_rope_rm,
                            cb_k_rope_tile,
                            rope_physical_ct,
                            rope_DHt,
                            rope_offset_16b,
                            /*manage_input_lifecycle=*/false>();
                    }
                    k_rm_cb.pop_front(tt::constants::TILE_HEIGHT);
                }
                k_in_cb.push_back(Skt * vDHt);
            } else {
                // tilize K rows -> [Skt, DHt] (waits on reader cb_k_rm -> absorbs the K-read stall)
                compute_kernel_lib::tilize<DHt, cb_k_rm, cb_k_in>(
                    /*num_blocks=*/Skt, /*total_input_pages=*/Skt * tt::constants::TILE_HEIGHT);
            }
            // fp8 K tilize leaves srcA in fp8. QK reads K (transposed -> srcA) and Q (srcB), so restore
            // srcA=cb_k_in (bfp8 for fp8 K), srcB=cb_q_in. Reconfigure the tiled descriptor geometry/strides;
            // mm_no_mop_init_short only programs the matmul MOP.
            reconfig_full_operand(cb_k_in, cb_q_in);
            // K tilize also leaves the packer in cb_k_in's format+strides (bfp8 for fp8). Restore bf16 once per
            // chunk for the downstream packs (cb_qk_im/max/sum/out share its geometry); configure_pack_width in
            // the qg loop refreshes only the MOP. No-op for bf16.
            pack_reconfig_data_format(cb_qk_im);

            const bool is_first = (chunk == 0);
            const bool is_last = (chunk == num_active_chunks - 1);
            const bool has_mask = (chunk_base + chunk) == token_chunks - 1;  // only the token's last active chunk

            // cb_qk_im / sum_cur / out_cur span all Sqt rows; the qg loop fills them band-by-band (sub_exp & PV
            // pack at absolute row offsets; reduce/corr reserve+push per band).
            qk_cb.reserve_back(Sqt * KT_stride);
            sum_cur.reserve_back(Sqt);
            out_cur.reserve_back(Sqt * vDHt);
            if constexpr (scaled_kv) {
                k_in_cb.wait_front(Skt * vDHt);
                rope_tile_cb.wait_front(Skt * (DHt - vDHt));
            } else {
                k_in_cb.wait_front(Skt * DHt);  // shared by every query group (QK + PV)
            }
            q_in_cb.wait_front(Sqt * DHt);

            // Boundary geometry (last chunk, same for every row): keys [valid_last, k_chunk) are sentinels ->
            // -inf. Key tiles [full_start, Skt) are fully masked (cb_neginf); a mid-tile boundary masks the
            // straddling tile part_t via cb_mask_part.
            const uint32_t k_chunk = Skt * tt::constants::TILE_WIDTH;
            const uint32_t valid_last = num_valid_keys - (token_chunks - 1) * k_chunk;
            const uint32_t full_start = (valid_last + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
            const uint32_t part_t = valid_last / tt::constants::TILE_WIDTH;
            const bool has_part_mask = has_mask && (valid_last % tt::constants::TILE_WIDTH != 0);

            // DST holds qsb query tile-rows; process Sqt rows in q_groups passes (one pass when qsb==Sqt).
            for (uint32_t qg = 0; qg < q_groups; ++qg) {
                const uint32_t row_base = qg * qsb;  // first query tile-row of this group

                // Set exp to the softmax scale; salad's correction below re-inits it to unit scale.
                exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();

                // ===== Phase 1: Q@Kᵀ -> cb_qk_im band, mask, running row-max =====
                {
                    CZONE("C-QK");
                    if constexpr (scaled_kv) {
                        constexpr uint32_t rope_DHt = DHt - vDHt;
                        // Both partial products read fields from the full-width Q matrix. The matmul stride must
                        // therefore remain DHt even though their contraction widths are vDHt and rope_DHt.
                        constexpr uint32_t q_row_stride = DHt;
                        // Latent contribution from the scaled mixed-format cache.
                        reconfig_data_format(cb_k_in, cb_q_in);
                        mm_no_mop_init_short(cb_q_in, cb_k_in, /*transpose=*/true, 1, qsb, q_row_stride);
                        configure_row_pack_width(cb_qk_im, 1);
                        for (uint32_t kt = 0; kt < Skt; ++kt) {
                            blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/KT_stride>(
                                cb_q_in,
                                cb_k_in,
                                cb_qk_im,
                                /*in0_index_start=*/row_base * DHt,
                                /*in1_index_start=*/kt * vDHt,
                                /*row_subblock_idx=*/qg,
                                /*out_col_offset=*/kt,
                                /*subblock_w=*/1,
                                /*subblock_h=*/qsb,
                                /*inner_dim=*/vDHt,
                                /*matmul_stride=*/q_row_stride,
                                /*skip_pack_configure=*/true);
                        }
                        // Unscaled BF16 RoPE contribution, accumulated into the same held score tiles.
                        reconfig_data_format(cb_k_rope_tile, cb_q_in);
                        mm_no_mop_init_short(cb_q_in, cb_k_rope_tile, /*transpose=*/true, 1, qsb, q_row_stride);
                        configure_row_pack_width(cb_qk_im, 1);
                        pack_reconfig_l1_acc(1);
                        for (uint32_t kt = 0; kt < Skt; ++kt) {
                            blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/KT_stride>(
                                cb_q_in,
                                cb_k_rope_tile,
                                cb_qk_im,
                                /*in0_index_start=*/row_base * DHt + vDHt,
                                /*in1_index_start=*/kt * rope_DHt,
                                /*row_subblock_idx=*/qg,
                                /*out_col_offset=*/kt,
                                /*subblock_w=*/1,
                                /*subblock_h=*/qsb,
                                /*inner_dim=*/rope_DHt,
                                /*matmul_stride=*/q_row_stride,
                                /*skip_pack_configure=*/true);
                        }
                        pack_reconfig_l1_acc(0);
                    } else {
                        // scores[q,sk]=ΣQ·K. in1=K, transpose=true (within-tile transpose => Kᵀ).
                        mm_no_mop_init_short(cb_q_in, cb_k_in, /*transpose=*/true, 1, qsb, DHt);
                        configure_row_pack_width(cb_qk_im, 1);
                        for (uint32_t kt = 0; kt < Skt; ++kt) {
                            blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/KT_stride>(
                                cb_q_in,
                                cb_k_in,
                                cb_qk_im,
                                /*in0_index_start=*/row_base * DHt,
                                /*in1_index_start=*/kt * DHt,
                                /*row_subblock_idx=*/qg,
                                /*out_col_offset=*/kt,
                                /*subblock_w=*/1,
                                /*subblock_h=*/qsb,
                                /*inner_dim=*/DHt,
                                /*matmul_stride=*/DHt,
                                /*skip_pack_configure=*/true);
                        }
                    }
                    // Publish the band to UNPACK while holding wr_ptr (for the in-place mask/sub_exp).
                    cb_push_back_hold_wr_ptr(cb_qk_im, qsb * KT_stride);
                }

                {
                    CZONE("C-SMX");
                    // QK left srcA=cb_k_in, srcB=cb_q_in. The mask/reduce/sub_exp read cb_qk_im (srcA) + bf16
                    // scalers (srcB); restore both to bf16. No-op for bf16 Q/K.
                    reconfig_data_format(cb_qk_im, cb_scale);
                    if (has_mask) {
                        qk_cb.wait_front((qg + 1) * qsb * KT_stride);  // band visible before the in-place mask
                        // Same mask for every query row: stamp each masked key tile across this group's rows
                        // (cb_qk_im row row_base+r, key tile t = (row_base+r)*Skt + t).
                        if (full_start < Skt) {
                            neginf_cb.wait_front(1);
                            add_bcast_rows_init(cb_qk_im, cb_neginf);
                            configure_single_tile_pack(cb_qk_im);
                            for (uint32_t r = 0; r < qsb; ++r) {
                                for (uint32_t t = full_start; t < Skt; ++t) {
                                    add_bcast_row_mask_tile(cb_qk_im, cb_neginf, (row_base + r) * Skt + t);
                                }
                            }
                        }
                        if (has_part_mask) {
                            mask_part_cb.wait_front(1);  // persistent until popped after the group loop
                            add_bcast_rows_init(cb_qk_im, cb_mask_part);
                            configure_single_tile_pack(cb_qk_im);
                            for (uint32_t r = 0; r < qsb; ++r) {
                                add_bcast_row_mask_tile(cb_qk_im, cb_mask_part, (row_base + r) * Skt + part_t);
                            }
                        }
                        pack_to_unpack_sync();  // mask PACK writes must be visible to the row-max UNPACK
                    }
                    // running row-max (MAX-only; eltwise-max against prev on chunk>0)
                    max_cur.reserve_back(qsb);
                    configure_single_tile_pack(max_cur.get_cb_id());
                    reduce_c_row_group<cb_qk_im, cb_scale, KT_stride>(
                        max_cur.get_cb_id(),
                        max_prev.get_cb_id(),
                        /*row_group_index=*/qg,
                        /*do_eltwise_max=*/!is_first,
                        qsb,
                        Skt);
                    max_cur.push_back(qsb);

                    // sub_exp in place: cb_qk_im = exp((cb_qk_im - max)*scale); partial row-sum -> sum_cur (L1-acc).
                    // Walk key-tile columns in exp_sbw steps (one step when the group fits DST); global_col_base
                    // drives the L1-accumulate across steps.
                    for (uint32_t kc = 0; kc < Skt; kc += exp_sbw) {
                        sub_exp_block_bcast_cols<false, scale_fp32>(
                            cb_qk_im,
                            max_cur.get_cb_id(),
                            sum_cur.get_cb_id(),
                            /*cols_in_row=*/KT_stride,
                            /*q_subblock=*/qg,
                            /*global_col_base=*/kc,
                            /*sbh=*/qsb,
                            /*sbw=*/exp_sbw);
                    }
                    pack_to_unpack_sync();  // sub_exp PACK writes must be visible to the V-matmul UNPACK
                }

                // ===== Phase 2: probs@V -> out_cur band =====
                {
                    CZONE("C-PV");
                    qk_cb.wait_front((qg + 1) * qsb * KT_stride);
                    // out[q,vd]=Σprobs·K, V = the latent feature cols. PV reads V into srcA and
                    // probs (cb_qk_im) into srcB (operands swap); set srcA=cb_k_in, srcB=cb_qk_im.
                    reconfig_data_format(cb_k_in, cb_qk_im);
                    mm_no_mop_init_short(cb_qk_im, cb_k_in, /*transpose=*/false, pv_w, qsb, Skt);
                    configure_row_pack_width(out_cur.get_cb_id(), pv_w);
                    for (uint32_t vd = 0; vd < vDHt; vd += pv_w) {
                        blocked_matmul_and_pack<false, /*in1_stride=*/scaled_kv ? vDHt : DHt, /*out_num_cols=*/vDHt>(
                            cb_qk_im,
                            cb_k_in,
                            out_cur.get_cb_id(),
                            /*in0_index_start=*/row_base * Skt,
                            /*in1_index_start=*/vd,
                            /*row_subblock_idx=*/qg,
                            /*out_col_offset=*/vd,
                            /*subblock_w=*/pv_w,
                            /*subblock_h=*/qsb,
                            /*inner_dim=*/Skt,
                            /*matmul_stride=*/KT_stride,
                            /*skip_pack_configure=*/true);
                    }
                    pack_to_unpack_sync();                // flush PV's HELD out_cur packs (push_back deferred to
                                                          // after the group loop) so SALAD's L1-accumulate (!is_first)
                                                          // reads them; normalize reads post-push_back, so covered
                    reconfig_data_format_srca(cb_qk_im);  // PV left srcA in cb_k_in's format; restore bf16
                }

                // ===== SALAD flash combine (skip on the first chunk) =====
                if (!is_first) {
                    CZONE("C-SAL");
                    // correction = exp((prev_max - cur_max)*scale) -> cb_corr col 0. Re-init exp at unit scale
                    // (sub_exp_first_col_blocks applies scale itself; the group's init baked scale_fp32).
                    exp_packthread_tile_init<EXP_APPROX_MODE>();
                    corr_cb.reserve_back(qsb);
                    sub_exp_first_col_blocks<false, scale_fp32>(
                        max_prev.get_cb_id(), max_cur.get_cb_id(), cb_corr, /*q_subblock=*/qg, qsb);
                    corr_cb.push_back(qsb);
                    // cur.out += prev.out*corr ; cur.sum += prev.sum*corr (L1-acc). salad packs width=dst_size,
                    // so restore Default packer geometry first (the per-chunk tilize left it in Tilize layout).
                    PACK((
                        llk_pack_init<ckernel::PackMode::Default, false, false, false>(out_cur.get_cb_id(), dst_size)));
                    pack_reconfig_l1_acc(1);
                    // out_prev & cb_corr consumed front-first per group (popped below); sum_prev cumulative (qg).
                    salad_correct_fused<qsb, vDHt, dst_size>(
                        out_prev.get_cb_id(),
                        sum_prev.get_cb_id(),
                        cb_corr,
                        out_cur.get_cb_id(),
                        sum_cur.get_cb_id(),
                        /*ob_q_subblock=*/0,
                        /*sum_q_subblock=*/qg,
                        /*write_q_subblock=*/qg);
                    pack_reconfig_l1_acc(0);
                    corr_cb.pop_front(qsb);
                    out_prev.pop_front(qsb * vDHt);  // this band's prev.out consumed into cur
                }
            }  // query groups

            // Pop cb_mask_part once: every group bcast-added the same straddling-tile mask.
            if (has_part_mask) {
                mask_part_cb.pop_front(1);
            }
            // prev max/sum consumed into cur (out_prev popped per band above).
            if (!is_first) {
                max_prev.pop_front(Sqt);
                sum_prev.pop_front(Sqt);
            }

            // Publish cur.sum / cur.out (running state for the next chunk, or the final result).
            sum_cur.push_back(Sqt);
            out_cur.push_back(Sqt * vDHt);

            if (is_last && split_k && seg_kind == sparse_sdpa::seg_kind::HEAD) {
                // split-K HEAD: hand the unnormalized state to the writer (-> core i-1, which merges it).
                copy_state_tiles<dst_size>(out_cur.get_cb_id(), cb_send_out, Sqt * vDHt);
                copy_state_tiles<dst_size>(sum_cur.get_cb_id(), cb_send_sum, Sqt);
                copy_state_tiles<dst_size>(max_cur.get_cb_id(), cb_send_max, Sqt);
            } else if (is_last && split_k && seg_kind == sparse_sdpa::seg_kind::TAIL) {
                // split-K TAIL: merge core i+1's HEAD state X into ours R, then normalize as usual.
                //   m = max(m_R, m_X); c_R = exp((m_R-m)*s); c_X = exp((m_X-m)*s)
                //   out = out_R*c_R + out_X*c_X; sum = sum_R*c_R + sum_X*c_X  (into the free prev buffers)
                CircularBuffer merge_ctrl(cb_merge_ctrl);
                merge_ctrl.wait_front(1);
                const uint32_t partial = ckernel::read_tile_value(cb_merge_ctrl, 0, 0);
                merge_ctrl.pop_front(1);
                uint32_t fin_sum = sum_cur.get_cb_id(), fin_out = out_cur.get_cb_id();
                if (partial == sparse_sdpa::SPLIT_PARTIAL_VALID) {
                    CircularBuffer recv_out(cb_recv_out), recv_max(cb_recv_max), recv_sum(cb_recv_sum);
                    CircularBuffer mmax(cb_mmax), corr2(cb_corr2);
                    recv_max.wait_front(Sqt);
                    max_cur.wait_front(Sqt);
                    mmax.reserve_back(Sqt);
                    reconfig_data_format_srca(cb_recv_max);
                    pack_reconfig_data_format(cb_mmax);
                    pack_reconfig_l1_acc(0);
                    configure_single_tile_pack(cb_mmax);
                    for (uint32_t r = 0; r < Sqt; ++r) {
                        tile_regs_acquire();
                        copy_init(max_cur.get_cb_id());
                        copy_tile(max_cur.get_cb_id(), r, 0);
                        copy_init(cb_recv_max);
                        copy_tile(cb_recv_max, r, 1);
                        binary_max_tile_init();
                        binary_max_tile(0, 1, 0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(0, cb_mmax);
                        tile_regs_release();
                    }
                    mmax.push_back(Sqt);
                    exp_packthread_tile_init<EXP_APPROX_MODE>();
                    corr_cb.reserve_back(Sqt);
                    sub_exp_first_col_blocks<false, scale_fp32>(max_cur.get_cb_id(), cb_mmax, cb_corr, 0, Sqt);
                    corr_cb.push_back(Sqt);
                    corr2.reserve_back(Sqt);
                    sub_exp_first_col_blocks<false, scale_fp32>(cb_recv_max, cb_mmax, cb_corr2, 0, Sqt);
                    corr2.push_back(Sqt);
                    // out_prev / sum_prev are empty here (consumed by this chunk's SALAD, or never filled).
                    out_prev.reserve_back(Sqt * vDHt);
                    sum_prev.reserve_back(Sqt);
                    reconfig_data_format(out_cur.get_cb_id(), cb_corr);
                    pack_reconfig_data_format(out_prev.get_cb_id());
                    PACK((llk_pack_init<ckernel::PackMode::Default, false, false, false>(out_prev.get_cb_id(), dst_size)));
                    pack_reconfig_l1_acc(0);
                    salad_correct_fused<Sqt, vDHt, dst_size>(
                        out_cur.get_cb_id(), sum_cur.get_cb_id(), cb_corr, out_prev.get_cb_id(), sum_prev.get_cb_id(), 0, 0, 0);
                    pack_to_unpack_sync();  // first-term packs land before the L1-accumulated second term
                    pack_reconfig_l1_acc(1);
                    salad_correct_fused<Sqt, vDHt, dst_size>(
                        cb_recv_out, cb_recv_sum, cb_corr2, out_prev.get_cb_id(), sum_prev.get_cb_id(), 0, 0, 0);
                    pack_reconfig_l1_acc(0);
                    out_prev.push_back(Sqt * vDHt);
                    sum_prev.push_back(Sqt);
                    out_cur.pop_front(Sqt * vDHt);
                    sum_cur.pop_front(Sqt);
                    recv_out.pop_front(Sqt * vDHt);
                    recv_max.pop_front(Sqt);
                    recv_sum.pop_front(Sqt);
                    corr_cb.pop_front(Sqt);
                    corr2.pop_front(Sqt);
                    mmax.pop_front(Sqt);
                    pack_to_unpack_sync();  // merged packs must be visible to normalize's unpack
                    fin_sum = sum_prev.get_cb_id();
                    fin_out = out_prev.get_cb_id();
                }
                normalize_row_streaming<
                    /*profiling_enabled=*/false,
                    vDHt,
                    dst_size,
                    cb_col_identity,
                    cb_recip_scratch,
                    cb_out_im,
                    scale_fp32,
                    use_attention_sink,
                    cb_attention_sink,
                    sink_per_row>(fin_sum, fin_out, Sqt, max_cur.get_cb_id());
                max_cur.pop_front(Sqt);
            } else if (is_last) {
                // Finalize: row-sum (matmul vs col-identity) -> recip -> out *= 1/sum -> cb_out_im.
                // normalize_row_streaming is DST-safe for any sbh, so it does all Sqt rows in one call.
                normalize_row_streaming<
                    /*profiling_enabled=*/false,
                    vDHt,
                    dst_size,
                    cb_col_identity,
                    cb_recip_scratch,
                    cb_out_im,
                    scale_fp32,
                    use_attention_sink,
                    cb_attention_sink,
                    sink_per_row>(sum_cur.get_cb_id(), out_cur.get_cb_id(), Sqt, max_cur.get_cb_id());
                max_cur.pop_front(Sqt);  // running max no longer needed
            }

            // Release the held cb_qk_im rows + this chunk's K (consumed by QK and PV).
            qk_cb.pop_front(Sqt * KT_stride);
            if constexpr (scaled_kv) {
                k_in_cb.pop_front(Skt * vDHt);
                rope_tile_cb.pop_front(Skt * (DHt - vDHt));
            } else {
                k_in_cb.pop_front(Skt * DHt);
            }

            swap_cb(max_prev, max_cur);  // prev <-> cur for the next chunk
            swap_cb(sum_prev, sum_cur);
            swap_cb(out_prev, out_cur);
        }

        q_in_cb.pop_front(Sqt * DHt);  // Q reused across all chunks; drop it so >1 token/core stays clean

        // cb_out_im was written by normalize_row_streaming; untilize -> row-major out for the writer.
        if (!(split_k && seg_kind == sparse_sdpa::seg_kind::HEAD)) {
            compute_kernel_lib::untilize<vDHt, cb_out_im, cb_out_rm>(/*num_blocks=*/Sqt);
        }
    }
}
