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
#include "../sparse_sdpa_common.hpp"
constexpr bool EXP_APPROX_MODE = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::MATH_APPROX_MODE) != 0;

#ifdef SPARSE_SDPA_FP32_SCORES
// fp32_scores: the FPU reads the Float32 scores / probabilities at TF32 (10 mantissa bits) by truncation, a coherent
// bias toward zero (scores: a flatter softmax). Each DEST tile is rounded to TF32 to nearest-even on the pack thread
// before it is packed (as the softmax exp runs there), so every later FPU read is exact and the rounding unbiased.
#if defined(TRISC_PACK)
namespace ckernel::sfpu {
template <int ITERATIONS = 8>
inline void calculate_round_tf32_rne() {
    for (int d = 0; d < ITERATIONS; ++d) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vUInt u = sfpi::as<sfpi::vUInt>(v);
        sfpi::vUInt lsb = (u >> 13) & 1;
        u = u + lsb;
        u = u + sfpi::vUInt(0xFFF);  // sign-magnitude: the add rounds the magnitude of either sign
        u = u & sfpi::vUInt(0xFFFFE000u);
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(u);
        sfpi::dst_reg++;
    }
}
#ifdef SPARSE_SDPA_BF16_PROBS
// bf16_probs: the probabilities rounded to bf16 to nearest-even (DeepSeek's sparse_attn casts P to bf16 before P@V);
// the Float32 CB then holds bf16 values, which the FPU reads exactly. Inputs are exp() values in [0, 1].
template <int ITERATIONS = 8>
inline void calculate_round_bf16_rne() {
    for (int d = 0; d < ITERATIONS; ++d) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vUInt u = sfpi::as<sfpi::vUInt>(v);
        sfpi::vUInt lsb = (u >> 16) & 1;
        u = u + lsb;
        u = u + sfpi::vUInt(0x7FFF);  // sign-magnitude: the add rounds the magnitude of either sign
        u = u & sfpi::vUInt(0xFFFF0000u);
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(u);
        sfpi::dst_reg++;
    }
}
#endif
// DEST tile 0 = round_tf32(tile 0 + tile 1 + ... + tile N-1), fp32 SFPU adds in that order.
template <int ITERATIONS, uint32_t N>
inline void calculate_sum_round_tf32_rne() {
    constexpr uint32_t dst_tile_size_sfpi = 32;  // one DEST tile in sfpi rows
    for (int d = 0; d < ITERATIONS; ++d) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        for (uint32_t j = 1; j < N; ++j) {
            v = v + sfpi::dst_reg[j * dst_tile_size_sfpi];
        }
        sfpi::vUInt u = sfpi::as<sfpi::vUInt>(v);
        sfpi::vUInt lsb = (u >> 13) & 1;
        u = u + lsb;
        u = u + sfpi::vUInt(0xFFF);
        u = u & sfpi::vUInt(0xFFFFE000u);
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(u);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
// Pack thread only (the softmax exp runs there too): the SFPU is one unit, and the math thread's next tiles overlap
// this thread's SFPU work on the other DEST half, so the math thread must not issue SFPU work meanwhile.
ALWI void round_tf32_packthread_tile(uint32_t idst) {
    PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_round_tf32_rne, (8), idst, VectorMode::RC)));
}
#ifdef SPARSE_SDPA_BF16_PROBS
ALWI void round_bf16_packthread_tile(uint32_t idst) {
    PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_round_bf16_rne, (8), idst, VectorMode::RC)));
}
#endif
template <uint32_t N>
ALWI void sum_round_tf32_packthread_tile0() {
    PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sum_round_tf32_rne, (8, N), 0, VectorMode::RC)));
}
#endif
// compute_streaming.hpp needs declarations from compute_common.hpp (LightweightMaskContext, reduce helpers,
// DEST_AUTO_LIMIT); include it first. Only compute_streaming primitives are used.
#include "compute_common.hpp"
#include "compute_streaming.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/dataflow/circular_buffer.h"  // CircularBuffer: COMPILE_FOR_TRISC-aware CB lifecycle
#include "api/tile_constants.h"            // tt::constants::TILE_HEIGHT

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

#ifdef SPARSE_SDPA_ROW_BFP4
#ifdef TRISC_UNPACK
#include "llk_unpack_common.h"
#include "ckernel_template.h"
// ROW_BFP4 (fork option): the gathered staging block [K_CHUNK rows' exponents | their nibbles] is a row-major bfloat4_b
// tensor; the unpacker reads it by datum index with the tile descriptor x = 16, y = K_DIM / 16 groups, z = K_CHUNK
// rows: datum ((z y_dim) + y) 16 is group y of row z, its exponent byte z y_dim + y of the section. One UNPACR reads a
// face row (16 values, 1 exponent) into the next SrcA row (z += 1, SrcA row += 1); the MOP does a face: 16 of them, the
// last with SrcA valid, then SrcB valid (as the plain copy's MOP). No data-movement RISC touches the bytes.
namespace row_bfp4_unpack {
using namespace ckernel;
using namespace ckernel::unpacker;

template <uint32_t K_DIM, uint32_t K_CHUNK>
inline void init() {
    static constexpr uint32_t row = TT_OP_UNPACR(
        SrcA, 0b01000001 /*SrcA row += 1, z += 1*/, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint32_t row_valid =
        TT_OP_UNPACR(SrcA, 0b01000001, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint32_t srcb_valid = TT_OP_UNPACR_NOP(
        SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xFFFFFFFF>(16 | (16 << 16));
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, 0xFFFFFFFF>((K_DIM / 16) | (K_CHUNK << 16));
    cfg_reg_rmw_tensix<
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32,
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT,
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(16);  // one SrcA row
    TTI_SETADCXX(p_setadc::UNP_A, 16 - 1, 0x0);
    // the matmul's unpack steps tiles through the offset register (InAddr = base + offset): start from the base
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC0_REG7_Offset_address_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC0_REG7_Offset_cntx1_address_ADDR32);
    TTI_NOP;
    ckernel_template mop(1, 16, row);
    // with one outer iteration the last-outer replacement is the one that applies to the 16th UNPACR: set both
    mop.set_last_inner_loop_instr(row_valid);
    mop.set_last_outer_loop_instr(row_valid);
    mop.set_end_op(srcb_valid);
    mop.program();
}

inline void uninit() {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);
    TTI_NOP;
    TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1010);  // ch0 / ch1 Y
    TTI_SETADCZW(0b001, 0, 0, 0, 0, 0b1111);
}

// tile (r0 / 32, g0 / 2): faces (rows r0.., group g0), (r0.., g0 + 1), (r0 + 16.., g0), (r0 + 16.., g0 + 1)
inline void tile(uint32_t base16, uint32_t r0, uint32_t g0) {
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    wait_for_next_context(2);
    _llk_unpack_configure_single_address_(base16, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    for (uint32_t f = 0; f < 4; ++f) {
        TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1000);  // SrcA row = 0
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Z, r0 + (f >> 1) * 16);
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, g0 + (f & 1));
        ckernel_template::run();
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
}  // namespace row_bfp4_unpack
#endif

// The chunk's staging block (cb_k4_rm, one page, 64 B aligned within it) -> its K tiles [Skt, DHt] in cb_k4: the
// unpacker reads the rows (row_bfp4_unpack), the stock datacopy -> DEST, the pack to bfloat4_b (the values are on the
// bfp4 grid: the pack copies them). The bfloat8_b window (SPARSE_SDPA_WINDOW_BFP8) the same way, by its CBs' format.
template <uint32_t K_DIM, uint32_t Skt, uint32_t DHt>
ALWI void row_bfp4_to_tiles(uint32_t cb_k4_rm, uint32_t cb_k4, uint32_t cb_k_in) {
    constexpr uint32_t K_CHUNK = Skt * tt::constants::TILE_HEIGHT;
    cb_wait_front(cb_k4_rm, 1);
    cb_reserve_back(cb_k4, Skt * DHt);
    reconfig_data_format_srca(cb_k4_rm);
    pack_reconfig_data_format(cb_k4);
    pack_init(cb_k4);         // the packer's tile geometry (the attention reshapes it for its row packs)
    pack_reconfig_l1_acc(0);  // plain packs (the attention L1-accumulates its running output)
    copy_init(cb_k4_rm);
    UNPACK((row_bfp4_unpack::init<K_DIM, K_CHUNK>()));
    for (uint32_t tr = 0; tr < Skt; ++tr) {
        for (uint32_t c = 0; c < DHt; ++c) {
            tile_regs_acquire();
            UNPACK((row_bfp4_unpack::tile(
                (((get_local_cb_interface(cb_k4_rm).fifo_rd_ptr << 4) + 63) & ~63u) / 16 - 1, tr * 32, 2 * c)));
            MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, false>(
                0, cb_k4_rm)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_k4);
            tile_regs_release();
        }
    }
    UNPACK((row_bfp4_unpack::uninit()));
    // leave the packer as the BF16 path's tilize does (tilize_uninit: a default pack init on the bf16 K CB), not in
    // the bfloat4_b output setup (its exponent section): the attention's packs follow
    pack_reconfig_data_format(cb_k_in);
    pack_init(cb_k_in);
    cb_push_back(cb_k4, Skt * DHt);
    cb_pop_front(cb_k4_rm, 1);
}
#endif

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
#ifdef SPARSE_SDPA_ROW_BFP4
    // ROW_BFP4 (fork option): chunks >= 1 arrive as bfloat4_b tiles the dataflow kernels shuffled from row records
    constexpr uint32_t cb_k4 = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::END);
    constexpr uint32_t cb_k4_rm = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::END + 1);
#ifdef SPARSE_SDPA_WINDOW_BFP8
    // the window (chunk 0) as bfloat8_b row records: its staging block -> bfloat8_b tiles, as the bfloat4_b chunks
    constexpr uint32_t cb_k8 = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::END + 2);
    constexpr uint32_t cb_k8_rm = get_compile_time_arg_val(sparse_sdpa::compute_ct_arg::END + 3);
#endif
#endif
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

    const uint32_t tok_count = get_arg_val<uint32_t>(1);

    // Q tilize is the first operation, so startup must configure raw-Q -> tiled-Q.
    compute_kernel_hw_startup(cb_q_rm, cb_q_in);

    scale_cb.wait_front(1);  // persistent reduce scaler; the streaming reduce assumes it is ready

    for (uint32_t tok = 0; tok < tok_count; ++tok) {
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
#ifdef SPARSE_SDPA_ROW_BFP4
        const uint32_t window_valid = ckernel::read_tile_value(
            cb_ctrl, /*tile=*/0, /*element_offset=*/sparse_sdpa::control_message::WINDOW_VALID);
#endif
        ctrl_cb.pop_front(1);

        // Flash running state, ping-pong. Reset every token; all buffers start empty.
        CircularBuffer max_prev(cb_max_a), max_cur(cb_max_b);
        CircularBuffer sum_prev(cb_sum_a), sum_cur(cb_sum_b);
        CircularBuffer out_prev(cb_out_a), out_cur(cb_out_b);

        for (uint32_t chunk = 0; chunk < num_active_chunks; ++chunk) {
#ifdef SPARSE_SDPA_ROW_BFP4
            const bool bfp4_chunk = chunk > 0;  // chunk 0 = the bf16 window, tilized below
#ifdef SPARSE_SDPA_WINDOW_BFP8
            const uint32_t cb_k = bfp4_chunk ? cb_k4 : cb_k8;
#else
            const uint32_t cb_k = bfp4_chunk ? cb_k4 : cb_k_in;
#endif
#else
            constexpr uint32_t cb_k = cb_k_in;
#endif
            CircularBuffer k_cur(cb_k);
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

                k_in_cb.reserve_back(Skt * vDHt);
                for (uint32_t row_tile = 0; row_tile < Skt; ++row_tile) {
                    k_rm_cb.wait_front(tt::constants::TILE_HEIGHT);
                    {
                        tilize_packed_field<
                            cb_k_rm,
                            cb_k_latent_tile,
                            latent_physical_ct,
                            latent_DHt,
                            /*field_offset_16b=*/0,
                            /*manage_input_lifecycle=*/false>();
                    }

                    {
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

                    {
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
#if defined(SPARSE_SDPA_ROW_BFP4) && defined(SPARSE_SDPA_WINDOW_BFP8)
                if (bfp4_chunk) {
                    row_bfp4_to_tiles<DHt * tt::constants::TILE_WIDTH, Skt, DHt>(cb_k4_rm, cb_k4, cb_k_in);
                } else {
                    row_bfp4_to_tiles<DHt * tt::constants::TILE_WIDTH, Skt, DHt>(cb_k8_rm, cb_k8, cb_k_in);
                }
#else
#ifdef SPARSE_SDPA_ROW_BFP4
                if (bfp4_chunk) {
                    row_bfp4_to_tiles<DHt * tt::constants::TILE_WIDTH, Skt, DHt>(cb_k4_rm, cb_k4, cb_k_in);
                } else
#endif
                    compute_kernel_lib::tilize<DHt, cb_k_rm, cb_k_in>(
                        /*num_blocks=*/Skt, /*total_input_pages=*/Skt * tt::constants::TILE_HEIGHT);
#endif
            }
            // fp8 K tilize leaves srcA in fp8. QK reads K (transposed -> srcA) and Q (srcB), so restore
            // srcA=cb_k_in (bfp8 for fp8 K), srcB=cb_q_in. Reconfigure the tiled descriptor geometry/strides;
            // mm_no_mop_init_short only programs the matmul MOP.
            reconfig_full_operand(cb_k, cb_q_in);
            // K tilize also leaves the packer in cb_k_in's format+strides (bfp8 for fp8). Restore bf16 once per
            // chunk for the downstream packs (cb_qk_im/max/sum/out share its geometry); configure_pack_width in
            // the qg loop refreshes only the MOP. No-op for bf16.
            pack_reconfig_data_format(cb_qk_im);

            const bool is_first = (chunk == 0);
            const bool is_last = (chunk == num_active_chunks - 1);
#ifdef SPARSE_SDPA_ROW_BFP4
            // the window chunk has its own sentinel tail; the top-k chunks' tail ends the list
            const bool has_mask = is_last || (chunk == 0 && window_valid < Skt * tt::constants::TILE_WIDTH);
#else
            const bool has_mask = is_last;  // only the last active chunk can hold sentinels
#endif

            // cb_qk_im / sum_cur / out_cur span all Sqt rows; the qg loop fills them band-by-band (sub_exp & PV
            // pack at absolute row offsets; reduce/corr reserve+push per band).
            qk_cb.reserve_back(Sqt * KT_stride);
            sum_cur.reserve_back(Sqt);
            out_cur.reserve_back(Sqt * vDHt);
            if constexpr (scaled_kv) {
                k_in_cb.wait_front(Skt * vDHt);
                rope_tile_cb.wait_front(Skt * (DHt - vDHt));
            } else {
                k_cur.wait_front(Skt * DHt);  // shared by every query group (QK + PV)
            }
            q_in_cb.wait_front(Sqt * DHt);

            // Boundary geometry (last chunk, same for every row): keys [valid_last, k_chunk) are sentinels ->
            // -inf. Key tiles [full_start, Skt) are fully masked (cb_neginf); a mid-tile boundary masks the
            // straddling tile part_t via cb_mask_part.
            const uint32_t k_chunk = Skt * tt::constants::TILE_WIDTH;
#ifdef SPARSE_SDPA_ROW_BFP4
            const uint32_t valid_last = chunk == 0 ? window_valid : num_valid_keys - (num_active_chunks - 1) * k_chunk;
#else
            const uint32_t valid_last = num_valid_keys - (num_active_chunks - 1) * k_chunk;
#endif
            const uint32_t full_start = (valid_last + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
            const uint32_t part_t = valid_last / tt::constants::TILE_WIDTH;
            const bool has_part_mask = has_mask && (valid_last % tt::constants::TILE_WIDTH != 0);

            // DST holds qsb query tile-rows; process Sqt rows in q_groups passes (one pass when qsb==Sqt).
            for (uint32_t qg = 0; qg < q_groups; ++qg) {
                const uint32_t row_base = qg * qsb;  // first query tile-row of this group

                // Set exp to the softmax scale; salad's correction below re-inits it to unit scale.
#ifdef SPARSE_SDPA_HIGH_PRECISION
                exp_packthread_tile_init<false>();  // exact exp; sub_exp applies the scale (see compute_streaming.hpp)
#else
                exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
#endif

                // ===== Phase 1: Q@Kᵀ -> cb_qk_im band, mask, running row-max =====
                {
#ifdef SPARSE_SDPA_FP32_SCORES
                    // Float32 scores: the previous group left the bf16 state (srcA / srcB / packer); QK reads
                    // srcA=cb_k, srcB=cb_q_in and packs the Float32 cb_qk_im.
                    reconfig_data_format(cb_k, cb_q_in);
                    pack_reconfig_data_format(cb_qk_im);
#endif
                    if constexpr (scaled_kv) {
                        constexpr uint32_t rope_DHt = DHt - vDHt;
                        // Both partial products read fields from the full-width Q matrix. The matmul stride must
                        // therefore remain DHt even though their contraction widths are vDHt and rope_DHt.
                        constexpr uint32_t q_row_stride = DHt;
                        // Latent contribution from the scaled mixed-format cache.
                        reconfig_data_format(cb_k, cb_q_in);
                        mm_no_mop_init_short(cb_q_in, cb_k, /*transpose=*/true, 1, qsb, q_row_stride);
                        configure_row_pack_width(cb_qk_im, 1);
                        for (uint32_t kt = 0; kt < Skt; ++kt) {
                            blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/KT_stride>(
                                cb_q_in,
                                cb_k,
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
                        mm_no_mop_init_short(cb_q_in, cb_k, /*transpose=*/true, 1, qsb, DHt);
                        configure_row_pack_width(cb_qk_im, 1);
#ifdef SPARSE_SDPA_FP32_SCORES
                        // blocked_matmul_and_pack with the scores rounded to TF32 (nearest) in DEST before the pack
                        for (uint32_t kt = 0; kt < Skt; ++kt) {
                            tile_regs_acquire();
                            for (uint32_t inner = 0; inner < DHt; ++inner) {
                                sdpa_matmul_block_no_mop(
                                    cb_q_in,
                                    cb_k,
                                    row_base * DHt + inner,
                                    kt * DHt + inner,
                                    /*dst_index=*/0,
                                    /*transpose=*/true,
                                    /*subblock_w=*/1,
                                    /*subblock_h=*/qsb,
                                    /*matmul_stride=*/DHt);
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            for (uint32_t t = 0; t < qsb; ++t) {
                                round_tf32_packthread_tile(t);
                            }
                            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                            pack_contiguous_rows_nocfg(cb_qk_im, qg * qsb, qsb, KT_stride, kt, 1);
                            tile_regs_release();
                        }
#else
                        for (uint32_t kt = 0; kt < Skt; ++kt) {
                            blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/KT_stride>(
                                cb_q_in,
                                cb_k,
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
#endif
                    }
                    // Publish the band to UNPACK while holding wr_ptr (for the in-place mask/sub_exp).
                    cb_push_back_hold_wr_ptr(cb_qk_im, qsb * KT_stride);
                }

                {
                    // QK left srcA=cb_k, srcB=cb_q_in. The mask/reduce/sub_exp read cb_qk_im (srcA) + bf16
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
#ifdef SPARSE_SDPA_FP32_SCORES
                    // the max from the Float32 scores: bf16 (fp32_scores), Float32 with bf16_probs (TF32-exact)
                    pack_reconfig_data_format(max_cur.get_cb_id());
#endif
                    configure_single_tile_pack(max_cur.get_cb_id());
                    reduce_c_row_group<cb_qk_im, cb_scale, KT_stride>(
                        max_cur.get_cb_id(),
                        max_prev.get_cb_id(),
                        /*row_group_index=*/qg,
                        /*do_eltwise_max=*/!is_first,
                        qsb,
                        Skt);
                    max_cur.push_back(qsb);
#ifdef SPARSE_SDPA_FP32_SCORES
                    pack_reconfig_data_format(cb_qk_im);  // sub_exp packs the Float32 probabilities in place
#endif

                    // sub_exp in place: cb_qk_im = exp((cb_qk_im - max)*scale); partial row-sum -> sum_cur (L1-acc).
                    // Walk key-tile columns in exp_sbw steps (one step when the group fits DST); global_col_base
                    // drives the L1-accumulate across steps.
#ifdef SPARSE_SDPA_BF16_PROBS
                    reconfig_data_format_srcb(max_cur.get_cb_id());  // the Float32 max (srcB of the subtraction)
#endif
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
#ifdef SPARSE_SDPA_BF16_PROBS
                    reconfig_data_format_srcb(cb_scale);  // back to the bf16 srcB state
#endif
#ifdef SPARSE_SDPA_HIGH_PRECISION
                    // Row-sum partial from the packed bf16 probs (the ones PV multiplies), exact in the Float32
                    // DEST and L1-accumulated into the Float32 sum_cur: sum_cur[r] = sum_kt probs[r, kt].
                    qk_cb.wait_front((qg + 1) * qsb * KT_stride);
                    copy_tile_to_dst_init_short(cb_qk_im);
                    pack_reconfig_data_format(sum_cur.get_cb_id());
                    configure_single_tile_pack(sum_cur.get_cb_id());
#ifdef SPARSE_SDPA_FP32_SCORES
                    if constexpr (Skt <= dst_size) {
                        // fp32_scores: the row's Skt prob tiles (TF32-exact) summed in DEST by SFPU fp32 adds and
                        // rounded to TF32 (nearest) on the pack thread, packed once: the normalize's identity matmul
                        // then reads the row-sum exactly
                        for (uint32_t r = 0; r < qsb; ++r) {
                            tile_regs_acquire();
                            for (uint32_t j = 0; j < Skt; ++j) {
                                copy_tile(cb_qk_im, (row_base + r) * KT_stride + j, j);
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            sum_round_tf32_packthread_tile0<Skt>();
                            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                            pack_tile<true>(0, sum_cur.get_cb_id(), row_base + r);
                            tile_regs_release();
                        }
                    } else
#endif
                        for (uint32_t r = 0; r < qsb; ++r) {
                            for (uint32_t kt0 = 0; kt0 < Skt; kt0 += dst_size) {
                                const uint32_t n = (Skt - kt0 < dst_size) ? (Skt - kt0) : dst_size;
                                tile_regs_acquire();
                                for (uint32_t j = 0; j < n; ++j) {
                                    copy_tile(cb_qk_im, (row_base + r) * KT_stride + kt0 + j, j);
                                }
                                tile_regs_commit();
                                tile_regs_wait();
                                for (uint32_t j = 0; j < n; ++j) {
                                    PACK((llk_pack_reconfig_l1_acc(kt0 + j > 0 ? 1 : 0)));
                                    pack_tile<true>(j, sum_cur.get_cb_id(), row_base + r);
                                }
                                tile_regs_release();
                            }
                        }
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    pack_reconfig_data_format(cb_qk_im);
#ifdef SPARSE_SDPA_BF16_PROBS
                    // bf16_probs: the row-sum above came from the TF32 probabilities (DeepSeek's kernel sums the fp32
                    // ones); now round them to bf16 (nearest) in place, the operand of probs@V (the kernel's bf16 P)
                    configure_single_tile_pack(cb_qk_im);
                    for (uint32_t r = 0; r < qsb; ++r) {
                        for (uint32_t kt0 = 0; kt0 < Skt; kt0 += dst_size) {
                            const uint32_t n = (Skt - kt0 < dst_size) ? (Skt - kt0) : dst_size;
                            const uint32_t t0 = (row_base + r) * KT_stride + kt0;
                            tile_regs_acquire();
                            for (uint32_t j = 0; j < n; ++j) {
                                copy_tile(cb_qk_im, t0 + j, j);
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            for (uint32_t j = 0; j < n; ++j) {
                                round_bf16_packthread_tile(j);
                            }
                            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                            for (uint32_t j = 0; j < n; ++j) {
                                pack_tile<true>(j, cb_qk_im, t0 + j);
                            }
                            tile_regs_release();
                        }
                    }
                    pack_to_unpack_sync();  // the in-place bf16 probabilities must be visible to the PV UNPACK
#endif
#endif
                }

                // ===== Phase 2: probs@V -> out_cur band =====
                {
                    qk_cb.wait_front((qg + 1) * qsb * KT_stride);
                    // out[q,vd]=Σprobs·K, V = the latent feature cols. PV reads V into srcA and
                    // probs (cb_qk_im) into srcB (operands swap); set srcA=cb_k, srcB=cb_qk_im.
                    reconfig_data_format(cb_k, cb_qk_im);
                    mm_no_mop_init_short(cb_qk_im, cb_k, /*transpose=*/false, 1, qsb, Skt);
#ifdef SPARSE_SDPA_HIGH_PRECISION
                    pack_reconfig_data_format(out_cur.get_cb_id());  // Float32 running out
#endif
                    configure_row_pack_width(out_cur.get_cb_id(), 1);
#ifdef SPARSE_SDPA_FP32_SCORES
                    // blocked_matmul_and_pack with the running output rounded to TF32 (nearest) in DEST before the
                    // pack: the SALAD correction and the normalize read it through the FPU (TF32, truncating)
                    for (uint32_t vd = 0; vd < vDHt; ++vd) {
                        tile_regs_acquire();
                        for (uint32_t inner = 0; inner < Skt; ++inner) {
                            sdpa_matmul_block_no_mop(
                                cb_qk_im,
                                cb_k,
                                row_base * Skt + inner,
                                vd + inner * DHt,
                                /*dst_index=*/0,
                                /*transpose=*/false,
                                /*subblock_w=*/1,
                                /*subblock_h=*/qsb,
                                /*matmul_stride=*/KT_stride);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t t = 0; t < qsb; ++t) {
                            round_tf32_packthread_tile(t);
                        }
                        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                        pack_contiguous_rows_nocfg(out_cur.get_cb_id(), qg * qsb, qsb, vDHt, vd, 1);
                        tile_regs_release();
                    }
#else
                    for (uint32_t vd = 0; vd < vDHt; ++vd) {
                        blocked_matmul_and_pack<false, /*in1_stride=*/scaled_kv ? vDHt : DHt, /*out_num_cols=*/vDHt>(
                            cb_qk_im,
                            cb_k,
                            out_cur.get_cb_id(),
                            /*in0_index_start=*/row_base * Skt,
                            /*in1_index_start=*/vd,
                            /*row_subblock_idx=*/qg,
                            /*out_col_offset=*/vd,
                            /*subblock_w=*/1,
                            /*subblock_h=*/qsb,
                            /*inner_dim=*/Skt,
                            /*matmul_stride=*/KT_stride,
                            /*skip_pack_configure=*/true);
                    }
#endif
                    pack_to_unpack_sync();                // flush PV's HELD out_cur packs (push_back deferred to
                                                          // after the group loop) so SALAD's L1-accumulate (!is_first)
                                                          // reads them; normalize reads post-push_back, so covered
#ifdef SPARSE_SDPA_FP32_SCORES
                    // cb_qk_im is Float32: restore the bf16 unpack (srcA and srcB; PV read the probs into srcB) and
                    // pack state from bf16 CBs (the SALAD correction and the normalize expect it)
                    reconfig_data_format(cb_scale, cb_scale);
                    pack_reconfig_data_format(cb_corr);
#else
                    reconfig_data_format_srca(cb_qk_im);  // PV left srcA in cb_k_in's format; restore bf16
#ifdef SPARSE_SDPA_HIGH_PRECISION
                    pack_reconfig_data_format(cb_qk_im);  // back to the bf16 pack state
#endif
#endif
                }

                // ===== SALAD flash combine (skip on the first chunk) =====
                if (!is_first) {
                    // correction = exp((prev_max - cur_max)*scale) -> cb_corr col 0. Re-init exp at unit scale
                    // (sub_exp_first_col_blocks applies scale itself; the group's init baked scale_fp32).
                    exp_packthread_tile_init<EXP_APPROX_MODE>();
                    corr_cb.reserve_back(qsb);
#ifdef SPARSE_SDPA_BF16_PROBS
                    reconfig_data_format(max_prev.get_cb_id(), max_cur.get_cb_id());  // the Float32 maxima
#endif
                    sub_exp_first_col_blocks<false, scale_fp32>(
                        max_prev.get_cb_id(), max_cur.get_cb_id(), cb_corr, /*q_subblock=*/qg, qsb);
#ifdef SPARSE_SDPA_BF16_PROBS
                    reconfig_data_format(cb_scale, cb_scale);
#endif
                    corr_cb.push_back(qsb);
                    // cur.out += prev.out*corr ; cur.sum += prev.sum*corr (L1-acc). salad packs width=dst_size,
                    // so restore Default packer geometry first (the per-chunk tilize left it in Tilize layout).
#ifdef SPARSE_SDPA_HIGH_PRECISION
                    reconfig_data_format(out_prev.get_cb_id(), cb_corr);
                    pack_reconfig_data_format(out_cur.get_cb_id());
#endif
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
#ifdef SPARSE_SDPA_HIGH_PRECISION
#ifdef SPARSE_SDPA_FP32_SCORES
                    reconfig_data_format(cb_scale, cb_scale);  // back to the bf16 unpack / pack state
                    pack_reconfig_data_format(cb_corr);
#else
                    reconfig_data_format(cb_qk_im, cb_qk_im);  // back to the bf16 unpack / pack state
                    pack_reconfig_data_format(cb_qk_im);
#endif
#endif
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

            if (is_last) {
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
                k_cur.pop_front(Skt * DHt);
            }

            swap_cb(max_prev, max_cur);  // prev <-> cur for the next chunk
            swap_cb(sum_prev, sum_cur);
            swap_cb(out_prev, out_cur);
        }

        q_in_cb.pop_front(Sqt * DHt);  // Q reused across all chunks; drop it so >1 token/core stays clean

        // cb_out_im was written by normalize_row_streaming; untilize -> row-major out for the writer.
        compute_kernel_lib::untilize<vDHt, cb_out_im, cb_out_rm>(/*num_blocks=*/Sqt);
    }
}
