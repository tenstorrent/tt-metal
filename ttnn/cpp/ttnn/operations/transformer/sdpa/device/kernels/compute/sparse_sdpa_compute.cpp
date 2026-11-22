// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sparse SDPA compute on the precision recipes (host: sparse_sdpa_program_factory.cpp). The recipe core loop
// (recipe/recipe_streaming.hpp) is dense SDPA's; this kernel is the sparse outer loop. A segment is one token's query
// rows (one row per head, H / 32 tile rows) against the K rows its index list selects, gathered by the reader into
// row-major chunks. Per segment and head pass (a pass is the Q tile rows that fit L1 at once): tilize the pass's Q
// rows, then per step tilize the gathered K rows of one or more chunks (the host's step group) into the recipe's K^T
// layout (scaled FP8: dequantize the latent field and tilize the RoPE field) and run one recipe step; the step holding
// the token's last chunk masks its sentinel tail and, when it holds fewer chunks than the group, its zero-filled rest
// (pack-thread key mask, recipe_tail.hpp). V is K's leading columns, read in place from the K^T chunk. Split-K
// (sparse_sdpa_common.hpp seg_kind): a HEAD segment hands its unnormalized state to the writer (the recipe checkpoint
// protocol), which sends it to core i-1; a TAIL segment merges that state into its own before normalizing.

#include <cstdint>

// The tilizes, the dequant and the recipe step reconfigure many times: the LLK Src zero-flag configurator stays out of
// line (kernel config buffer).
#define LLK_ZEROFLAG_OUTLINE 1

// Optimization: unpack and pack at the kernel's level (the host picks it per recipe; watcher builds: -Os), math at -O2
// (the fast untilize's LLK needs its constants folded, which -Os does not do). Blackhole only.
#if defined(TRISC_MATH)
#pragma GCC optimize("O2")
#elif defined(WATCHER_ENABLED)
#pragma GCC optimize("Os")
#endif

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/tile_constants.h"
#include "compute_common.hpp"
#include "recipe/recipe_tail.hpp"
#include "recipe/recipe_sfpu.hpp"
#include "recipe/recipe_streaming.hpp"
#include "recipe/recipe_ring.hpp"
#include "sparse_recipe.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "sparse_sdpa_fused_dequant.hpp"

namespace {
namespace cb = sparse_sdpa::recipe_cb;
using namespace recipe_ring_cb;

constexpr uint32_t q_tiles = get_named_compile_time_arg_val("q_tiles");        // Q tile rows per pass (with padding)
constexpr uint32_t head_tiles = get_named_compile_time_arg_val("head_tiles");  // the token's Q tile rows (H / 32)
constexpr uint32_t passes = get_named_compile_time_arg_val("passes");
constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");            // one recipe step
constexpr uint32_t gather_tiles = get_named_compile_time_arg_val("gather_tiles");  // one gathered chunk
constexpr uint32_t d_tiles = get_named_compile_time_arg_val("d_tiles");
constexpr uint32_t vd_tiles = get_named_compile_time_arg_val("vd_tiles");
constexpr uint32_t scale_fp32 = get_named_compile_time_arg_val("scale");
constexpr bool v_in_k = get_named_compile_time_arg_val("v_in_k") != 0;
constexpr bool scaled_kv = get_named_compile_time_arg_val("scaled_kv") != 0;
constexpr bool fused_dequant = get_named_compile_time_arg_val("fused_dequant") != 0;
constexpr uint32_t packed_row_bytes = get_named_compile_time_arg_val("packed_row_bytes");
constexpr bool split_k = get_named_compile_time_arg_val("split_k") != 0;
static_assert(recipe::ring && recipe::ring_mask == 0 && !recipe::ring_stream_state, "the host's sparse recipe");
static_assert(!split_k || passes == 1, "split-K merges one pass per token");
static_assert(!fused_dequant || (scaled_kv && !recipe::sink), "the fused dequant scales V (undone on the output)");

constexpr uint32_t cb_v = v_in_k ? k : v;
constexpr uint32_t k_chunk_keys = gather_tiles * tt::constants::TILE_WIDTH;  // one gathered chunk
constexpr uint32_t group = k_tiles / gather_tiles;                           // gathered chunks per step
static_assert(k_tiles % gather_tiles == 0);
constexpr uint32_t rope_tiles = d_tiles - vd_tiles;
using sparse_recipe::restore_state;

// Tilize `tiles` tiles of the fronted 32-row slab of row-major cb_rm (from tile column first_in) into tiles out0 +
// t * out_stride of cb_out (reserved by the caller): the K^T layout puts a K row's tile d at d * k_tiles. The caller
// ran tilize_init with the slab's physical width.
template <uint32_t cb_rm, uint32_t cb_out>
ALWI void tilize_slab_strided(uint32_t tiles, uint32_t first_in, uint32_t out0, uint32_t out_stride) {
    for (uint32_t t = 0; t < tiles; ++t) {
        UNPACK((llk_unpack_tilize(cb_rm, first_in + t)));
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));
        MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
            0, cb_rm)));
        PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(0, cb_out, out0 + t * out_stride)));
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// The pass's Q rows (q_tiles tile rows of the reader's row-major pages, zero rows past the token's heads) into the
// recipe Q CB. Fused dequant: Q's latent columns times 2^-OFFSET (the K latent carries 2^OFFSET; Q.K unchanged).
// The sparse wrapper's own steps (tilizes, dequant, merge, output) build for size: the kernel config buffer holds them
// beside the recipe step (an FP32 recipe with scaled FP8 at 96 heads was 1.5 KB over at -O2).
#define SPARSE_COLD __attribute__((noinline, optimize("Os")))

SPARSE_COLD void tilize_q() {
    sparse_recipe::tilize_q<cb::q_rm, d_tiles, q_tiles>();
    if constexpr (fused_dequant) {
        sparse_sdpa::fused_dequant::prescale_q_latent<q, q_tiles, d_tiles, vd_tiles>();
        recipe_pack_to_unpack_barrier();
    }
    restore_state();
}

// Zero K (and V) tile rows [rows, k_tiles) of the reserved step: a short last group's keys past its chunks (masked;
// zeros keep the masked PV products finite).
SPARSE_COLD void zero_k_rows(uint32_t rows) {
    reconfig_data_format(identity_scale, identity_scale);
    pack_reconfig_data_format(k);
    configure_single_tile_pack(k);
    sub_init(identity_scale, identity_scale);
    tile_regs_acquire();
    sub_tiles(identity_scale, identity_scale, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t row_tile = rows; row_tile < k_tiles; ++row_tile) {
        for (uint32_t d = 0; d < d_tiles; ++d) {
            pack_tile<true>(0, k, d * k_tiles + row_tile);
        }
        if constexpr (!v_in_k) {
            for (uint32_t c = 0; c < vd_tiles; ++c) {
                pack_tile<true>(0, v, row_tile * vd_tiles + c);
            }
        }
    }
    tile_regs_release();
}

// One step's gathered chunks (`rows` tile rows) into the K CB in the K^T layout (tile (d, kt) at d * k_tiles + kt), and
// V in the V CB when it is not read in place; tile rows past `rows` are zero.
SPARSE_COLD void produce_k_chunk(uint32_t rows) {
    CircularBuffer(k).reserve_back(k_tiles * d_tiles);
    if constexpr (!v_in_k) {
        CircularBuffer(v).reserve_back(k_tiles * vd_tiles);
    }
    if constexpr (scaled_kv) {
        // Packed rows: [vd FP8 latent bytes | vd / 128 FP32 scales | rope BF16 values]. The latent field dequantizes
        // into K's first vd tile rows of the K^T layout, the RoPE field tilizes into the rest.
        constexpr uint32_t latent_width = vd_tiles * tt::constants::TILE_WIDTH;
        // (The function is not a template: these hold for the scaled builds only.)
        static_assert(sparse_sdpa::SCALE_BLOCK_WIDTH % tt::constants::TILE_WIDTH == 0);
        constexpr uint32_t tiles_per_scale_block = sparse_sdpa::SCALE_BLOCK_WIDTH / tt::constants::TILE_WIDTH;
        constexpr uint32_t scale_blocks = sparse_sdpa::scale_block_count(latent_width);
        constexpr uint32_t scale_bytes = sparse_sdpa::scaled_kv_scale_bytes(latent_width);
        static_assert(!scaled_kv || sparse_sdpa::scaled_kv_rope_offset_is_aligned(latent_width));
        static_assert(!scaled_kv || packed_row_bytes % (tt::constants::TILE_WIDTH * sizeof(uint16_t)) == 0);
        constexpr uint32_t latent_physical_ct = packed_row_bytes / tt::constants::TILE_WIDTH;
        constexpr uint32_t rope_physical_ct = packed_row_bytes / (tt::constants::TILE_WIDTH * sizeof(uint16_t));
        constexpr uint32_t rope_offset_16b =
            (latent_width + scale_bytes) / sparse_sdpa::PACKED_FIELD_ADDRESS_UNIT_BYTES;
        static_assert(!scaled_kv || v_in_k, "scaled KV reads V in place");
        for (uint32_t row_tile = 0; row_tile < rows; ++row_tile) {
            CircularBuffer(cb::k_rm).wait_front(tt::constants::TILE_HEIGHT);
            CircularBuffer(cb::k_scale).wait_front(scale_blocks);
            if constexpr (fused_dequant) {
                sparse_sdpa::fused_dequant::latent_slab<
                    cb::k_rm,
                    cb::k_scale,
                    k,
                    latent_physical_ct,
                    scale_blocks,
                    tiles_per_scale_block,
                    k_tiles>(row_tile);
            } else {
                // Two passes: tilize the FP8 latent (BF16, exact), then multiply by the FP32 row scales.
                reconfig_data_format_srca(cb::k_rm);
                pack_reconfig_data_format(cb::k_latent);
                tilize_init(cb::k_rm, latent_physical_ct, cb::k_latent);
                CircularBuffer(cb::k_latent).reserve_back(vd_tiles);
                tilize_slab_strided<cb::k_rm, cb::k_latent>(vd_tiles, 0, 0, 1);
                CircularBuffer(cb::k_latent).push_back(vd_tiles);
                tilize_uninit(cb::k_rm, cb::k_latent);
                CircularBuffer(cb::k_latent).wait_front(vd_tiles);
                reconfig_data_format(cb::k_latent, cb::k_scale);
                pack_reconfig_data_format(k);
                mul_bcast_cols_init(cb::k_latent, cb::k_scale);
                for (uint32_t block = 0; block < scale_blocks; ++block) {
                    for (uint32_t tile = 0; tile < tiles_per_scale_block; ++tile) {
                        const uint32_t col = block * tiles_per_scale_block + tile;
                        tile_regs_acquire();
                        mul_tiles_bcast_cols(cb::k_latent, cb::k_scale, col, block, 0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile<true>(0, k, col * k_tiles + row_tile);
                        tile_regs_release();
                    }
                }
                CircularBuffer(cb::k_latent).pop_front(vd_tiles);
            }
            CircularBuffer(cb::k_scale).pop_front(scale_blocks);
            // cb::k_rope_rm is a format-only BF16 view of the packed slab: its UNPACK read pointer follows the owner.
            UNPACK(get_local_cb_interface(cb::k_rope_rm).fifo_rd_ptr = get_local_cb_interface(cb::k_rm).fifo_rd_ptr;);
            reconfig_data_format_srca(cb::k_rope_rm);
            pack_reconfig_data_format(k);
            tilize_init(cb::k_rope_rm, rope_physical_ct, k);
            uint32_t saved_rd_ptr = 0;
            UNPACK({
                auto& iface = get_local_cb_interface(cb::k_rope_rm);
                saved_rd_ptr = iface.fifo_rd_ptr;
                iface.fifo_rd_ptr = saved_rd_ptr + rope_offset_16b;
            });
            tilize_slab_strided<cb::k_rope_rm, k>(rope_tiles, 0, vd_tiles * k_tiles + row_tile, k_tiles);
            UNPACK(get_local_cb_interface(cb::k_rope_rm).fifo_rd_ptr = saved_rd_ptr;);
            tilize_uninit(cb::k_rope_rm, k);
            CircularBuffer(cb::k_rm).pop_front(tt::constants::TILE_HEIGHT);
        }
    } else {
        // BF16 or FP8 rows, d_tiles wide.
        reconfig_data_format_srca(cb::k_rm);
        pack_reconfig_data_format(k);
        tilize_init(cb::k_rm, d_tiles, k);
        for (uint32_t row_tile = 0; row_tile < rows; ++row_tile) {
            CircularBuffer(cb::k_rm).wait_front(tt::constants::TILE_HEIGHT);
            tilize_slab_strided<cb::k_rm, k>(d_tiles, 0, row_tile, k_tiles);
            if constexpr (!v_in_k) {
                // V: the row's leading vd tiles again, in V's (key-major) layout. V's CB has K's format (host), and a
                // pack reconfigure here would leave the packer's tilize mode (Blackhole tilize_init).
                tilize_slab_strided<cb::k_rm, v>(vd_tiles, 0, row_tile * vd_tiles, 1);
            }
            CircularBuffer(cb::k_rm).pop_front(tt::constants::TILE_HEIGHT);
        }
        tilize_uninit(cb::k_rm, k);
    }
    if constexpr (group > 1) {
        if (rows < k_tiles) {
            zero_k_rows(rows);
        }
    }
    CircularBuffer(k).push_back(k_tiles * d_tiles);
    if constexpr (!v_in_k) {
        CircularBuffer(v).push_back(k_tiles * vd_tiles);
    }
    restore_state();
}
}  // namespace

// Split-K merge (sparse_sdpa_common.hpp seg_kind::TAIL). SFPI code exists in the MATH and PACK builds only.
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// dest tile 0: the TAIL's maxima m_a, tile 1: the neighbour's m_b (column 0 is read). Tile 0 <- c_a = exp(scale * (m_a
// - m)), tile 1 <- c_b = exp(scale * (m_b - m)), m = max(m_a, m_b): both states rescaled to the common maximum.
template <uint32_t scale_bits>
inline void calculate_sparse_merge_corrections() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    constexpr float scale = __builtin_bit_cast(float, scale_bits);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[32];
        sfpi::vFloat m = a;
        v_if(b > a) { m = b; }
        v_endif;
        sfpi::dst_reg[0] = _sfpu_exp_fp32_accurate_((a - m) * scale);
        sfpi::dst_reg[32] = _sfpu_exp_fp32_accurate_((b - m) * scale);
        sfpi::dst_reg++;
    }
}
// dest tiles [0, w): the TAIL's state, [w, 2w): the neighbour's, tiles 2w and 2w + 1: column-broadcast c_a and c_b.
// state = state * c_a + neighbour * c_b.
template <int w>
inline void calculate_sparse_merge() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat ca = sfpi::dst_reg[32 * 2 * w];
        sfpi::vFloat cb = sfpi::dst_reg[32 * (2 * w + 1)];
#pragma GCC unroll 2
        for (int t = 0; t < w; ++t) {
            sfpi::vFloat x = sfpi::dst_reg[32 * t];
            sfpi::vFloat y = sfpi::dst_reg[32 * (w + t)];
            sfpi::dst_reg[32 * t] = x * ca + y * cb;
        }
        sfpi::dst_reg++;
    }
}
// In-place power-of-two scale of a whole tile (the fused dequant's 2^-OFFSET on the output).
template <uint32_t factor_bits>
inline void calculate_sparse_output_scale() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    constexpr float factor = __builtin_bit_cast(float, factor_bits);
#pragma GCC unroll 8
    for (int i = 0; i < 32; ++i) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * factor;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {
// Copy tile `index` of `cb` into dest tile `dest` (the recipe's state formats; FP32 state CBs unpack to dest).
ALWI void copy_state_tile(uint32_t cb, uint32_t index, uint32_t dest) {
    sdpa_stream_reconfig_srca(cb);
    copy_init(cb);
    copy_tile(cb, index, dest);
}
// Column 0 of tile `index` of `cb` broadcast across dest tile `dest`.
ALWI void bcast_state_tile(uint32_t cb, uint32_t index, uint32_t dest) {
    sdpa_stream_reconfig_srca(cb);
    unary_bcast_init<BroadcastType::COL>(cb);
    unary_bcast<BroadcastType::COL>(cb, index, dest);
    unary_bcast_uninit<BroadcastType::COL>(cb);
}

// The TAIL's state after its last chunk (not normalized; recipe_checkpoint's layout: the BF16 recipes keep O in CB 9's
// rows (plane 0 of 2) and l in CB 13 at their bases, popped; the FP32 recipes in the pushed prev banks), merged with
// the neighbour's HEAD state (cb::recv_*: O rows of vd tiles, maxima, l) or alone when the HEAD had no keys, then
// normalized into the output CB. The merged state goes where the normalization reads it: the BF16 recipes' CB 9 / 13 in
// place, the FP32 recipes' cur banks.
SPARSE_COLD void merge_and_normalize(RecipeAccumulatorState& state, bool merge) {
    constexpr uint32_t o_stride = sdpa_out_stride;  // numerator CB tiles per Q row, over vd
    const uint32_t a_out = recipe::fp32 ? state.prev.out : state.cur.out;
    const uint32_t a_sum = recipe::fp32 ? state.prev.sum : state.cur.sum;
    const uint32_t a_max = state.prev.max;
    // Where the normalization reads.
    uint32_t n_out = a_out, n_sum = a_sum;
    if constexpr (recipe::fp32) {
        PACK(sdpa_pack.format_cb = INVALID_CB; sdpa_pack.width = 0;)
    }
    if (merge) {
        n_out = state.cur.out;
        n_sum = state.cur.sum;
        CircularBuffer(n_out).reserve_back(q_tiles * vd_tiles * o_stride);
        CircularBuffer(n_sum).reserve_back(q_tiles);
        CircularBuffer(cb::recv_out).wait_front(q_tiles * vd_tiles);
        CircularBuffer(cb::recv_max).wait_front(q_tiles);
        CircularBuffer(cb::recv_sum).wait_front(q_tiles);
        // Corrections per Q row: c_a into CB exp_max_diff, c_b into cb::merge_corr.
        CircularBuffer(exp_max_diff).reserve_back(q_tiles);
        CircularBuffer(cb::merge_corr).reserve_back(q_tiles);
        PACK((llk_pack_reconfig_l1_acc(0)));
        for (uint32_t r = 0; r < q_tiles; ++r) {
            tile_regs_acquire();
            copy_state_tile(a_max, r, 0);
            copy_state_tile(cb::recv_max, r, 1);
            tile_regs_commit();
            tile_regs_wait();
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sparse_merge_corrections, (scale_fp32), 0, VectorMode::None)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_reconfig_data_format(exp_max_diff);
            configure_single_tile_pack(exp_max_diff);
            pack_tile<true>(0, exp_max_diff, r);
            pack_reconfig_data_format(cb::merge_corr);
            configure_single_tile_pack(cb::merge_corr);
            pack_tile<true>(1, cb::merge_corr, r);
            tile_regs_release();
        }
        CircularBuffer(exp_max_diff).push_back(q_tiles);
        CircularBuffer(cb::merge_corr).push_back(q_tiles);
        CircularBuffer(exp_max_diff).wait_front(q_tiles);
        CircularBuffer(cb::merge_corr).wait_front(q_tiles);
        // l = l_a * c_a + l_b * c_b, O = O_a * c_a + O_b * c_b: ow O tiles per dest pass (2 + 2 + 2 of the 16-bit
        // dest's 8 tiles; 1 + 1 + 2 of the FP32 dest's 4).
        constexpr uint32_t ow = !recipe::fp32 && vd_tiles % 2 == 0 ? 2 : 1;
        for (uint32_t r = 0; r < q_tiles; ++r) {
            tile_regs_acquire();
            copy_state_tile(a_sum, r, 0);
            copy_state_tile(cb::recv_sum, r, 1);
            bcast_state_tile(exp_max_diff, r, 2);
            bcast_state_tile(cb::merge_corr, r, 3);
            tile_regs_commit();
            tile_regs_wait();
            PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sparse_merge, (1), 0, VectorMode::None)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_reconfig_data_format(n_sum);
            configure_single_tile_pack(n_sum);
            pack_tile<true>(0, n_sum, r);
            tile_regs_release();
            for (uint32_t j = 0; j < vd_tiles; j += ow) {
                tile_regs_acquire();
                for (uint32_t t = 0; t < ow; ++t) {
                    copy_state_tile(a_out, r * vd_tiles * o_stride + j + t, t);
                    copy_state_tile(cb::recv_out, r * vd_tiles + j + t, ow + t);
                }
                bcast_state_tile(exp_max_diff, r, 2 * ow);
                bcast_state_tile(cb::merge_corr, r, 2 * ow + 1);
                tile_regs_commit();
                tile_regs_wait();
                PACK((
                    SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sparse_merge, (ow), 0, VectorMode::None)));
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                pack_reconfig_data_format(n_out);
                configure_single_tile_pack(n_out);
                for (uint32_t t = 0; t < ow; ++t) {
                    pack_tile<true>(t, n_out, r * vd_tiles * o_stride + j + t);
                }
                tile_regs_release();
            }
        }
        CircularBuffer(exp_max_diff).pop_front(q_tiles);
        CircularBuffer(cb::merge_corr).pop_front(q_tiles);
        CircularBuffer(cb::recv_out).pop_front(q_tiles * vd_tiles);
        CircularBuffer(cb::recv_max).pop_front(q_tiles);
        CircularBuffer(cb::recv_sum).pop_front(q_tiles);
        if constexpr (recipe::fp32) {
            sdpa_cb_pop_front_out_of_line(a_out, q_tiles * vd_tiles);
            sdpa_cb_pop_front_out_of_line(a_sum, q_tiles);
        }
    } else if constexpr (!recipe::fp32) {
        CircularBuffer(n_out).reserve_back(q_tiles * vd_tiles * o_stride);
        CircularBuffer(n_sum).reserve_back(q_tiles);
    }
    if constexpr (!recipe::fp32) {
        // CB 9 / 13 hold the (merged) state at their bases: publish it to the normalization.
        CircularBuffer(n_out).push_back(q_tiles * vd_tiles * o_stride);
        CircularBuffer(n_sum).push_back(q_tiles);
    } else if (merge) {
        CircularBuffer(n_out).push_back(q_tiles * vd_tiles);
        CircularBuffer(n_sum).push_back(q_tiles);
    }
    sdpa_cb_pop_front_out_of_line(a_max, q_tiles);
    if constexpr (recipe::fp32) {
        PACK(sdpa_pack.format_cb = INVALID_CB; sdpa_pack.width = 0;)
    }
    pack_reconfig_data_format(recip_scratch);
    // The step's own copy of the normalization (the cold one with fused chunks).
    if constexpr (recipe::fused_active) {
        recipe_cold_step::normalize_row_streaming<
            vd_tiles,
            compute_kernel_lib::DEST_AUTO_LIMIT,
            col_identity,
            recip_scratch,
            out,
            scale_fp32>(n_sum, n_out, q_tiles);
    } else {
        recipe_hot_step::normalize_row_streaming<
            vd_tiles,
            compute_kernel_lib::DEST_AUTO_LIMIT,
            col_identity,
            recip_scratch,
            out,
            scale_fp32>(n_sum, n_out, q_tiles);
    }
    restore_state();
}

// The pass's normalized rows (CB out, q_tiles rows) to the writer: fused dequant scales them by 2^-OFFSET in place
// (V carried 2^OFFSET), then the real rows untilize into cb::out_rm and the padding rows are dropped.
SPARSE_COLD void emit_output(uint32_t rows) {
    CircularBuffer(out).wait_front(q_tiles * vd_tiles);
    if constexpr (fused_dequant) {
        constexpr uint32_t inverse_bits = static_cast<uint32_t>(127 - sparse_sdpa::FUSED_DEQUANT_EXP_OFFSET) << 23;
        constexpr uint32_t batch = compute_kernel_lib::DEST_AUTO_LIMIT;
        reconfig_data_format_srca(out);
        pack_reconfig_data_format(out);
        configure_single_tile_pack(out);
        copy_init(out);
        for (uint32_t t0 = 0; t0 < rows * vd_tiles; t0 += batch) {
            const uint32_t n = rows * vd_tiles - t0 < batch ? rows * vd_tiles - t0 : batch;
            tile_regs_acquire();
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(out, t0 + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                PACK((SFPU_UNARY_CALL(
                    DST_SYNC_MODE,
                    DST_ACCUM_MODE,
                    calculate_sparse_output_scale,
                    (inverse_bits),
                    i,
                    VectorMode::None)));
            }
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile<true>(i, out, t0 + i);
            }
            tile_regs_release();
        }
        recipe_pack_to_unpack_barrier();
    }
    sparse_recipe::emit_rows<q_tiles, vd_tiles, cb::out_rm>(rows);
}
}  // namespace

void kernel_main() {
    const uint32_t tok_count = get_arg_val<uint32_t>(sparse_sdpa::split_rt_arg::TOK_COUNT);
    const uint32_t head_c0 = get_arg_val<uint32_t>(sparse_sdpa::split_rt_arg::HEAD_C0);
    const uint32_t tail_c1 = get_arg_val<uint32_t>(sparse_sdpa::split_rt_arg::TAIL_C1);
    const uint32_t n_segs = (head_c0 != 0 ? 1u : 0u) + tok_count + (tail_c1 != 0 ? 1u : 0u);

    compute_kernel_hw_startup<SrcOrder::Reverse>(q, k, out);
    if constexpr (recipe::lofi) {
        recipe_mm_init(q, k, true, 1, 1, 1);
    } else {
        matmul_init(q, k);
    }
    // The reduce scaler, the column identity and the per-row sinks stay fronted for the whole kernel.
    CircularBuffer(identity_scale).wait_front(1);
    CircularBuffer(col_identity).wait_front(1);
    if constexpr (recipe::sink) {
        CircularBuffer(recipe::sink_cb).wait_front(passes * q_tiles);
    }
    init_sdpa_streaming_semaphores();
    restore_state();

    RecipeAccumulatorState state;
    for (uint32_t seg = 0; seg < n_segs; ++seg) {
        // The reader's control page: active chunks of this segment (0 for an empty split-K HEAD), the token's valid
        // keys, the segment's first (token-global) chunk and its kind.
        CircularBuffer(cb::ctrl).wait_front(1);
        const uint32_t active_chunks =
            ckernel::read_tile_value(cb::ctrl, 0, sparse_sdpa::control_message::ACTIVE_CHUNKS);
        const uint32_t valid_keys = ckernel::read_tile_value(cb::ctrl, 0, sparse_sdpa::control_message::VALID_KEYS);
        const uint32_t chunk_base = ckernel::read_tile_value(cb::ctrl, 0, sparse_sdpa::control_message::CHUNK_BASE);
        const uint32_t seg_kind = ckernel::read_tile_value(cb::ctrl, 0, sparse_sdpa::control_message::SEG_KIND);
        CircularBuffer(cb::ctrl).pop_front(1);
        if (active_chunks == 0) {
            continue;  // empty split-K HEAD: nothing to compute or send
        }
        const bool split_head = split_k && seg_kind == sparse_sdpa::seg_kind::HEAD;
        const bool split_tail = split_k && seg_kind == sparse_sdpa::seg_kind::TAIL;
        // The token's sentinel tail lies in its last chunk: keys [valid_last, k_chunk_keys) there are masked.
        const uint32_t token_chunks = (valid_keys + k_chunk_keys - 1) / k_chunk_keys;
        const uint32_t valid_last = valid_keys - (token_chunks - 1) * k_chunk_keys;
        for (uint32_t pass = 0; pass < passes; ++pass) {
            const uint32_t pass_rows = head_tiles - pass * q_tiles < q_tiles ? head_tiles - pass * q_tiles : q_tiles;
            tilize_q();
            recipe_ring_reset_state(state);
            if constexpr (recipe::sink) {
                recipe_sink_row_base = pass * q_tiles;
                recipe_sink_row = 0;
            }
            for (uint32_t chunk = 0; chunk < active_chunks; chunk += group) {
                // This step's gathered chunks; the token's last one ends at its valid keys.
                const uint32_t n = active_chunks - chunk < group ? active_chunks - chunk : group;
                produce_k_chunk(n * gather_tiles);
                const bool token_end = chunk_base + chunk + n == token_chunks;
                const uint32_t valid = (n - 1) * k_chunk_keys + (token_end ? valid_last : k_chunk_keys);
                recipe_ring_mask.valid_rows = valid;
                recipe_ring_mask.masked = valid < k_tiles * tt::constants::TILE_WIDTH;
                const bool last = chunk + n == active_chunks;
                sparse_recipe::run_chunk<q_tiles, k_tiles, d_tiles, vd_tiles, scale_fp32, cb_v>(
                    state, last && !split_head && !split_tail, last);
            }
            if (split_head) {
                // The writer sends the unnormalized state to core i-1 (recipe_checkpoint: compute waits for its ack).
                recipe_checkpoint<q_tiles, checkpoint_request, checkpoint_ack, vd_tiles>(state, 0, false);
                restore_state();
                continue;
            }
            if (split_tail) {
                CircularBuffer(cb::merge_ctrl).wait_front(1);
                const uint32_t partial = ckernel::read_tile_value(cb::merge_ctrl, 0, 0);
                CircularBuffer(cb::merge_ctrl).pop_front(1);
                merge_and_normalize(state, partial == sparse_sdpa::SPLIT_PARTIAL_VALID);
            }
            emit_output(pass_rows);
        }
    }
}
