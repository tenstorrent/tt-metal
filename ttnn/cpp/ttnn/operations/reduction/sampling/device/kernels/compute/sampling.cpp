// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/topk.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/reduce.h"
#include "api/compute/transpose.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "ckernel_sfpu.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#define DEBUG_PRINT 0
using namespace ckernel;

// 1: Blackhole runs the top-k stage on fused [bf16 | u16 index] keys in a 32-bit DEST section.
#ifndef SAMPLING_TOPK_FUSED_32B_DEST
#ifdef ARCH_BLACKHOLE
#define SAMPLING_TOPK_FUSED_32B_DEST 1
#else
#define SAMPLING_TOPK_FUSED_32B_DEST 0
#endif
#endif

#if SAMPLING_TOPK_FUSED_32B_DEST && defined(TRISC_MATH)
namespace sampling_fused {
using namespace ckernel;
using namespace ckernel::sfpu;

// _topk_fuse_tile_ for value tiles moved in as raw u16 words: DEST 0,1 hold [garbage | bf16 bits].
// canonicalize_negzero: -0 becomes +0 first, as the comparator path does before its local sort.
template <bool largest, bool canonicalize_negzero>
inline void fuse_raw16_slab() {
    constexpr int body = canonicalize_negzero ? 15 : 10;
    TOPK_SFPENCC_ALL_LANES_ON();
    sfpi::vConstIntPrgm0 = TOPK_LO16_MASK;
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    load_replay_buf<Exec>(0, body, [] {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 128);
        TTI_SFPSHFT(16, 0, p_sfpu::LREG0, 1);
        if constexpr (canonicalize_negzero) {
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPSHFT(1, 0, p_sfpu::LREG2, 1);
            TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
            TOPK_SFPENCC_ALL_LANES_ON();
        }
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, largest ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TOPK_SFPENCC_ALL_LANES_ON();
        TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
    });
    for (int i = 1; i < 64; i++) {
        lltt::replay(0, body);
    }
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    topk_replay_init = 0;
}

// Splits keys into u16 value and index words in the packer-visible high half (mode 9, as the u16 index pack).
// flush_denormals mirrors the comparator path's BF16 SFPSTORE, which flushes denormals to signed zero.
template <bool largest, bool flush_denormals>
inline void defuse_raw16(const int num_tiles) {
    constexpr std::uint32_t pack_u16 = TOPK_SFPSTORE_MODE_PACK_UINT16;
    constexpr int body = flush_denormals ? 15 : 10;
    TOPK_SFPENCC_ALL_LANES_ON();
    sfpi::vConstIntPrgm0 = TOPK_LO16_MASK;
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    load_replay_buf<Exec>(0, body, [] {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, largest ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TOPK_SFPENCC_ALL_LANES_ON();
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        if constexpr (flush_denormals) {
            TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_NODEBIAS);
            TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPSHFT((-31) & 0xFFF, 0, p_sfpu::LREG0, 1);
            TTI_SFPSHFT(31, 0, p_sfpu::LREG0, 1);
            TOPK_SFPENCC_ALL_LANES_ON();
        }
        TTI_SFPSHFT((-16) & 0xFFF, 0, p_sfpu::LREG0, 1);
        TTI_SFPSTORE(p_sfpu::LREG0, pack_u16, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, pack_u16, ADDR_MOD_7, 128);
        TTI_INCRWC(0, 2, 0, 0);
    });
    const int n = 32 * num_tiles;
    for (int i = 1; i < n; i++) {
        lltt::replay(0, body);
    }
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    topk_replay_init = 0;
}
}  // namespace sampling_fused
#endif

static void generate_rand_tile(const uint32_t dfb_id, const uint32_t seed) {
    compute_kernel_hw_startup(dfb_id, dfb_id);
    copy_init(dfb_id);

    DataflowBuffer dfb_obj(static_cast<uint16_t>(dfb_id));

    // The random tile is packed to BF16 before the strict cumulative-probability
    // comparison. Keep the FP32 endpoint below the BF16 midpoint to 1.0 so the
    // packed threshold remains strictly less than 1.0.
    constexpr uint32_t rand_scale = 0x3F7F7FFFU;
    constexpr uint32_t rand_from = 0;

    if (seed != 0) {
        rand_tile_init(seed);
    }
    dfb_obj.reserve_back(1);

    tile_regs_acquire();
    rand_tile(0, rand_from, rand_scale);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb_id, 0);
    tile_regs_release();

    dfb_obj.push_back(1);
}

template <uint32_t in0_dfb, uint32_t in1_dfb, uint32_t rows, uint32_t cols>
void sub_exp_block_bcast_cols_inplace() {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced

    DataflowBuffer in0_dfb_obj(in0_dfb);
    DataflowBuffer in1_dfb_obj(in1_dfb);

    sub_bcast_cols_init(in0_dfb, in1_dfb);
    exp_tile_init<true>();
    in0_dfb_obj.wait_front(rows * cols);
    in1_dfb_obj.wait_front(rows);

    constexpr uint32_t dst_tiles = 1;       // SUB_EXP_GRANULARITY;
    constexpr uint32_t granularity = cols;  // #>> LOG2_SUB_EXP_GRANULARITY;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_dfb, in1_dfb, j, i, j);
                exp_tile<true>(j);
            }
            tile_regs_commit();

            in0_dfb_obj.pop_front(dst_tiles);
            in0_dfb_obj.reserve_back(dst_tiles);

            tile_regs_wait();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, in0_dfb);
            }
            tile_regs_release();

            in0_dfb_obj.push_back(dst_tiles);
        }
    }
}

static void add_block_inplace(uint32_t in0_dfb, uint32_t in1_dfb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced
    DataflowBuffer in0_dfb_obj(static_cast<uint16_t>(in0_dfb));
    DataflowBuffer in1_dfb_obj(static_cast<uint16_t>(in1_dfb));

    reconfig_data_format(in0_dfb, in1_dfb);
    add_init(in0_dfb, in1_dfb);
    in0_dfb_obj.wait_front(static_cast<uint16_t>(num_tiles));
    in1_dfb_obj.wait_front(static_cast<uint16_t>(num_tiles));
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        add_tiles(in0_dfb, in1_dfb, 0, i, 0);
        tile_regs_commit();

        in0_dfb_obj.pop_front(1);
        in0_dfb_obj.reserve_back(1);

        tile_regs_wait();
        pack_reconfig_data_format(in0_dfb);
        pack_tile(0, in0_dfb);
        tile_regs_release();

        in0_dfb_obj.push_back(1);
    }
}

static void mul_block_bcast_cols(uint32_t in0_dfb, uint32_t in1_dfb, uint32_t out_dfb, uint32_t rows, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows consumed

    DataflowBuffer in0_dfb_obj(static_cast<uint16_t>(in0_dfb));
    DataflowBuffer in1_dfb_obj(static_cast<uint16_t>(in1_dfb));
    DataflowBuffer out_dfb_obj(static_cast<uint16_t>(out_dfb));

    const uint32_t num_tiles = rows * cols;
    mul_bcast_cols_init(in0_dfb, in1_dfb);
    in0_dfb_obj.wait_front(static_cast<uint16_t>(num_tiles));
    in1_dfb_obj.wait_front(static_cast<uint16_t>(rows));
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(in0_dfb, in1_dfb, 0, i, 0);
            tile_regs_commit();

            in0_dfb_obj.pop_front(1);
            out_dfb_obj.reserve_back(1);

            tile_regs_wait();
            pack_tile(0, out_dfb);
            tile_regs_release();

            out_dfb_obj.push_back(1);
        }
    }
    in1_dfb_obj.pop_front(static_cast<uint16_t>(rows));
}

static void recip_block_inplace(uint32_t in_dfb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles produced
    DataflowBuffer in_dfb_obj(static_cast<uint16_t>(in_dfb));

    copy_init(in_dfb);
    recip_tile_init();

    in_dfb_obj.wait_front(static_cast<uint16_t>(num_tiles));
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(in_dfb, 0, 0);
        recip_tile(0);
        tile_regs_commit();

        in_dfb_obj.pop_front(1);
        in_dfb_obj.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, in_dfb);
        tile_regs_release();

        in_dfb_obj.push_back(1);
    }
}

template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_dfb,
    uint32_t scale_dfb,
    uint32_t out_dfb,
    uint32_t rows,
    uint32_t cols>
void reduce_c() {
    // Postcondition: in0_cb has rows*cols produced (WaitUpfrontNoPop — tiles not consumed)
    // Postcondition: out_cb has rows produced
    compute_kernel_lib::reduce<
        pool_type,
        reduce_dim,
        in0_dfb,
        scale_dfb,
        out_dfb,
        compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        compute_kernel_lib::ReduceInputBlockShape::of(rows, cols));
    UNPACK(tensix_sync());  // Workaround for issue #9370
}

template <
    uint32_t Ht,
    uint32_t Wt,
    uint32_t K,
    uint32_t logWt,
    uint32_t logk,
    uint32_t input_dfb_index,
    uint32_t index_dfb_index,
    uint32_t input_transposed_dfb_index,
    uint32_t index_transposed_dfb_index,
    uint32_t values_dfb_index,
    uint32_t output_ind_dfb_index,
    uint32_t tile_width,
    bool first_call,
    bool stable_sort>
void top_k() {
    // dest indices for where to unpack the tiles for the llk
    // the input goes in index 0,1 and the index goes in index 2,3
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;
    ckernel::topk_tile_init();

    DataflowBuffer input_dfb(input_dfb_index);
    DataflowBuffer index_dfb(index_dfb_index);
    DataflowBuffer input_transposed_dfb(input_transposed_dfb_index);
    DataflowBuffer index_transposed_dfb(index_transposed_dfb_index);
    DataflowBuffer values_dfb(values_dfb_index);
    DataflowBuffer output_ind_dfb(output_ind_dfb_index);

    if (first_call) {
        transpose_init(input_dfb_index);
    }
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        const bool ascending = false;
        input_transposed_dfb.reserve_back(Wt);
        index_transposed_dfb.reserve_back(Wt);

        // streaming in input and index tiles to transpose and bitonic local sort them, two tiles at a time
        for (uint32_t wt = 0; wt < Wt; wt += 2) {
            // local sort into k groups
            input_dfb.wait_front(2);
            index_dfb.wait_front(2);

            tile_regs_acquire();
            reconfig_data_format_srca(input_dfb_index);
            transpose_init(input_dfb_index);
            transpose_tile(input_dfb_index, 0, 0);
            transpose_tile(input_dfb_index, 1, 1);

            reconfig_data_format_srca(index_dfb_index);
            transpose_init(index_dfb_index);
            transpose_tile(index_dfb_index, 0, 2);
            transpose_tile(index_dfb_index, 1, 3);

            // llk_topk_sort -> inplace
            // stable_sort: equal values keep their original (lowest) position, so the candidate the
            // top-k keeps for a tie does not depend on how the bitonic network happens to swap.
            if constexpr (stable_sort) {
                ckernel::topk_canonicalize_negzero_values(0);
            }
            ckernel::topk_local_sort<
                stable_sort,
                DST_ACCUM_MODE,
                /*fused=*/false,
                /*rank_stamped=*/false,
                ckernel::TopkTieOrder::Descending>(0, (int)ascending, logk - 1);

            tile_regs_commit();

            input_dfb.pop_front(2);
            index_dfb.pop_front(2);

            tile_regs_wait();
            // pack value tiles into cb_intermed0
            pack_reconfig_data_format(input_transposed_dfb_index);
            pack_tile(0, input_transposed_dfb_index);
            pack_tile(1, input_transposed_dfb_index);

            // pack index tiles into cb_intermed1
            pack_reconfig_data_format(index_transposed_dfb_index);
            pack_tile(2, index_transposed_dfb_index);
            pack_tile(3, index_transposed_dfb_index);
            tile_regs_release();
        }

        input_transposed_dfb.push_back(Wt);
        index_transposed_dfb.push_back(Wt);

        // iterative divide and conquer on pairs of tiles (bitonic topk merge and rebuild)
        // first iteration we compare 0th and 1st tile, then 2nd and 3rd, etc. We get the sorted top 32 values in each
        // pair. second iteration we compare 0th and 2nd tile, then 4th and 6th, etc. logWt iteration we compare 0th and
        // Wt/2 tile single buffer as we can pack tiles back in-place
        for (uint32_t m_iter = 0; m_iter < logWt; ++m_iter) {
            bool a = false;
            input_transposed_dfb.wait_front(Wt);
            index_transposed_dfb.wait_front(Wt);

            for (uint32_t left_ind = 0; left_ind < Wt - (1u << m_iter); left_ind += 2u << m_iter) {
                const uint32_t right_ind = left_ind + (1u << m_iter);
                tile_regs_acquire();

                reconfig_data_format_srca(index_transposed_dfb_index, input_transposed_dfb_index);
                copy_init(input_transposed_dfb_index);
                copy_tile(input_transposed_dfb_index, left_ind, input_dest_start);
                copy_tile(input_transposed_dfb_index, right_ind, input_dest_end);

                // unpack indices into dest
                reconfig_data_format_srca(input_transposed_dfb_index, index_transposed_dfb_index);
                copy_init(index_transposed_dfb_index);
                copy_tile(index_transposed_dfb_index, left_ind, index_dest_start);
                copy_tile(index_transposed_dfb_index, right_ind, index_dest_end);

                // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
                ckernel::topk_merge<
                    /*idir=*/false,
                    stable_sort,
                    DST_ACCUM_MODE,
                    /*fused=*/false,
                    /*rank_stamped=*/false,
                    ckernel::TopkTieOrder::Descending>(0, m_iter, K);
                // sort within the larger 32 values
                ckernel::topk_rebuild<
                    stable_sort,
                    DST_ACCUM_MODE,
                    /*fused=*/false,
                    /*rank_stamped=*/false,
                    ckernel::TopkTieOrder::Descending>(0, (uint32_t)a, m_iter, K, logk, true);

                tile_regs_commit();
                tile_regs_wait();
                // pack value tiles in-place in the single-buffered cb_intermed0, we only need the upper 32 values for
                // topk, which was in input_dest_start
                pack_reconfig_data_format(input_transposed_dfb_index);
                pack_tile<true>(input_dest_start, input_transposed_dfb_index, left_ind);

                // pack index tiles in-place in the single-buffered cb_intermed1, we only need the upper 32 values for
                // topk, which was in index_dest_start
                pack_reconfig_data_format(index_transposed_dfb_index);
                pack_tile<true>(index_dest_start, index_transposed_dfb_index, left_ind);
                tile_regs_release();
                a = !a;
            }

            input_transposed_dfb.reserve_back(Wt);
            index_transposed_dfb.reserve_back(Wt);

            input_transposed_dfb.pop_front(Wt);
            index_transposed_dfb.pop_front(Wt);

            input_transposed_dfb.push_back(Wt);
            index_transposed_dfb.push_back(Wt);
        }

        constexpr uint32_t Kt = K % tile_width == 0 ? K / tile_width : (K / tile_width) + 1;

        // transpose value tiles and pack into output buffer
        reconfig_data_format_srca(input_transposed_dfb_index);
        transpose_init(input_transposed_dfb_index);
        pack_reconfig_data_format(input_transposed_dfb_index);
        input_transposed_dfb.wait_front(Wt);
        for (uint32_t i = 0; i < Kt; ++i) {
            tile_regs_acquire();
            transpose_tile(input_transposed_dfb_index, i, 0);
            tile_regs_commit();

            values_dfb.reserve_back(1);

            tile_regs_wait();
            pack_tile(0, values_dfb_index);
            tile_regs_release();

            values_dfb.push_back(1);
        }
        input_transposed_dfb.pop_front(Wt);

        // transpose index tiles and pack into output buffer
        reconfig_data_format_srca(index_transposed_dfb_index);
        transpose_init(index_transposed_dfb_index);
        pack_reconfig_data_format(index_transposed_dfb_index);
        index_transposed_dfb.wait_front(Wt);
        for (uint32_t i = 0; i < Kt; ++i) {
            tile_regs_acquire();
            transpose_tile(index_transposed_dfb_index, i, 0);
            tile_regs_commit();

            output_ind_dfb.reserve_back(1);

            tile_regs_wait();
            pack_tile(0, output_ind_dfb_index);
            tile_regs_release();

            output_ind_dfb.push_back(1);
        }
        index_transposed_dfb.pop_front(Wt);
    }
    sfpu::_init_sfpu_config_reg();
}

#if SAMPLING_TOPK_FUSED_32B_DEST
// top_k<stable_sort = true> order from fused keys in 32-bit DEST; bf16 values travel as raw u16 words,
// so unpacker, datacopy MOP and packer all take the UInt16 index CB format.
template <
    uint32_t Ht,
    uint32_t Wt,
    uint32_t K,
    uint32_t logWt,
    uint32_t logk,
    uint32_t input_dfb_index,
    uint32_t index_dfb_index,
    uint32_t input_transposed_dfb_index,
    uint32_t index_transposed_dfb_index,
    uint32_t values_dfb_index,
    uint32_t output_ind_dfb_index,
    uint32_t tile_width>
void top_k_fused_32b_dest() {
    constexpr bool largest = true;

    DataflowBuffer input_dfb(input_dfb_index);
    DataflowBuffer index_dfb(index_dfb_index);
    DataflowBuffer input_transposed_dfb(input_transposed_dfb_index);
    DataflowBuffer index_transposed_dfb(index_transposed_dfb_index);
    DataflowBuffer values_dfb(values_dfb_index);
    DataflowBuffer output_ind_dfb(output_ind_dfb_index);

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        set_fp32_dest_acc<true>();
        ckernel::topk_tile_init</*fused=*/true>();
        reconfig_data_format_srca(index_dfb_index);
        PACK((llk_pack_reconfig_data_format<true>(index_transposed_dfb_index)));

        input_transposed_dfb.reserve_back(Wt);
        index_transposed_dfb.reserve_back(Wt);

        for (uint32_t wt = 0; wt < Wt; wt += 2) {
            input_dfb.wait_front(2);
            index_dfb.wait_front(2);

            tile_regs_acquire();
            transpose_init<true>(index_dfb_index);
            transpose_tile<true>(input_dfb_index, 0, 0);
            transpose_tile<true>(input_dfb_index, 1, 1);
            transpose_tile<true>(index_dfb_index, 0, 2);
            transpose_tile<true>(index_dfb_index, 1, 3);
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                sampling_fused::fuse_raw16_slab<largest, true>, 0, VectorMode::RC_custom)));
            ckernel::topk_local_sort</*stable_sort=*/false, /*is_fp32_dest_acc_en=*/true, /*fused=*/true>(
                0, /*idir=*/0, logk - 1);
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                sampling_fused::defuse_raw16<largest, true>, 0, VectorMode::RC_custom, 2)));
            tile_regs_commit<true>();

            input_dfb.pop_front(2);
            index_dfb.pop_front(2);

            tile_regs_wait();
            pack_tile<false, true>(0, input_transposed_dfb_index);
            pack_tile<false, true>(1, input_transposed_dfb_index);
            pack_tile<false, true>(2, index_transposed_dfb_index);
            pack_tile<false, true>(3, index_transposed_dfb_index);
            tile_regs_release<true>();
        }

        input_transposed_dfb.push_back(Wt);
        index_transposed_dfb.push_back(Wt);

        for (uint32_t m_iter = 0; m_iter < logWt; ++m_iter) {
            bool a = false;
            input_transposed_dfb.wait_front(Wt);
            index_transposed_dfb.wait_front(Wt);

            for (uint32_t left_ind = 0; left_ind < Wt - (1u << m_iter); left_ind += 2u << m_iter) {
                const uint32_t right_ind = left_ind + (1u << m_iter);
                tile_regs_acquire();
                copy_init<true>(index_transposed_dfb_index);
                copy_tile<true>(input_transposed_dfb_index, left_ind, 0);
                copy_tile<true>(input_transposed_dfb_index, right_ind, 1);
                copy_tile<true>(index_transposed_dfb_index, left_ind, 2);
                copy_tile<true>(index_transposed_dfb_index, right_ind, 3);
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sampling_fused::fuse_raw16_slab<largest, false>, 0, VectorMode::RC_custom)));
                ckernel::topk_merge</*idir=*/false, /*stable_sort=*/false, /*is_fp32_dest_acc_en=*/true, /*fused=*/true>(
                    0, m_iter, K);
                ckernel::topk_rebuild</*stable_sort=*/false, /*is_fp32_dest_acc_en=*/true, /*fused=*/true>(
                    0, (uint32_t)a, m_iter, K, logk, true);
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sampling_fused::defuse_raw16<largest, false>, 0, VectorMode::RC_custom, 1)));
                tile_regs_commit<true>();

                tile_regs_wait();
                pack_tile<true, true>(0, input_transposed_dfb_index, left_ind);
                pack_tile<true, true>(2, index_transposed_dfb_index, left_ind);
                tile_regs_release<true>();
                a = !a;
            }

            input_transposed_dfb.reserve_back(Wt);
            index_transposed_dfb.reserve_back(Wt);

            input_transposed_dfb.pop_front(Wt);
            index_transposed_dfb.pop_front(Wt);

            input_transposed_dfb.push_back(Wt);
            index_transposed_dfb.push_back(Wt);
        }

        restore_fp32_dest_acc<true>();

        constexpr uint32_t Kt = K % tile_width == 0 ? K / tile_width : (K / tile_width) + 1;

        // From here on identical to top_k: 16-bit DEST transposes back to row layout.
        reconfig_data_format_srca(input_transposed_dfb_index);
        transpose_init(input_transposed_dfb_index);
        pack_reconfig_data_format(input_transposed_dfb_index);
        input_transposed_dfb.wait_front(Wt);
        for (uint32_t i = 0; i < Kt; ++i) {
            tile_regs_acquire();
            transpose_tile(input_transposed_dfb_index, i, 0);
            tile_regs_commit();

            values_dfb.reserve_back(1);

            tile_regs_wait();
            pack_tile(0, values_dfb_index);
            tile_regs_release();

            values_dfb.push_back(1);
        }
        input_transposed_dfb.pop_front(Wt);

        reconfig_data_format_srca(index_transposed_dfb_index);
        transpose_init(index_transposed_dfb_index);
        pack_reconfig_data_format(index_transposed_dfb_index);
        index_transposed_dfb.wait_front(Wt);
        for (uint32_t i = 0; i < Kt; ++i) {
            tile_regs_acquire();
            transpose_tile(index_transposed_dfb_index, i, 0);
            tile_regs_commit();

            output_ind_dfb.reserve_back(1);

            tile_regs_wait();
            pack_tile(0, output_ind_dfb_index);
            tile_regs_release();

            output_ind_dfb.push_back(1);
        }
        index_transposed_dfb.pop_front(Wt);
    }
    sfpu::_init_sfpu_config_reg();
}
#endif

template <uint32_t in0_dfb, uint32_t in1_scalar_dfb, uint32_t num_tiles>
void mul_block_bcast_scalar_inplace() {
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_scalar_cb has 1 produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_scalar_cb has 1 produced

    DataflowBuffer in0_dfb_obj(in0_dfb);
    DataflowBuffer in1_scalar_dfb_obj(in1_scalar_dfb);

    const uint32_t dst_tiles = num_tiles;
    const uint32_t granularity = 1;

    reconfig_data_format(in0_dfb, in1_scalar_dfb);
    mul_bcast_scalar_init(in0_dfb, in1_scalar_dfb);
    in0_dfb_obj.wait_front(num_tiles);
    in1_scalar_dfb_obj.wait_front(1);

    for (uint32_t g = 0; g < granularity; ++g) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            mul_tiles_bcast_scalar(in0_dfb, in1_scalar_dfb, i, 0, i);
        }
        tile_regs_commit();

        in0_dfb_obj.pop_front(static_cast<uint16_t>(dst_tiles));
        in0_dfb_obj.reserve_back(static_cast<uint16_t>(dst_tiles));

        tile_regs_wait();
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            pack_tile(i, in0_dfb);
        }
        tile_regs_release();

        in0_dfb_obj.push_back(static_cast<uint16_t>(dst_tiles));
    }
}

void kernel_main() {
    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto logWt = get_arg(args::logWt);
    constexpr auto seed = get_arg(args::seed);
    constexpr auto tile_width = get_arg(args::tile_width);
    // Stable top-k: on exact value ties the candidate at the lowest position wins, so the sampled
    // token does not depend on how the bitonic network happens to swap equal values.
    constexpr bool stable_sort = get_arg(args::stable_sort) == 1;
    generate_rand_tile(dfb::rand_tile, seed);

    const uint32_t nearest32_K = 32;
    const uint32_t logk = 5;  // log(32)

    // top-k
#if SAMPLING_TOPK_FUSED_32B_DEST
    if constexpr (stable_sort && !DST_ACCUM_MODE) {
        top_k_fused_32b_dest<
            Ht,
            Wt,
            nearest32_K,
            logWt,
            logk,
            dfb::input_values,
            dfb::index,
            dfb::input_transposed,
            dfb::index_transposed,
            dfb::values,
            dfb::output_ind,
            tile_width>();
    } else
#endif
    top_k<
        Ht,
        Wt,
        nearest32_K,
        logWt,
        logk,
        dfb::input_values,
        dfb::index,
        dfb::input_transposed,
        dfb::index_transposed,
        dfb::values,
        dfb::output_ind,
        tile_width,
        true,
        stable_sort>();
    constexpr uint32_t Kt = nearest32_K / tile_width;

    // scale temperature

    // mask out all values except the top-k
    DataflowBuffer topk_mask_dfb(dfb::topk_mask);
    topk_mask_dfb.wait_front(Kt);
    add_block_inplace(dfb::values, dfb::topk_mask, Ht * Kt);
    mul_block_bcast_scalar_inplace<dfb::values, dfb::temp, Ht * Kt>();
    // softmax
    reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb::values, dfb::scaler_max, dfb::cur_max, Ht, Kt>();

    sub_exp_block_bcast_cols_inplace<dfb::values, dfb::cur_max, Ht, Kt>();
    reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb::values, dfb::scaler_sum, dfb::cur_sum, Ht, Kt>();
    recip_block_inplace(dfb::cur_sum, Ht);
    mul_block_bcast_cols(dfb::values, dfb::cur_sum, dfb::local_vals, Ht, Kt);

    // Buffers this kernel waited and left unpopped, popped here so they are left balanced.
    // sub_exp_block_bcast_cols_inplace waits Ht tiles of dfb::cur_max, which is produced and
    // consumed entirely within this kernel. add_block_inplace waits Ht * Kt tiles of
    // dfb::topk_mask, and mul_block_bcast_scalar_inplace waits 1 tile of dfb::temp.
    DataflowBuffer(dfb::cur_max).pop_front(Ht);
    DataflowBuffer(dfb::topk_mask).pop_front(Ht * Kt);
    DataflowBuffer(dfb::temp).pop_front(1);

    // dfb::scaler_max and dfb::scaler_sum are pushed once by the writer and waited inside
    // compute_kernel_lib::reduce, which leaves them unpopped so one pushed tile serves every reduce
    // call. Pop both here so they are left balanced.
    DataflowBuffer(dfb::scaler_max).pop_front(1);
    DataflowBuffer(dfb::scaler_sum).pop_front(1);
}
