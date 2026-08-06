// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_srcs.h"

using namespace ckernel;
using namespace ckernel::math;

// Unary SFPU pipeline over SrcS, on the ISOLATE_SFPU TRISC:
//     L1 -> UNP_S -> SrcS -> SFPU -> SrcS -> PACK1 -> L1
//
// SrcS is 2 banks x 3 slices. Slices 0/1 are written by UNP_S and read by the
// SFPU; slice 2 is written by the SFPU and read by PACK1. A unary op uses input
// slice 0 and output slice 2. The *bank* is what rotates, and each engine has
// its own bank pointer, advanced by that engine's done signal:
//
//     unpack sets data_valid  -> SFPU read-done clears it, freeing the bank
//     SFPU store sets pack_ready -> PACK1 read-done clears it, recycling the
//                                   write credit the SFPU needs for the next slice
//
// Two banks is therefore the exact free-running depth of this pipeline: the
// first two slices (= the first 16x16 face of a 32x32 tile) complete on reset
// credits, and every slice after that depends on PACK1 having recycled a
// credit. Starving the unpacker, or flooding the 8-entry instruction buffer so
// the pack/SFPU done markers are delayed, corrupts everything past the first
// face while leaving face 0 correct.
//
// This implementation therefore mirrors the streaming structure validated by
// isolate_sfpu_add_quasar_test.cpp rather than the simpler shape of
// isolate_sfpu_square_quasar_test.cpp:
//   - the per-slice SFPU body lives in the replay buffer, so the instruction
//     buffer holds a single TT_REPLAY per slice instead of the whole body;
//   - the unpacker is preloaded several slices ahead so the SFPU never waits
//     on it, then drained after the steady-state loop.
//
// A 32x32 tile spans slice_count(mode) slices of ydim(mode) rows each; the SFPU
// covers SFP_ROWS (2) rows per pass, so one slice is ydim >> 1 passes.
//
// TODO: SrcS geometry is fixed at 32x32 (srcs_dims is not TensorShape
// programmable yet), so these helpers reject other tile shapes by construction.

/**
 * @brief Slices the unpacker runs ahead of the SFPU, to keep a bank always ready.
 *
 * SrcS has SRCS_BANK_COUNT banks and a unary op consumes one bank per dvalid, so
 * at most SRCS_BANK_COUNT slices can be in flight. Preloading more than
 * SRCS_BANK_COUNT - 1 oversubscribes the banks and the SFPU re-processes a slice,
 * reading back its own output slice: the result is the op applied twice
 * (x^4 for square) on the oversubscribed slices, not stale or missing data.
 *
 * Note the binary SrcS kernels preload deeper because each of their unpack pairs
 * fills two slices under a single dvalid, which consumes bank credits at a
 * different rate -- do not copy their preload depth here.
 */
constexpr std::uint32_t SRCS_BANK_COUNT    = 2;
constexpr std::uint32_t SRCS_PRELOAD_COUNT = 3;

/**
 * @brief Unpack one L1 chunk into the SrcS input slices and hand the bank to the SFPU.
 *
 * Issues the same unpack pair the binary SrcS kernels use
 * (isolate_sfpu_add_quasar_test.cpp), which is the streaming shape empirically
 * validated across all four faces of a tile: the first UNPACR2 writes slice 0 and
 * advances the SrcS slice pointer without publishing the bank, the second writes
 * slice 1, advances the L1 tile index and sets dvalid.
 *
 * A unary op only reads slice 0, so both halves of the pair address the same
 * buffer descriptor and the same L1 chunk -- slice 1 receives a duplicate of the
 * input and is never read. That costs unpack bandwidth we would rather not spend,
 * but a single-UNPACR2 variant (write slice 0, set dvalid, never touch slice 1)
 * makes the SFPU re-run on already-computed slices and produce the op applied
 * twice, so the pair is load-bearing until the single-slice handshake is
 * understood. See tt-llk issue #1635 for the related auto-loop erratum.
 *
 * @param buf_desc_id: Buffer descriptor ID of the L1 input, values = 0-31.
 */
inline void _llk_unpack_srcs_slice_(const std::uint8_t buf_desc_id)
{
    TT_UNPACR2_TILE_INC(0b1 /*advance SrcS slice*/, 0b0 /*hold L1 tile*/, buf_desc_id, 0b0 /*no dvalid*/);
    TT_UNPACR2_TILE_INC(0b0 /*hold SrcS slice*/, 0b1 /*advance L1 tile*/, buf_desc_id, 0b1 /*set dvalid*/);
}

/**
 * @brief Configure the unary SrcS SFPU pipeline: unpack (UNP_S), pack (PACK1) and SFPU.
 *
 * Programs the SrcS auto-loop for the packer, points UNP_S and PACK1 at their
 * buffer descriptors and inits the SFPU. SrcS geometry is derived from the
 * unpack descriptor's register format, not passed by the caller.
 *
 * @tparam INSTRN_COUNT: Pack instructions per SrcS auto-loop (see llk_srcs.h).
 * @param td_unpack: Descriptor for the L1 input, already registered in the buffer descriptor table.
 * @param td_pack: Descriptor for the L1 output, already registered in the buffer descriptor table.
 * @param implied_math_format: When false, disables implied SrcS math format for this TRISC.
 * @note Pair with @ref _llk_sfpu_srcs_ to run the op and @ref _llk_sfpu_srcs_done_ to drain.
 */
template <std::uint8_t INSTRN_COUNT = 1>
inline void _llk_sfpu_srcs_init_(const tdma_descriptor_t& td_unpack, const tdma_descriptor_t& td_pack, const bool implied_math_format = true)
{
    const bool srcs_32bit_mode = _is_srcs_32bit_mode_(static_cast<DataFormat>(td_unpack.reg_data_format));

    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(td_unpack);
    _llk_pack_hw_configure_<p_pacr::PACK1, false>(td_pack, ckernel::ReluConfig::none());

    cfg[DISABLE_IMPLIED_SRCS_FORMAT_ADDR32 + TRISC_ID] = !implied_math_format;

    // Only the packer gets an auto-loop. The unpacker is driven explicitly below
    // so its instructions can be interleaved with the SFPU body for preloading.
    _llk_pack_srcs_config_for_tile_<INSTRN_COUNT>(srcs_32bit_mode);
    _llk_math_eltwise_sfpu_init_();
}

/**
 * @brief Run a unary SFPU op over num_tiles tiles on the SrcS path.
 *
 * Loads @p sfpu_op into the replay buffer once, then per tile: arms the L1 read
 * and write counters, preloads the unpacker, and streams every slice through
 * the SFPU, clearing the SrcS valids after each so the banks recycle.
 *
 * @tparam INSTRN_COUNT: Must match the value passed to _llk_sfpu_srcs_init_.
 * @tparam SfpuOp: Callable emitting the per-slice SFPU instruction sequence.
 * @param num_tiles: Number of 32x32 tiles to process.
 * @param td_unpack: Descriptor for the L1 input.
 * @param td_pack: Descriptor for the L1 output.
 * @param replay_buf_len: Instruction count of one @p sfpu_op expansion.
 * @param sfpu_op: Reads input slice 0, writes output slice 2, for one whole slice.
 * @note Call @ref _llk_sfpu_srcs_init_ first and @ref _llk_sfpu_srcs_done_ afterwards.
 */
template <std::uint8_t INSTRN_COUNT = 1, typename SfpuOp>
inline void _llk_sfpu_srcs_(
    const std::uint32_t num_tiles, const tdma_descriptor_t& td_unpack, const tdma_descriptor_t& td_pack, const std::uint32_t replay_buf_len, SfpuOp&& sfpu_op)
{
    const bool srcs_32bit_mode      = _is_srcs_32bit_mode_(static_cast<DataFormat>(td_unpack.reg_data_format));
    const std::uint32_t slice_count = srcs_dims::slice_count(srcs_32bit_mode);
    const std::uint32_t ydim        = srcs_dims::ydim(srcs_32bit_mode);

    // Unary slice convention: read input slice 0, write output slice 2.
    const int num_sfpu_iterations = static_cast<int>(ydim >> 1); // SFP_ROWS == 2
    const int load_base_addr      = ckernel::math::SFPU_SRCS_BASE_ADDR;
    const int store_base_addr     = ckernel::math::SFPU_SRCS_BASE_ADDR + 2 * static_cast<int>(ydim);

    // The SFPU body is replayed rather than inlined: at slice_count slices per
    // tile it would otherwise flood the 8-entry instruction buffer and delay the
    // done markers that recycle the SrcS bank credits.
    load_replay_buf(0, replay_buf_len, false, 0, 0, [&] { sfpu_op(load_base_addr, store_base_addr, num_sfpu_iterations); });

    for (std::uint32_t i = 0; i < num_tiles; ++i)
    {
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, i * slice_count);

        // Pack is armed before the SFPU work: PACK1 recycles the write credit
        // each SFPU store consumes, so it has to already be draining slice 2.
        _llk_pack_srcs_<INSTRN_COUNT>(td_pack.buf_desc_id, i * slice_count);

        // Run the unpacker ahead of the SFPU so a bank is always ready.
#pragma GCC unroll SRCS_PRELOAD_COUNT
        for (std::uint32_t j = 0; j < SRCS_PRELOAD_COUNT; j++)
        {
            _llk_unpack_srcs_slice_(td_unpack.buf_desc_id);
        }

        for (std::uint32_t slice = 0; slice < slice_count - SRCS_PRELOAD_COUNT; slice++)
        {
            _llk_unpack_srcs_slice_(td_unpack.buf_desc_id);
            TT_REPLAY(0, replay_buf_len, 0, 0, 0, 0);
            _llk_math_eltwise_sfpu_srcs_clear_vlds_<true, true>(); // Recycles the read and write banks
        }

        // Drain the slices whose unpack was preloaded above.
#pragma GCC unroll SRCS_PRELOAD_COUNT
        for (std::uint32_t j = 0; j < SRCS_PRELOAD_COUNT; j++)
        {
            TT_REPLAY(0, replay_buf_len, 0, 0, 0, 0);
            _llk_math_eltwise_sfpu_srcs_clear_vlds_<true, true>();
        }
    }
}

/**
 * @brief Drain the SFPU, unpacker and packer after a SrcS SFPU op.
 *
 * @note Call once after the final @ref _llk_sfpu_srcs_ of an operation.
 */
inline void _llk_sfpu_srcs_done_()
{
    wait_sfpu_idle();
    wait_unpack_idle();
    wait_pack_idle();
}
