// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_srcs.h"

using namespace ckernel;
using namespace ckernel::math;

// SrcS SFPU pipeline (UNP_S -> SrcS -> SFPU -> PACK1 -> L1), all on one TRISC (ISOLATE_SFPU).
//
// SrcS index convention (owned here so callers do not repeat it):
//   - The input tile is unpacked into the first SrcS slice, read from SFPU_SRCS_BASE_ADDR.
//   - The SFPU result is written to the third SrcS slice, at SFPU_SRCS_BASE_ADDR + 2 * YDIM,
//     which the packer then streams out to L1.
// One 32x32 tile spans srcs_dims::slice_count(mode) SrcS slices; each slice holds
// (YDIM >> 1) SFPU rows (SFP_ROWS == 2).

/**
 * @brief Configure the SrcS SFPU pipeline for a unary op: unpack (UNP_S), pack (PACK1) and SFPU.
 *
 * Builds the unpack/pack buffer descriptors, programs the SrcS auto-loop config for one tile and
 * initializes the SFPU. SrcS geometry (XDIM/YDIM/ZDIM/slice_count) is derived internally from the
 * unpack destination format, so callers do not pass operand-derivable dimensions.
 *
 * @tparam INSTRN_COUNT: Number of unpack/pack instructions in the SrcS auto-loop (see llk_srcs.h).
 * @param l1_in_addr_16B: L1 input buffer address (16B units) unpacked into SrcS.
 * @param unpack_S_src_format: L1 input format for the unpacker.
 * @param unpack_S_dst_format: SrcS destination format (also selects 32-bit SrcS mode).
 * @param buf_desc_id_unpack: Buffer descriptor table ID for the unpack buffer (0-31).
 * @param l1_out_addr_16B: L1 output buffer address (16B units) packed from SrcS.
 * @param pack_S_src_format: SrcS source format for the packer.
 * @param pack_S_dst_format: L1 output format for the packer.
 * @param buf_desc_id_pack: Buffer descriptor table ID for the pack buffer (0-31).
 * @param implied_math_format: When false, disables implied SrcS math format for this TRISC.
 */
template <std::uint8_t INSTRN_COUNT = 1>
inline void _llk_sfpu_srcs_init_(
    const std::uint32_t l1_in_addr_16B,
    const DataFormat unpack_S_src_format,
    const DataFormat unpack_S_dst_format,
    const std::uint8_t buf_desc_id_unpack,
    const std::uint32_t l1_out_addr_16B,
    const DataFormat pack_S_src_format,
    const DataFormat pack_S_dst_format,
    const std::uint8_t buf_desc_id_pack,
    const bool implied_math_format)
{
    const bool srcs_32bit_mode = _is_srcs_32bit_mode_(unpack_S_dst_format);
    const std::uint32_t ydim   = srcs_dims::ydim(srcs_32bit_mode);

    // Unpack BD: L1 input -> SrcS
    buffer_descriptor_u bd_unpack = {0};
    tdma_descriptor_t td_unpack;
    bd_unpack.f.l1_addr_16B   = l1_in_addr_16B;
    bd_unpack.f.format        = static_cast<std::uint8_t>(unpack_S_src_format);
    bd_unpack.f.x_dim         = srcs_dims::XDIM;
    bd_unpack.f.y_dim         = ydim;
    bd_unpack.f.z_dim         = srcs_dims::ZDIM;
    td_unpack.buf_desc        = bd_unpack;
    td_unpack.buf_desc_id     = buf_desc_id_unpack;
    td_unpack.reg_data_format = static_cast<std::uint8_t>(unpack_S_dst_format);
    _configure_buf_desc_table_(td_unpack.buf_desc_id, td_unpack.buf_desc);
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(td_unpack);

    // Pack BD: SrcS -> L1 output
    buffer_descriptor_u bd_pack = {0};
    tdma_descriptor_t td_pack;
    bd_pack.f.l1_addr_16B   = l1_out_addr_16B;
    bd_pack.f.format        = static_cast<std::uint8_t>(pack_S_dst_format);
    bd_pack.f.x_dim         = srcs_dims::XDIM;
    bd_pack.f.y_dim         = ydim;
    bd_pack.f.z_dim         = srcs_dims::ZDIM;
    td_pack.buf_desc        = bd_pack;
    td_pack.buf_desc_id     = buf_desc_id_pack;
    td_pack.reg_data_format = static_cast<std::uint8_t>(pack_S_src_format);
    _configure_buf_desc_table_(td_pack.buf_desc_id, td_pack.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK1>(td_pack);

    cfg[DISABLE_IMPLIED_SRCS_FORMAT_ADDR32 + TRISC_ID] = !implied_math_format;

    _llk_unpack_srcs_config_for_tile_<INSTRN_COUNT>(srcs_32bit_mode);
    _llk_pack_srcs_config_for_tile_<INSTRN_COUNT>(srcs_32bit_mode);
    _llk_math_eltwise_sfpu_init_();
}

/**
 * @brief Run a unary SFPU op over num_tiles tiles on the SrcS path.
 *
 * For each tile the input is unpacked to SrcS and the output packed from SrcS; for each SrcS slice
 * the caller-supplied @p sfpu_op computes the result and the SrcS valid flags are cleared. The load
 * and store base addresses (SrcS index convention above) and the per-slice SFPU row count are
 * computed here and handed to @p sfpu_op, keeping the compute op free of SrcS bookkeeping.
 *
 * @tparam INSTRN_COUNT: Number of unpack/pack instructions in the SrcS auto-loop (see llk_srcs.h).
 * @tparam SfpuOp: Callable invoked once per slice as
 *         sfpu_op(int load_base_addr, int store_base_addr, int num_sfpu_iterations).
 * @param num_tiles: Number of 32x32 tiles to process.
 * @param unpack_S_dst_format: SrcS destination format used to derive SrcS geometry.
 * @param buf_desc_id_unpack: Buffer descriptor table ID configured in _llk_sfpu_srcs_init_.
 * @param buf_desc_id_pack: Buffer descriptor table ID configured in _llk_sfpu_srcs_init_.
 * @param sfpu_op: Per-slice SFPU computation.
 */
template <std::uint8_t INSTRN_COUNT = 1, typename SfpuOp>
inline void _llk_sfpu_srcs_(
    const std::uint32_t num_tiles,
    const DataFormat unpack_S_dst_format,
    const std::uint8_t buf_desc_id_unpack,
    const std::uint8_t buf_desc_id_pack,
    SfpuOp&& sfpu_op)
{
    const bool srcs_32bit_mode      = _is_srcs_32bit_mode_(unpack_S_dst_format);
    const std::uint32_t ydim        = srcs_dims::ydim(srcs_32bit_mode);
    const std::uint32_t slice_count = srcs_dims::slice_count(srcs_32bit_mode);

    const int num_sfpu_iterations = static_cast<int>(ydim >> 1); // SFP_ROWS == 2
    const int load_base_addr      = ckernel::math::SFPU_SRCS_BASE_ADDR;
    const int store_base_addr     = ckernel::math::SFPU_SRCS_BASE_ADDR + 2 * static_cast<int>(ydim);

    for (std::uint32_t i = 0; i < num_tiles; ++i)
    {
        _llk_unpack_srcs_<INSTRN_COUNT>(buf_desc_id_unpack, i * slice_count); // Sets dvalid for SFPU to read

        // Pack is issued before the SFPU loop: the SFPU loop fills the instruction buffer and can
        // clog it, leading to hangs if the pack is queued after.
        _llk_pack_srcs_<INSTRN_COUNT>(buf_desc_id_pack, i * slice_count); // Sets dvalid for SFPU to write

        for (std::uint32_t slice = 0; slice < slice_count; slice++)
        {
            sfpu_op(load_base_addr, store_base_addr, num_sfpu_iterations);
            _llk_math_eltwise_sfpu_srcs_clear_vlds_<true, true>(); // Clears dvalid for SFPU read and write
        }
    }
}
