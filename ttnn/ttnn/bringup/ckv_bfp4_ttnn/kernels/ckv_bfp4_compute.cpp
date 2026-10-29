// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Per band of 32 rows (16 bf16 tiles across 512 columns), all 16 tiles in a full-sync DEST:
//   copy -> SFPU round onto the bfloat4_b grid (CKV_MBITS 3; bfloat8_b: 7) (ckv_round_bfp4.hpp, the host packer's rule)
//   [CKV_ROUNDED_OUT: the 16 tiles packed as bf16 to cb_rounded]
//   the 32 row records packed straight into cb_rows by the packer, no data-movement RISC work: a record is
//   [32 exponents | the same 32 | 256 nibble bytes] (320 B; bfloat8_b: 512 value bytes, 576 B). A bfloat4_b pack writes
//   one stream: the exponents (one byte per 16 values) from its address, the values Exp_section_size x 16 B later. Each
//   key row is 16 PACRs (a tile row, both face halves) closed by Last, so stream A (address +0, section 64 B) writes
//   exponents at +0 and values at +64, and stream B (address +32, section 32 B) the exponents at +32 and the same
//   values again. The values are on the bfp4 grid, so the pack only copies their bits.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ckv_round_bfp4.hpp"

#ifndef CKV_MBITS
#define CKV_MBITS 3
#endif
#ifndef CKV_ROW_BYTES
#define CKV_ROW_BYTES 320
#endif

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack_untilize.h"
#include "ckernel_template.h"

namespace ckv_rows_pack {
using namespace ckernel;
using namespace ckernel::packer;

constexpr uint32_t TILES = 16;       // a band: 512 columns
constexpr uint32_t ROW_BYTES = CKV_ROW_BYTES;  // a record

// the packer reads one tile row per PACR (two interfaces: face halves) stepping Z over the band's tiles; the row's
// last PACR closes the L1 stream (Last) and steps the face row and the L1 row (ADDR_MOD_1)
inline void init(uint32_t src_bytes) {  // src_bytes: the packer's input datum size (its DEST addressing unit)
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    _llk_pack_untilize_configure_addrmod_();
    const uint32_t pacr = TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        ADDR_MOD_0,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::TWO_INTFS_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        0);
    const uint32_t close = TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::TWO_INTFS_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        1);
    ckernel_template mop(FACE_R_DIM, TILES, pacr);
    mop.set_last_inner_loop_instr(close);
    mop.set_last_outer_loop_instr(close);
    mop.program();
    // DEST: the next tile is a tile of datums on (Z, in the input format's bytes); L1: the next record is ROW_BYTES on
    // (channel 1 Y, set per pass)
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(TILE_NUM_FACES * FACE_R_DIM * FACE_C_DIM * src_bytes);
    // DEST: the next face row is 16 datums on (Y)
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_XY_REG_0_Ystride_RMW>(FACE_C_DIM * src_bytes);
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
}

#ifdef CKV_BLOCKS
// blocks: one stream for the whole band (exponents of all 32 rows, then their values): a row steps only the face row
inline void program_mop(bool close_at_end) {
    const uint32_t pacr = TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        ADDR_MOD_0,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::TWO_INTFS_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        0);
    const uint32_t step = TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::TWO_INTFS_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        0);
    const uint32_t close = TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::TWO_INTFS_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        1);
    ckernel_template mop(FACE_R_DIM, TILES, pacr);
    mop.set_last_inner_loop_instr(step);
    mop.set_last_outer_loop_instr(close_at_end ? close : step);
    mop.program();
}
#endif

// one stream over the band's 32 rows: address byte_addr, exponent section exp16 x 16 B
inline void pass(uint32_t byte_addr, uint32_t exp16) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Exp_section_size_RMW>(exp16);
    program_packer_destination(byte_addr / 16 - 1);
#ifdef CKV_BLOCKS
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_XY_REG_1_Ystride_RMW>(0);
    program_mop(false);
#else
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_XY_REG_1_Ystride_RMW>(ROW_BYTES);
#endif
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);  // ch0 / ch1 Z
    TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 0);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b1011);  // ch0 X Y, ch1 Y
    ckernel_template::run();                          // rows 0..15 (faces 0 / 1)
#ifdef CKV_BLOCKS
    program_mop(true);
#endif
    TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 2 * FACE_R_DIM);
    TTI_SETADC(p_setadc::PAC, p_setadc::CH_1, p_setadc::SET_Y, FACE_R_DIM);
    ckernel_template::run();  // rows 16..31 (faces 2 / 3)
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b1010);
}

// back to the default tile pack: Exp_section_size = 4 faces, strides as set_packer_strides leaves them
inline void uninit(uint32_t src_bytes) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Exp_section_size_RMW>(4);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_XY_REG_1_Ystride_RMW>(0);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(FACE_R_DIM * FACE_C_DIM * src_bytes);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
}
}  // namespace ckv_rows_pack
#endif

void kernel_main() {
    const uint32_t n_bands = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rows = get_compile_time_arg_val(1);
#ifdef CKV_ROUNDED_OUT
    constexpr uint32_t cb_rounded = get_compile_time_arg_val(2);
#endif
    constexpr uint32_t TILES = 16;

    compute_kernel_hw_startup(cb_in, cb_rows);
    // BH: the strided (row) DEST read the rows pack uses needs the DEST address remap, as pack untilize sets it
    MATH((llk_math_reconfig_remap(true)));
    MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
    for (uint32_t b = 0; b < n_bands; ++b) {
        cb_wait_front(cb_in, TILES);
        copy_init(cb_in);
        tile_regs_acquire();
        for (uint32_t t = 0; t < TILES; ++t) {
            copy_tile(cb_in, t, t);
            MATH((
                _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::ckv_round_bfp_face<CKV_MBITS>, t, VectorMode::RC)));
        }
        tile_regs_commit();
        cb_pop_front(cb_in, TILES);
        tile_regs_wait();
#ifdef CKV_ROUNDED_OUT
        cb_reserve_back(cb_rounded, TILES);
        pack_reconfig_data_format(cb_rounded);
        pack_init(cb_rounded);
        for (uint32_t t = 0; t < TILES; ++t) {
            pack_tile(t, cb_rounded);
        }
        cb_push_back(cb_rounded, TILES);
#endif
        cb_reserve_back(cb_rows, 1);
        pack_reconfig_data_format(cb_rows);
        PACK(({
            const uint32_t rows_addr = get_local_cb_interface(cb_rows).fifo_wr_ptr << 4;
            const uint32_t src_bytes = ckernel::datum_size_in_bytes(pack_src_format[cb_rows]);
            ckv_rows_pack::init(src_bytes);
#ifdef CKV_BLOCKS
            // [32 rows' exponents | 32 B | the same | to 64 B | 32 rows' values] (sparse_sdpa window_block_*)
            ckv_rows_pack::pass(rows_addr, CKV_BLOCK_DATA / 16);
            ckv_rows_pack::pass(rows_addr + CKV_BLOCK_EXP2, (CKV_BLOCK_DATA - CKV_BLOCK_EXP2) / 16);
#else
            ckv_rows_pack::pass(rows_addr, 4);       // exponents at +0, values at +64
            ckv_rows_pack::pass(rows_addr + 32, 2);  // exponents at +32, values at +64 again
#endif
            ckv_rows_pack::uninit(src_bytes);
        }));
        cb_push_back(cb_rows, 1);
        tile_regs_release();
    }
}
