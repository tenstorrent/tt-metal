// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize compute — block operation `tilize_block`.
//
// One compute_kernel_lib::tilize call per block (block_height_tiles tile-rows x width tiles):
// init / uninit / reconfig once per block (Phase 0: once per core). The helper owns all CB
// wait/pop/reserve/push (symmetric tile-sized pages, block width per tile-row). The ragged last
// column block is a second instantiation on its own CB pair.
//
// Fp32Mode::Lossless: a 32-bit input CB is tagged UnpackToDestFp32 by the host (with
// fp32_dest_acc_en), so fp32 takes the bit-exact standard tilize path instead of fast tilize
// (tf32 truncation). The mode has no effect on non-fp32 input formats.
//
// bfloat4_b output (bfp4_round): the packer TRUNCATES every 16-element shared-exponent group to a
// 3-bit magnitude. Before the tilize, each window of round_tile_rows tile-rows is streamed through
// `unary<Bfp4Round>` (copy page -> DEST, SFPU round-to-nearest-even onto the group's bfp4 grid,
// pack bf16) into cb_bfp4_rounded, and the tilize reads that. A 16-element group is 16 contiguous
// elements of a stick, i.e. one 16-wide face row of the (non-tilized) page in DEST, so the group
// is visible to the SFPU without the tilize. On-grid values survive the packer's truncation
// exactly: the output equals the host's round-to-nearest bfp4 conversion. The custom SFPU element
// exists because no kernel_lib / LLK op rounds onto a per-row shared-exponent grid.

// Helper bypass (Perf 2, measured): on the reorder-free coalesced path (TILIZE_REORDER_FREE) the
// tilize does NOT go through compute_kernel_lib::tilize. The reader lands each chunk bank-grouped
// (no L1->L1 reorder copies) and the fast-tilize UNPACK step is replaced by rft_unpack_block, which
// re-programs the unpacker's X/Y dims so its Y/Z address counters walk sticks across bank groups;
// fast-tilize init/uninit are called at LLK level (the API routes ct == 1 to standard tilize on BH).
// No helper can give the tilize unpacker a per-row source address pattern. Op [1,1,16384,64]
// 11.02 -> 10.53 us, [1,1,16384,32] 6.92 -> 5.74 us (BH p150b). Do not revert to the helper call
// without re-measuring; see changelog "Perf 2".

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#endif

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// x rounded to nearest-even on the bfp4 grid of its group: step 2^(E - 127 - 2) where E (biased)
// is the group's max exponent, magnitude clamped to 7 steps (host converter saturation). In units
// of the step |x| is in [0, 8); |x| / step + 2^23 lands in [2^23, 2^23 + 8) where the fp32 ulp is
// 1, so the MAD rounds it to an integer, half to even. Clamp there, then one more MAD scales back
// and removes the 2^23 (both exact: powers of two times small integers). Elements below half a
// step become 0. Groups with E < 3 (denormal range; the 2^(2 - Eu) inverse would overflow) -> 0.
constexpr float BFP4_RNE_MAGIC = 8388608.0f;  // 2^23
constexpr float BFP4_MAX_STEPS = 7.0f;        // 3-bit magnitude
sfpi_inline sfpi::vFloat bfp4_round_value(
    sfpi::vFloat x, sfpi::vFloat step, sfpi::vFloat inv_step, sfpi::vFloat neg_magic_step) {
    sfpi::vFloat q = sfpi::abs(x) * inv_step + BFP4_RNE_MAGIC;
    q = sfpi::min(q, BFP4_RNE_MAGIC + BFP4_MAX_STEPS);
    return sfpi::copysgn(q * step + neg_magic_step, x);
}

// One face (16 rows x 16 cols) per call; VectorMode::RC calls it once per face. A row band is
// dst_reg[0] (rows r..r+3, cols 0..7) + dst_reg[1] (same rows, cols 8..15); within a register a
// face row is one 8-lane sub-vector, so a row's max is a lane-wise max of the two halves followed
// by a max over the sub-vector (rotate-by-1/2/4 within the sub-vector).
inline void calculate_bfp4_round() {
    constexpr int ROW_BANDS_PER_FACE = 4;
#pragma GCC unroll 4
    for (int band = 0; band < ROW_BANDS_PER_FACE; ++band) {
        const sfpi::vFloat lo = sfpi::dst_reg[0];
        const sfpi::vFloat hi = sfpi::dst_reg[1];
        sfpi::vFloat m = sfpi::max(sfpi::abs(lo), sfpi::abs(hi));
        sfpi::vFloat r = sfpi::vFloat(sfpi::subvec_shflror1(m));
        m = sfpi::max(m, r);
        r = sfpi::vFloat(sfpi::subvec_shflror1(m));
        r = sfpi::vFloat(sfpi::subvec_shflror1(r));
        m = sfpi::max(m, r);
        r = sfpi::vFloat(sfpi::subvec_shflror1(m));
        r = sfpi::vFloat(sfpi::subvec_shflror1(r));
        r = sfpi::vFloat(sfpi::subvec_shflror1(r));
        r = sfpi::vFloat(sfpi::subvec_shflror1(r));
        m = sfpi::max(m, r);
        const sfpi::vInt emax_b = sfpi::exexp(m, sfpi::ExponentMode::Biased);
        const sfpi::vFloat step = sfpi::setexp(sfpi::vFloat(1.0f), emax_b - 2);          // 2^(Eu - 2)
        sfpi::vFloat inv_step = sfpi::setexp(sfpi::vFloat(1.0f), (127 + 129) - emax_b);  // 2^(2 - Eu)
        v_if(emax_b < 3) { inv_step = 0.0f; }
        v_endif;
        const sfpi::vFloat neg_magic_step = step * -BFP4_RNE_MAGIC;
        sfpi::dst_reg[0] = bfp4_round_value(lo, step, inv_step, neg_magic_step);
        sfpi::dst_reg[1] = bfp4_round_value(hi, step, inv_step, neg_magic_step);
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
#endif

#ifdef TILIZE_REORDER_FREE
// ---------------------------------------------------------------------------------------------
// reorder_free_tilize (perf experiment). Raw LLK: bypasses compute_kernel_lib::tilize /
// fast_tilize_block's UNPACK step. The reader no longer restores row order with per-stick L1->L1
// NoC copies; the chunk lands bank-grouped in cb_rm_sticks (stick s of the chunk at group s % 8,
// group row s / 8, group stride = group_rows sticks) and the fast-tilize UNPACKER undoes the
// permutation through its address counters: XDim := one stick (ct*32 datums) so CH0_Y steps one
// stick, YDim := group_rows so CH0_Z steps one bank group. Each dvalid batch of 8 reads (sticks
// 8d..8d+7 of a tile-row) walks Z = 0..7 (groups) at Y = d; the dvalid read bumps Y, a SETADCZW
// resets CH0_Z. Math / pack fast-tilize stages are unchanged (same SrcA contract per dvalid).
// No helper exposes a per-row source address pattern for the tilize unpacker.
// ---------------------------------------------------------------------------------------------
#ifdef TRISC_UNPACK
inline void rft_unpack_override(uint32_t ct_dim, uint32_t group_rows) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    // XDim (multi-context: Tile_x_dim_cntx0/1) = one stick.
    TT_SETDMAREG(0, ct_dim * TILE_C_DIM, 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, ct_dim * TILE_C_DIM, 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    // YDim = group_rows (sticks per bank group), ZDim = 16 (unused).
    TT_SETDMAREG(0, group_rows, 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, FACE_R_DIM, 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    // MOP: plain read = CH0_Z+=1 (next group), CH1_Z+=1; dvalid read = CH0_Y+=1 (next stick), CH1_Z+=1.
    constexpr std::uint8_t ADDRMOD_READ = 0b00'01'00'01;
    constexpr std::uint8_t ADDRMOD_LAST = 0b00'01'01'00;
    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,
        false,
        TT_OP_UNPACR_COMMON(SrcA, ADDRMOD_READ, 0),
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_UNPACR_COMMON(SrcA, ADDRMOD_LAST, 1),
        TT_OP_NOP,
        TT_OP_NOP);
    tmp.program();
}

// One unit (unit_dim tiles) of tile-row `t` of the chunk at the CB read pointer.
inline void rft_unpack_block(uint32_t operand, uint32_t t, uint32_t col_tile, uint32_t stick_bytes) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t base_address =
        (get_local_cb_interface(operand_id).fifo_rd_ptr - 1) + ((t * 4 * stick_bytes + col_tile * TILE_C_DIM * 2) >> 4);
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    _llk_unpack_configure_single_address_(base_address, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    TTI_SETADCXY(p_setadc::UNP_A, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b1111);
    TT_MOP_CFG(0);
#pragma GCC unroll 4
    for (uint32_t d = 0; d < 4; ++d) {
        TT_MOP(0, 8 - 1, 0x80);                             // 8 reads, dvalid on the 8th
        TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0001);  // CH0_Z = 0 (group 0)
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

template <uint32_t ct, uint32_t cb_in, uint32_t cb_out>
ALWI void rft_tilize_row(uint32_t t) {
    constexpr uint32_t stick_bytes = ct * TILE_C_DIM * 2;
    uint32_t tiles_done = 0;
    uint32_t prev_chunk = 0;
    PACK((llk_pack_fast_tilize_row_begin(cb_out, 0)));
    while (tiles_done < ct) {
        const uint32_t remaining = ct - tiles_done;
        const uint32_t chunk = (remaining > 5) ? 4 : (remaining == 5) ? 2 : remaining;
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));
        if (chunk != prev_chunk) {
            UNPACK((llk_unpack_fast_tilize_reinit_xdim(chunk)));
            PACK((llk_pack_fast_tilize_reinit_unit_dim(cb_out, chunk)));
            prev_chunk = chunk;
        }
        UNPACK((rft_unpack_block(cb_in, t, tiles_done, stick_bytes)));
        MATH((llk_math_fast_tilize_block_<DST_ACCUM_MODE>(0, cb_in, 4)));
        PACK((llk_pack_fast_tilize_row_chunk(0, cb_out, chunk)));
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
        tiles_done += chunk;
    }
    PACK((llk_pack_fast_tilize_row_end()));
}
#endif

namespace {
// Chain element: round DEST slot D0 onto the bfp4 grid (see calculate_bfp4_round).
struct Bfp4Round : compute_kernel_lib::UnaryOp<Bfp4Round, compute_kernel_lib::Dst::D0> {
    static ALWI void init() { MATH(SFPU_UNARY_INIT(unused)); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_bfp4_round, slot_offset, VectorMode::RC));
    }
};

// `reconfigure`: only when the tilize source format differs from the one compute_kernel_hw_startup
// configured (the bfp4 round pass: unary leaves the unpacker on cb_in, the tilize reads the bf16
// cb_bfp4_rounded). Every other source (cb_rm_sticks / its tail) has the startup format, and the
// reconfigure is pure fixed cost: ~150-200 ns per launch on small sharded shards (BH, measured).
template <uint32_t block_width_tiles, uint32_t cb_in, uint32_t cb_out, bool reconfigure>
ALWI void tilize_block(uint32_t rows) {
    using namespace compute_kernel_lib::tilize_config;
    compute_kernel_lib::tilize<
        block_width_tiles,
        cb_in,
        cb_out,
        InitUninitMode::InitAndUninit,
        WaitMode::WaitBlock,
        reconfigure ? ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
                    : ReconfigureRegisterDatatypeMode::NoReconfigure,
        Fp32Mode::Lossless>(rows);
}

// One row block of `rows` tile-rows; with bfp4_round, rounded first in windows of
// round_tile_rows tile-rows (the round pass and the tilize share the TRISCs and alternate).
template <
    uint32_t block_width_tiles,
    uint32_t cb_in,
    uint32_t cb_out,
    bool bfp4_round,
    uint32_t cb_bfp4_rounded,
    uint32_t round_tile_rows>
ALWI void tilize_rows(uint32_t rows) {
    if constexpr (!bfp4_round) {
        // Occupancy: the helper waits on cb_in / reserves cb_out per tile-row inside this zone.
        MaybeDeviceZoneScope("compute_tilize");
        tilize_block<block_width_tiles, cb_in, cb_out, false>(rows);
    } else {
        using namespace compute_kernel_lib;
        for (uint32_t r = 0; r < rows; r += round_tile_rows) {
            const uint32_t n = (rows - r < round_tile_rows) ? rows - r : round_tile_rows;
            {
                MaybeDeviceZoneScope("compute_bfp4_round");
                unary<Bfp4Round, input(cb_in), output(cb_bfp4_rounded)>(IterationShape::tiles(n * block_width_tiles));
            }
            {
                MaybeDeviceZoneScope("compute_tilize");
                tilize_block<block_width_tiles, cb_bfp4_rounded, cb_out, true>(n);
            }
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_rm_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm_sticks_tail = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tiles_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tiles_out_tail = get_compile_time_arg_val(3);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t last_block_width_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_col_blocks = get_compile_time_arg_val(6);
    constexpr bool has_tail = get_compile_time_arg_val(7) != 0;
    constexpr bool bfp4_round = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t cb_bfp4_rounded = get_compile_time_arg_val(9);
    constexpr uint32_t round_tile_rows = get_compile_time_arg_val(10);
    constexpr uint32_t num_full_col_blocks = num_col_blocks - (has_tail ? 1 : 0);

    const uint32_t core_rows_tiles = get_arg_val<uint32_t>(0);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(1);

#ifdef TILIZE_REORDER_FREE
    {
        constexpr uint32_t staging_tile_rows = get_compile_time_arg_val(11);
        constexpr uint32_t first_chunk_tile_rows = get_compile_time_arg_val(12);
        constexpr uint32_t group_rows = get_compile_time_arg_val(13);
        constexpr uint32_t ct = block_width_tiles;
        constexpr uint32_t slot_pages = staging_tile_rows * ct;
        static_assert(num_col_blocks == 1 && !has_tail && !bfp4_round && ct >= 1, "rft domain");
        // Fast-tilize init/uninit called at LLK level: the API's fast_tilize_init routes ct == 1
        // to the standard tilize on BH; here unit_dim 1 runs through the fast-tilize MOPs too.
        constexpr uint32_t first_unit = (ct > 5) ? 4 : (ct == 5) ? 2 : ct;
        compute_kernel_hw_startup(cb_rm_sticks, cb_tiles_out);
        UNPACK((llk_unpack_fast_tilize_init(cb_rm_sticks, ct, first_unit)));
        MATH((llk_math_fast_tilize_init<DST_ACCUM_MODE>(cb_rm_sticks)));
        PACK((llk_pack_fast_tilize_init<DST_ACCUM_MODE>(cb_rm_sticks, cb_tiles_out, first_unit)));
        UNPACK((rft_unpack_override(ct, group_rows)));
        for (uint32_t row0 = 0; row0 < core_rows_tiles;) {
            const uint32_t nominal = row0 == 0 ? first_chunk_tile_rows : staging_tile_rows;
            const uint32_t rows = core_rows_tiles - row0 < nominal ? core_rows_tiles - row0 : nominal;
            {
                MaybeDeviceZoneScope("compute_cb_wait");
                cb_wait_front(cb_rm_sticks, slot_pages);
            }
            for (uint32_t t = 0; t < rows; ++t) {
                {
                    MaybeDeviceZoneScope("compute_cb_reserve");
                    cb_reserve_back(cb_tiles_out, ct);
                }
                {
                    MaybeDeviceZoneScope("compute_tilize");  // payload only: waits are zoned above
                    rft_tilize_row<ct, cb_rm_sticks, cb_tiles_out>(t);
                }
                cb_push_back(cb_tiles_out, ct);
            }
            cb_pop_front(cb_rm_sticks, slot_pages);
            row0 += rows;
        }
        UNPACK((llk_unpack_fast_tilize_uninit<DST_ACCUM_MODE>()));
        MATH((llk_math_fast_tilize_uninit<DST_ACCUM_MODE>(cb_rm_sticks)));
        PACK((llk_pack_fast_tilize_uninit<DST_ACCUM_MODE>(cb_tiles_out)));
        return;
    }
#endif
    compute_kernel_hw_startup(cb_rm_sticks, cb_tiles_out);

    for (uint32_t col_block = 0; col_block < num_full_col_blocks; ++col_block) {
        for (uint32_t rb_start = 0; rb_start < core_rows_tiles; rb_start += block_height_tiles) {
            const uint32_t rows =
                (core_rows_tiles - rb_start < block_height_tiles) ? core_rows_tiles - rb_start : block_height_tiles;
            tilize_rows<block_width_tiles, cb_rm_sticks, cb_tiles_out, bfp4_round, cb_bfp4_rounded, round_tile_rows>(
                rows);
        }
    }

    if constexpr (has_tail) {
        for (uint32_t rb_start = 0; rb_start < core_rows_tiles; rb_start += block_height_tiles) {
            const uint32_t rows =
                (core_rows_tiles - rb_start < block_height_tiles) ? core_rows_tiles - rb_start : block_height_tiles;
            tilize_rows<
                last_block_width_tiles,
                cb_rm_sticks_tail,
                cb_tiles_out_tail,
                bfp4_round,
                cb_bfp4_rounded,
                round_tile_rows>(rows);
        }
    }
}
