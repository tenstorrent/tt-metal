// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// Replay buffer: 9 block_increment reuse blocks and 1 inner_increment reuse block, 3 instructions each.
// The MOP template windows and the odd-kt_dim tail both index the buffer in reuse blocks.
constexpr std::uint32_t SDPA_CUSTOM_MM_REUSE_BLOCK_LEN                = 3;
constexpr std::uint32_t SDPA_CUSTOM_MM_REUSE_NUM_BLOCK_INCREMENT_BLKS = 9;
constexpr std::uint32_t SDPA_CUSTOM_MM_REUSE_REPLAY_BUF_LEN           = (SDPA_CUSTOM_MM_REUSE_NUM_BLOCK_INCREMENT_BLKS + 1) * SDPA_CUSTOM_MM_REUSE_BLOCK_LEN;

// SDPA_CUSTOM_MM_REUSE_DEST_SRCB
// Custom matmul that reuses SrcB from dest and only unpacks SrcA.
// Output height and width should be single tile with tile shape [1, 32].
//
// The nt_dim and kt_dim loops run as one MOP over k-row pairs, plus a direct
// replay of the last k-row when kt_dim is odd, using CFGSHIFTMASK with two
// scratch registers:
//   SCRATCH_SEC0 = block_increment (advance to next tile within a k-row)
//   SCRATCH_SEC1 = inner_increment (jump from end of one k-row to start of next)
//
// The replay buffer is 30 instructions, containing 9 block_increment
// reuse blocks followed by 1 inner_increment reuse block (3 insns each).
// The MOP template selects sliding windows into this buffer to cover any
// nt_dim from 1 to 16. Each MOP iteration covers 2 k-rows (via B-mode or
// halo-mode), so kt_dim/2 MOP iterations cover the k-row pairs; an odd kt_dim
// then replays its last k-row directly from the same buffer windows, and
// kt_dim == 1 issues no MOP at all.
//
// Constraints:
// - ct_dim = 1, rt_dim = 1 (single output tile)
// - nt_dim: 1 to 16
// - kt_dim: any integer from 1 to 256 (inclusive)
// - kernel_broadcast_a = 0, kernel_broadcast_b = 0
inline void _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_mop_config_(const std::uint32_t nt_dim)
{
    // Replay buffer layout (30 instructions):
    //   [0-26]:  9 reuse blocks with block_increment (3 insns each)
    //   [27-29]: 1 final reuse block with inner_increment (3 insns)
    //
    // Each reuse block: SrcA unpack + CFGSHIFTMASK + NOP (3 insns)
    // block_increment reuse: CFGSHIFTMASK selects SCRATCH_SEC0
    // inner_increment reuse (final): CFGSHIFTMASK selects SCRATCH_SEC1
    //
    // This supports first_half up to 9 tiles and second_half up to 9 tiles = 18 max.
    // Practical limit is 16 due to dst size.

    load_replay_buf(
        0,
        SDPA_CUSTOM_MM_REUSE_REPLAY_BUF_LEN,
        []
        {
            for (std::uint32_t i = 0; i < SDPA_CUSTOM_MM_REUSE_NUM_BLOCK_INCREMENT_BLKS; i++)
            {
                TTI_UNPACR_COMMON(SrcA, 0b00000000, 1);
                TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
                TTI_NOP;
            }
            TTI_UNPACR_COMMON(SrcA, 0b00000000, 1);
            TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 1, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_NOP;
        });

    // Mop covers two k-rows per iteration, allowing up to 256 kt_dim with 128 MOP iterations.
    // zmask is always 0 (skip path never used), which is required for iterations beyond 32.
    //
    // Each k-row processes nt_dim tiles: (nt_dim-1) tiles with block_increment, then
    // 1 tile with inner_increment. The tiles are split between two halves for the template:
    //   first_half_iterations = ceil(nt_dim/2)
    //   second_half_iterations = floor(nt_dim/2)
    //
    // first_half window: starts at buffer beginning, covers first_half_iterations * 3 insns
    // second_half window: ends at buffer end (always includes inner_increment block),
    //   covers second_half_iterations * 3 insns
    //
    // nt_dim == 1: B-mode (A0 + B), both pointing to the inner_increment block
    // nt_dim >= 2: Halo-mode (A0, A1, A2, A3), alternating first_half/second_half

    const std::uint32_t first_half_iters  = (nt_dim + 1) >> 1;
    const std::uint32_t second_half_iters = nt_dim >> 1;

    const std::uint32_t first_half_len  = first_half_iters * SDPA_CUSTOM_MM_REUSE_BLOCK_LEN;
    const std::uint32_t second_half_len = second_half_iters * SDPA_CUSTOM_MM_REUSE_BLOCK_LEN;
    const std::uint32_t first_half      = lltt::replay_insn(0, first_half_len);
    const std::uint32_t second_half     = lltt::replay_insn(SDPA_CUSTOM_MM_REUSE_REPLAY_BUF_LEN - second_half_len, second_half_len);

    // For nt_dim == 1: only inner_increment block is needed, use B-mode
    // block_increment == inner_increment when nt_dim == 1 so either block works,
    // but we use the inner_increment block (last 3 insns) for correctness
    const std::uint32_t single_tile = lltt::replay_insn(SDPA_CUSTOM_MM_REUSE_REPLAY_BUF_LEN - SDPA_CUSTOM_MM_REUSE_BLOCK_LEN, SDPA_CUSTOM_MM_REUSE_BLOCK_LEN);

    ckernel_unpack_template tmp = ckernel_unpack_template(
        nt_dim == 1,                            // B-mode when single tile per k-row
        nt_dim != 1,                            // Halo-mode when multiple tiles per k-row
        nt_dim == 1 ? single_tile : first_half, // A0
        second_half,                            // A1
        nt_dim == 1 ? single_tile : first_half, // A2
        second_half,                            // A3
        0,                                      // Skip A (unused, zmask always 0)
        single_tile,                            // B (used in B-mode for nt_dim==1)
        0                                       // Skip B (unused)
    );

    tmp.program();
    TTI_MOP_CFG(0);
}

__attribute__((always_inline)) inline void _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init_(
    const std::uint32_t nt_dim = 1, const std::uint32_t unpA_face_r_dim = FACE_R_DIM, const std::uint32_t unpA_num_faces = 4)
{
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const std::uint32_t unpA_x_end = unpA_num_faces * unpA_face_r_dim * FACE_C_DIM - 1;
    TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);

    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_mop_config_(nt_dim);

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}

// MOP over k-row pairs, plus a replay tail for the last k-row when kt_dim is odd.
// CFGSHIFTMASK auto-increments the SrcA L1 address:
//   SCRATCH_SEC0 = block_increment (tile_size_a, advances within a k-row)
//   SCRATCH_SEC1 = inner_increment (jumps from end of one k-row to start of next)
// - in1_k_stride: stride between K tiles in in1 (default 1 for contiguous, use out_w for row-major layout)
inline void _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_(
    const std::uint32_t base_address_a,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_size_a,
    const std::uint32_t kt_dim       = 1,
    const std::uint32_t nt_dim       = 1,
    const std::uint32_t in1_k_stride = 1)
{
    volatile std::uint32_t* cfg = get_cfg_pointer();

    const std::uint32_t address_a       = base_address_a + tile_size_a * tile_index_a;
    const std::uint32_t block_increment = tile_size_a;
    // inner_increment: after (nt_dim-1) block_increments we're at the last tile of the k-row.
    // We need to jump to the first tile of the next k-row.
    // After processing a full k-row, address has advanced by (nt_dim-1) * tile_size_a
    // via block_increments. The inner_increment must bring us to the next k-row:
    //   (nt_dim - 1) * tile_size_a + inner_increment = in1_k_stride * tile_size_a
    //   inner_increment = (in1_k_stride - nt_dim + 1) * tile_size_a
    const std::uint32_t inner_increment = (in1_k_stride - nt_dim + 1) * tile_size_a;

    wait_for_next_context(1);
    reset_config_context();

    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
    cfg[SCRATCH_SEC0_val_ADDR32]             = block_increment;
    cfg[SCRATCH_SEC1_val_ADDR32]             = inner_increment;

    semaphore_post(semaphore::UNPACK_SYNC);

    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    const std::uint32_t kt_pairs = kt_dim / 2;
    if (kt_pairs > 0)
    {
        TT_MOP(0, kt_pairs - 1, 0);
    }
    if (kt_dim % 2 != 0)
    {
        // One k-row, as the MOP's first template slot plus its second: first_half from the buffer start
        // (block_increment reuses), then second_half from the buffer end (ending on the inner_increment reuse).
        // With nt_dim == 1 the k-row is the inner_increment reuse alone, like the MOP's B-mode slot.
        const std::uint32_t first_half_tiles  = nt_dim == 1 ? 0 : (nt_dim + 1) / 2;
        const std::uint32_t second_half_tiles = nt_dim == 1 ? 1 : nt_dim / 2;
        if (first_half_tiles > 0)
        {
            lltt::replay(0, first_half_tiles * SDPA_CUSTOM_MM_REUSE_BLOCK_LEN);
        }
        const std::uint32_t second_half_instruction_count = second_half_tiles * SDPA_CUSTOM_MM_REUSE_BLOCK_LEN;
        lltt::replay(SDPA_CUSTOM_MM_REUSE_REPLAY_BUF_LEN - second_half_instruction_count, second_half_instruction_count);
    }

    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}
