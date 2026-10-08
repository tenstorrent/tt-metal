// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"

namespace moe_fused_swiglu::compute {

struct MatmulShape {
    uint32_t m_subblocks;
    uint32_t n_subblocks;
    uint32_t subblock_h;
    uint32_t subblock_w;
    uint32_t k_tiles;
    uint32_t k_blocks;
    uint32_t last_in1_subblock_w_valid = 0;
    bool wait_in0_per_m_subblock = true;

    static constexpr MatmulShape of(
        uint32_t m_subblocks,
        uint32_t n_subblocks,
        uint32_t subblock_h,
        uint32_t subblock_w,
        uint32_t k_tiles,
        uint32_t k_blocks) {
        return {m_subblocks, n_subblocks, subblock_h, subblock_w, k_tiles, k_blocks};
    }
};

enum class MatmulTarget { Interm, Out };

struct NoPreKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

struct NoIn1Offset {
    ALWI uint32_t operator()(uint32_t) const { return 0; }
};

struct FullKSteps {
    ALWI uint32_t operator()(uint32_t, uint32_t k_tiles) const { return k_tiles; }
};

ALWI void pack_row_strided(uint32_t target_cb, uint32_t column, uint32_t row_width, uint32_t height, uint32_t width) {
    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t col = 0; col < width; ++col) {
            pack_tile<true>(row * width + col, target_cb, row * row_width + column + col);
        }
    }
}

template <
    bool init_matmul,
    bool retain_in0,
    bool retain_in1,
    MatmulTarget target,
    typename KSteps,
    typename PreK = NoPreKBlock,
    typename In1Offset = NoIn1Offset>
ALWI void matmul_row_major(
    CircularBuffer& in0,
    CircularBuffer& in1,
    CircularBuffer& out,
    CircularBuffer& interm,
    const MatmulShape& shape,
    uint32_t in1_width,
    uint32_t out_row_width,
    KSteps k_steps,
    PreK pre_k = {},
    In1Offset in1_offset = {},
    uint32_t out_column_offset = 0) {
    const uint32_t in0_cb = in0.get_cb_id();
    const uint32_t in1_cb = in1.get_cb_id();
    const uint32_t out_cb = out.get_cb_id();
    const uint32_t interm_cb = interm.get_cb_id();
    const uint32_t in0_subblock_tiles = shape.subblock_h * shape.k_tiles;
    const uint32_t in0_block_tiles = shape.m_subblocks * in0_subblock_tiles;
    const uint32_t in1_block_tiles = in1_width * shape.k_tiles;
    const uint32_t row_group_tiles = shape.subblock_h * out_row_width;

    if constexpr (init_matmul) {
        matmul_block_init(in0_cb, in1_cb, false, shape.subblock_w, shape.subblock_h, shape.k_tiles);
    }

    bool reload_partials = false;
    for (uint32_t k_block = 0; k_block < shape.k_blocks; ++k_block) {
        const bool is_last = k_block + 1 == shape.k_blocks;
        pre_k(k_block, shape.k_blocks, is_last);
        const uint32_t inner_steps = k_steps(k_block, shape.k_tiles);
        if constexpr (!retain_in0) {
            in0.wait_front(in0_block_tiles);
        } else if (!shape.wait_in0_per_m_subblock) {
            in0.wait_front(in0_block_tiles);
        }
        if constexpr (!retain_in1) {
            in1.wait_front(in1_block_tiles);
        }
        if constexpr (target == MatmulTarget::Out) {
            if (reload_partials) {
                UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
                UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
            }
        }

        for (uint32_t m_subblock = 0; m_subblock < shape.m_subblocks; ++m_subblock) {
            if constexpr (retain_in0) {
                if (shape.wait_in0_per_m_subblock) {
                    in0.wait_front((m_subblock + 1) * in0_subblock_tiles);
                }
            }
            uint32_t in1_index = in1_offset(k_block);
            for (uint32_t n_subblock = 0; n_subblock < shape.n_subblocks; ++n_subblock) {
                tile_regs_acquire();
                const uint32_t n_width = (shape.last_in1_subblock_w_valid != 0 && n_subblock + 1 == shape.n_subblocks)
                                             ? shape.last_in1_subblock_w_valid
                                             : shape.subblock_w;
                if (reload_partials) {
                    copy_tile_to_dst_init_short_with_dt(in1_cb, interm_cb);
                    const uint32_t source_base = m_subblock * row_group_tiles + n_subblock * shape.subblock_w;
                    for (uint32_t row = 0; row < shape.subblock_h; ++row) {
                        copy_block_matmul_partials(
                            interm_cb, source_base + row * out_row_width, row * shape.subblock_w, shape.subblock_w);
                    }
                    reconfig_data_format(in1_cb, in0_cb);
                    PACK((pack_reconfig_data_format(interm_cb)));
                    matmul_block_init(in0_cb, in1_cb, false, shape.subblock_w, shape.subblock_h, shape.k_tiles);
                }

                uint32_t in0_index = m_subblock * in0_subblock_tiles;
                for (uint32_t step = 0; step < inner_steps; ++step) {
                    ckernel::matmul_block(
                        in0_cb, in1_cb, in0_index, in1_index, 0, false, n_width, shape.subblock_h, shape.k_tiles);
                    ++in0_index;
                    in1_index += in1_width;
                }

                const uint32_t column =
                    m_subblock * row_group_tiles + n_subblock * shape.subblock_w + out_column_offset;
                if (is_last) {
                    tile_regs_commit();
                    tile_regs_wait();
                    const uint32_t target_cb = target == MatmulTarget::Interm ? interm_cb : out_cb;
                    PACK((pack_reconfig_data_format(target_cb)));
                    PACK((llk_pack_reconfig_l1_acc(target == MatmulTarget::Interm ? (k_block == 0 ? 0 : 1) : 0)));
                    pack_row_strided(target_cb, column, out_row_width, shape.subblock_h, shape.subblock_w);
                    tile_regs_release();
                } else {
                    tile_regs_commit();
                    tile_regs_wait();
                    PACK((pack_reconfig_data_format(interm_cb)));
                    PACK((llk_pack_reconfig_l1_acc(k_block == 0 ? 0 : 1)));
                    pack_row_strided(interm_cb, column, out_row_width, shape.subblock_h, shape.subblock_w);
                    tile_regs_release();
                }
                in1_index = in1_offset(k_block) + (n_subblock + 1) * shape.subblock_w;
            }
        }

        if constexpr (target == MatmulTarget::Out) {
            reload_partials = k_block + 2 == shape.k_blocks;
            if (reload_partials) {
                PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
            }
        }
        if constexpr (!retain_in0) {
            in0.pop_front(in0_block_tiles);
        }
        if constexpr (!retain_in1) {
            in1.pop_front(in1_block_tiles);
        }
    }
}

ALWI void add_silu_elementwise(
    CircularBuffer& partials, CircularBuffer& bias, CircularBuffer& out, uint32_t tiles, uint32_t bias_offset) {
    const uint32_t partials_cb = partials.get_cb_id();
    const uint32_t bias_cb = bias.get_cb_id();
    const uint32_t out_cb = out.get_cb_id();
    reconfig_data_format_srca(partials_cb);
    reconfig_data_format_srcb(bias_cb);
    pack_reconfig_data_format(out_cb);
    add_tiles_init(partials_cb, bias_cb);
    partials.wait_front(tiles);
    out.reserve_back(tiles);
    tile_regs_acquire();
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        add_tiles(partials_cb, bias_cb, tile, bias_offset + tile, tile);
    }
    tile_regs_commit();
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        silu_tile_pack(tile);
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        pack_tile(tile, out_cb);
    }
    tile_regs_release();
    partials.pop_front(tiles);
    out.push_back(tiles);
}

template <uint32_t width_tiles, uint32_t input_cb, uint32_t output_cb>
ALWI void tilize_row(uint32_t input_pages, uint32_t output_tile_offset) {
    DataflowBuffer input(input_cb);
    input.wait_front(input_pages);
    reconfig_data_format_srca(input_cb);
    PACK((pack_reconfig_data_format(output_cb)));
    fast_tilize_init(input_cb, width_tiles, output_cb);
    fast_tilize_block(input_cb, width_tiles, output_cb, 0, output_tile_offset);
    fast_tilize_uninit(input_cb, output_cb, width_tiles);
    input.pop_front(input_pages);
}

}  // namespace moe_fused_swiglu::compute
