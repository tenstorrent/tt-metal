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

// Profile-only wait/MAC accounting. Compiled in only when the including kernel defines
// MOE_FUSED_SWIGLU_PROF_ACC (moe_fused_swiglu_compute.cpp does under MOE_FUSED_SWIGLU_STAGE_PROFILE).
// Every CB wait inside a matmul or activation zone is timed off the wall clock and summed per slot on
// each TRISC (cb_wait_front spins on UNPACK, cb_reserve_back on PACK; the other threads time a no-op),
// and every matmul_block adds its tile-MAC count, so the host subtracts exact wait time from the zone
// time instead of estimating it. A per-wait zone would not fit the 125-record budget; the kernel emits
// the sums as three DeviceTimestampedData records per TRISC at the end.
#if defined(MOE_FUSED_SWIGLU_PROF_ACC)
namespace moe_fused_swiglu::prof {
constexpr uint32_t MM = 0;
constexpr uint32_t ACT = 1;
inline uint32_t wait_cyc[2];
inline uint32_t tmacs;
ALWI uint32_t clk() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
}  // namespace moe_fused_swiglu::prof
#define MOE_PROF_WAIT(slot, ...)                                                            \
    do {                                                                                    \
        const uint32_t prof_t0_ = moe_fused_swiglu::prof::clk();                            \
        __VA_ARGS__;                                                                        \
        moe_fused_swiglu::prof::wait_cyc[slot] += moe_fused_swiglu::prof::clk() - prof_t0_; \
    } while (0)
#define MOE_PROF_TMACS(n) (moe_fused_swiglu::prof::tmacs += (n))
#else
#define MOE_PROF_WAIT(slot, ...) \
    do {                         \
        __VA_ARGS__;             \
    } while (0)
#define MOE_PROF_TMACS(n) ((void)0)
#endif

// Op-local fast SiLU. The stock calculate_silu always runs exp_21f plus a Newton-refined reciprocal,
// whatever APPROX says, and changing it globally would move every op that defaults to approx mode.
// Here the sigmoid's reciprocal is the raw SFPARECIP estimate (~7 bits); the result is rounded to
// bf16 anyway, so the Newton step buys almost nothing for this op's PCC.
//   MOE_SILU_FAST == 1: exp_21f(-x), raw reciprocal.
//   MOE_SILU_FAST == 2: Schraudolph exp (exponent-field linear 2^f, bias-tuned), raw reciprocal.
#ifndef MOE_SILU_FAST
#define MOE_SILU_FAST 2
#endif
#if defined(TRISC_PACK) || defined(TRISC_MATH)
namespace ckernel::sfpu {
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat moe_silu_fast_value(sfpi::vFloat x) {
    sfpi::vFloat e;
#if MOE_SILU_FAST == 1
    e = _sfpu_exp_21f_bf16_<true>(-x);
#else
    // 2^(t) with t = -x/ln2: build the float whose bits are (t + 127) * 2^23. The mantissa is the
    // linear 1 + frac, off by up to 6%; the -0.0579 bias recentres that to about +-3%.
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat t = x * -ONE_LN2 + (127.f - 0.0579f);
    t = sfpi::clamp(t, 0.0f, 255.0f);
    e = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(t));
#endif
    return x * sfpi::approx_recip(e + 1.0f);
}

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_moe_silu_fast() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat result = moe_silu_fast_value<is_fp32_dest_acc_en>(sfpi::dst_reg[0]);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// silu(gate) * up with both operands already in DEST: the plain-SiLU twin of situ_glu / swiglu_oai,
// so the fused fold_binary_act_blocked path serves SiLU too and the bf16 slice CBs drop out.
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_moe_silu_glu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
        sfpi::vFloat result = moe_silu_fast_value<is_fp32_dest_acc_en>(gate) * up;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

inline void moe_silu_glu_init() {}
}  // namespace ckernel::sfpu
#endif

namespace ckernel {
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void moe_silu_fast_tile_pack(uint32_t idst) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_moe_silu_fast,
        (is_fp32_dest_acc_en, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC));
}

ALWI void moe_silu_glu_tile(uint32_t gate, uint32_t up, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_moe_silu_glu,
        (DST_ACCUM_MODE, 8 /* ITERATIONS */),
        gate,
        up,
        out,
        VectorMode::RC)));
}

ALWI void moe_silu_glu_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(sfpu::moe_silu_glu_init)));
}
}  // namespace ckernel

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
        MOE_PROF_WAIT(0, pre_k(k_block, shape.k_blocks, is_last));
        const uint32_t inner_steps = k_steps(k_block, shape.k_tiles);
        if constexpr (!retain_in0) {
            MOE_PROF_WAIT(0, in0.wait_front(in0_block_tiles));
        } else if (!shape.wait_in0_per_m_subblock) {
            MOE_PROF_WAIT(0, in0.wait_front(in0_block_tiles));
        }
        if constexpr (!retain_in1) {
            MOE_PROF_WAIT(0, in1.wait_front(in1_block_tiles));
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
                    MOE_PROF_WAIT(0, in0.wait_front((m_subblock + 1) * in0_subblock_tiles));
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
                MOE_PROF_TMACS(inner_steps * n_width * shape.subblock_h);

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
    MOE_PROF_WAIT(1, partials.wait_front(tiles));
    MOE_PROF_WAIT(1, out.reserve_back(tiles));
    tile_regs_acquire();
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        add_tiles(partials_cb, bias_cb, tile, bias_offset + tile, tile);
    }
    tile_regs_commit();
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        moe_silu_fast_tile_pack(tile);
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
