// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Every core: for each expert in the schedule's order, takes the expert's weight slice once and multiplies this
// core's blocks of that expert's region by it, waiting for the slice chunk by chunk on the first block. Each output
// subblock is one row of tiles: K in one block, accumulated in DST in the order ttnn.matmul's 2D kernel uses, so
// the two agree bit for bit.
// The activation runs on the pack thread, beside the math of the next subblock.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/dataflow_buffer.h"
#ifdef SFPU_ACTIVATION
#include "ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp"
#endif
#include "../dataflow/expert_schedule.hpp"

#if defined(SFPU_ACTIVATION) && (defined(TRISC_MATH) || defined(TRISC_PACK))
namespace ckernel::sfpu {

// GELU(x) = x * Phi(t), t = clamp(x, -kGeluPolyClip, kGeluPolyClip), with Phi(t) = 0.5 + t * p(t^2) and p a
// degree-4 minimax fit of the normal CDF weighted by |t|: |GELU error| <= 1.3e-3 for every x, 15 SFPU
// instructions a row. Activation parameter 2; meant for a bfloat8_b output, whose rounding is several times that
// error.
constexpr float kGeluPolyClip = 3.5f;
constexpr float kGeluPolyC0 = 3.9340987315e-01f;
constexpr float kGeluPolyC1 = -5.8935974044e-02f;
constexpr float kGeluPolyC2 = 6.4495455820e-03f;
constexpr float kGeluPolyC3 = -3.8477153700e-04f;
constexpr float kGeluPolyC4 = 9.3667710000e-06f;

inline void gelu_polynomial_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = kGeluPolyC2;
    sfpi::vConstFloatPrgm1 = kGeluPolyC3;
    sfpi::vConstFloatPrgm2 = kGeluPolyC4;
}

// One face, two rows at a time so each row's dependent steps fill the other's latency: one row at a time the chain
// ran 23 cycles a row, and on the pack thread it bounded w1 at HiFi2. The clamp is copysgn(min(|x|, c), x), equal to
// clamp(x, -c, c) for every finite x and both zeros, and x is read from DST again for the last product, which keeps
// both rows' state and C0, C1 in the 8 registers. The arithmetic is the one-row form's, bit for bit.
template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_gelu_polynomial() {
    const sfpi::vFloat c1 = kGeluPolyC1;
    const sfpi::vFloat c0 = kGeluPolyC0;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat ta = sfpi::symmetric_clamp(sfpi::dst_reg[0], kGeluPolyClip);
        sfpi::vFloat tb = sfpi::symmetric_clamp(sfpi::dst_reg[1], kGeluPolyClip);
        sfpi::vFloat ua = ta * ta;
        sfpi::vFloat ub = tb * tb;
        sfpi::vFloat pa = ua * sfpi::vConstFloatPrgm2 + sfpi::vConstFloatPrgm1;
        sfpi::vFloat pb = ub * sfpi::vConstFloatPrgm2 + sfpi::vConstFloatPrgm1;
        pa = ua * pa + sfpi::vConstFloatPrgm0;
        pb = ub * pb + sfpi::vConstFloatPrgm0;
        pa = ua * pa + c1;
        pb = ub * pb + c1;
        pa = ua * pa + c0;
        pb = ub * pb + c0;
        pa = ta * pa + 0.5f;
        pb = tb * pb + 0.5f;
        sfpi::vFloat ra = sfpi::vFloat(sfpi::dst_reg[0]) * pa;
        sfpi::vFloat rb = sfpi::vFloat(sfpi::dst_reg[1]) * pb;
        if constexpr (!is_fp32_dest_acc_en) {
            ra = sfpi::convert<sfpi::vFloat16b>(ra, sfpi::RoundMode::Nearest);
            rb = sfpi::convert<sfpi::vFloat16b>(rb, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = ra;
        sfpi::dst_reg[1] = rb;
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
#endif

#ifdef SFPU_ACTIVATION
// apply_activation_from_pack with the polynomial GELU.
FORCE_INLINE void apply_gelu_polynomial_from_pack(uint32_t out_subblock_num_tiles) {
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    // The subblock's tiles are consecutive in DST: one pass over all their faces.
    PACK((_llk_math_eltwise_sfpu_start_(0)));
    for (uint32_t face = 0; face < 4 * out_subblock_num_tiles; ++face) {
        PACK((ckernel::sfpu::calculate_gelu_polynomial<DST_ACCUM_MODE>()));
        PACK((_llk_math_eltwise_sfpu_inc_dst_face_addr_()));
    }
    PACK((_llk_math_eltwise_sfpu_done_()));
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
}
#endif

void kernel_main() {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t slot_n_tiles = get_named_compile_time_arg_val("slot_n_tiles");
    constexpr uint32_t subblock_w = get_named_compile_time_arg_val("subblock_w");
    constexpr bool row_major_output = get_named_compile_time_arg_val("row_major_output") == 1;
    constexpr bool transpose_weight = get_named_compile_time_arg_val("transpose_weight") == 1;
    constexpr uint32_t input_cb_id = get_named_compile_time_arg_val("input_cb");
    constexpr uint32_t weight_cb_id = get_named_compile_time_arg_val("weight_cb");
    constexpr uint32_t output_cb_id = get_named_compile_time_arg_val("output_cb");
    constexpr uint32_t schedule_cb_id = get_named_compile_time_arg_val("schedule_cb");
    // A bias is the weight slot's last row, multiplied by a tile of ones after the K loop.
    constexpr bool has_bias = get_named_compile_time_arg_val("has_bias") == 1;
    constexpr uint32_t ones_cb_id = get_named_compile_time_arg_val("ones_cb");
    constexpr uint32_t slot_tiles = (k_tiles + (has_bias ? 1 : 0)) * slot_n_tiles;
    // The slot is subblock-major: each subblock's K rows of subblock_w tiles, then its bias tiles.
    constexpr uint32_t subblock_base = (k_tiles + (has_bias ? 1 : 0)) * subblock_w;
    constexpr uint32_t k_step = subblock_w;
    constexpr bool subblock_outer = get_named_compile_time_arg_val("subblock_outer") == 1;
    static_assert(!subblock_outer || (num_experts == 1 && !row_major_output));
    constexpr bool tail_split = get_named_compile_time_arg_val("tail_split") == 1;
    // Blocks a subblock-outer group; 0: all of the lane's blocks in one group.
    constexpr uint32_t group_blocks = get_named_compile_time_arg_val("group_blocks");
    static_assert(!tail_split || subblock_outer);
    // The weight stream's pieces: an expert's last block pops each whole one once its K steps are issued.
    constexpr uint32_t chunk_tiles = get_named_compile_time_arg_val("chunk_tiles");

    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t num_subblocks = n_tiles / subblock_w;

    DataflowBuffer input_cb(input_cb_id);
    DataflowBuffer weight_cb(weight_cb_id);
    DataflowBuffer output_cb(output_cb_id);
    DataflowBuffer schedule_cb(schedule_cb_id);

#ifdef SFPU_ACTIVATION
    constexpr KernelActivation activation_type =
        static_cast<KernelActivation>(get_named_compile_time_arg_val("activation_type"));
    constexpr uint32_t activation_param0 = get_named_compile_time_arg_val("activation_param0");
    // Parameter 2 is the polynomial GELU of this file; 0 and 1 the stock tanh GELU's accurate and fast forms.
    constexpr bool gelu_polynomial = activation_param0 == 2;
    if constexpr (gelu_polynomial) {
        PACK(llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(ckernel::sfpu::gelu_polynomial_init));
    } else {
        ActivationInitHelper<activation_type, activation_param0>::init();
    }
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(input_cb_id, weight_cb_id, output_cb_id);
    matmul_block_init(input_cb_id, weight_cb_id, transpose_weight, subblock_w, 1, k_tiles);
    if constexpr (row_major_output) {
        pack_untilize_dest_init<subblock_w, slot_n_tiles>(output_cb_id);
    }

    if constexpr (has_bias) {
        DataflowBuffer ones_cb(ones_cb_id);
        ones_cb.wait_front(1);
    }
    schedule_cb.wait_front(1);
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        const uint32_t expert_blocks = read_tile_value(schedule_cb_id, 0, expert);
        if (expert_blocks == grouped_matmul::kSkipExpert) {
            continue;
        }
        // Tiles of the slot waited for so far: the first block takes each K step as soon as its tiles are in, the
        // slice arriving in chunks.
        // Slot positions count from the slot's start; the ring's read pointer is `released` tiles past it.
        uint32_t waited = 0;
        uint32_t released = 0;
        auto wait_weights = [&](uint32_t tiles) {
            if (tiles > waited) {
                waited = tiles;
                weight_cb.wait_front(waited - released);
            }
        };
        // Pops every whole piece below consumed, a slot position whose tiles have all been taken.
        auto release_to = [&](uint32_t consumed) {
            while (released + chunk_tiles <= consumed) {
                weight_cb.pop_front(chunk_tiles);
                released += chunk_tiles;
            }
        };
        // One output subblock of the input block at input_offset tiles into the input ring; with release, the
        // expert's last use of the slice.
        auto run_subblock = [&](uint32_t input_offset, uint32_t subblock, bool release) {
            tile_regs_acquire();
            uint32_t weight_index = subblock * subblock_base;
            for (uint32_t k = 0; k < k_tiles; ++k) {
                wait_weights(weight_index + subblock_w);
                matmul_block(
                    input_cb_id,
                    weight_cb_id,
                    input_offset + k,
                    weight_index - released,
                    0,
                    transpose_weight,
                    subblock_w,
                    1,
                    k_tiles);
                weight_index += k_step;
                if (release) {
                    release_to(weight_index);
                }
            }
            if constexpr (has_bias) {
                wait_weights(weight_index + subblock_w);
                // ones x the bias row: the bias added to every row, in DST's fp32, before the activation.
                matmul_block(
                    ones_cb_id, weight_cb_id, 0, weight_index - released, 0, transpose_weight, subblock_w, 1, k_tiles);
                if (release) {
                    release_to(weight_index + subblock_w);
                }
            }
            tile_regs_commit();
            if constexpr (!row_major_output) {
                output_cb.reserve_back(subblock_w);
            }
#ifdef SFPU_ACTIVATION
            if constexpr (gelu_polynomial) {
                apply_gelu_polynomial_from_pack(subblock_w);
            } else {
                apply_activation_from_pack<activation_type, activation_param0>(subblock_w);
            }
#else
            tile_regs_wait();
#endif
            if constexpr (row_major_output) {
                pack_untilize_dest<subblock_w, slot_n_tiles>(output_cb_id, 1, subblock);
            } else {
                pack_block(0, output_cb_id, subblock_w);
            }
            tile_regs_release();
            if constexpr (!row_major_output) {
                output_cb.push_back(subblock_w);
            }
        };

        if constexpr (subblock_outer) {
            // One expert whose blocks the input ring holds at once: each subblock of the weights serves every block
            // before the next subblock is needed, so the slice streams in under the math. Outputs come subblock by
            // subblock, block by block within one (input_stream.cpp writes them in that order).
            // With the tail split, words 1 and 2 of the schedule are the subblocks [begin, end) the lane computes of
            // one more block, the last held. Groups of group_blocks blocks, the tail block after the last group's.
            const uint32_t tail_begin = tail_split ? read_tile_value(schedule_cb_id, 0, 1) : grouped_matmul::kNoUnit;
            const uint32_t tail_end = tail_split ? read_tile_value(schedule_cb_id, 0, 2) : grouped_matmul::kNoUnit;
            const uint32_t group = group_blocks == 0 ? expert_blocks : group_blocks;
            const uint32_t groups = expert_blocks == 0 ? (tail_begin != grouped_matmul::kNoUnit ? 1 : 0)
                                                       : (expert_blocks + group - 1) / group;
            for (uint32_t index = 0; index < groups; ++index) {
                const uint32_t first = index * group;
                const uint32_t count = expert_blocks - first < group ? expert_blocks - first : group;
                const bool tail_here = index + 1 == groups && tail_begin != grouped_matmul::kNoUnit;
                const uint32_t held = count + (tail_here ? 1 : 0);
                for (uint32_t subblock = 0; subblock < num_subblocks; ++subblock) {
                    for (uint32_t block = 0; block < count; ++block) {
                        if (subblock == 0) {
                            input_cb.wait_front((block + 1) * k_tiles);
                        }
                        run_subblock(block * k_tiles, subblock, false);
                    }
                    if (tail_here && subblock >= tail_begin && subblock < tail_end) {
                        input_cb.wait_front(held * k_tiles);
                        run_subblock(count * k_tiles, subblock, false);
                    }
                }
                input_cb.wait_front(held * k_tiles);
                input_cb.pop_front(held * k_tiles);
            }
        } else {
            for (uint32_t block = 0; block < expert_blocks; ++block) {
                input_cb.wait_front(k_tiles);
                if constexpr (row_major_output) {
                    output_cb.reserve_back(n_tiles);
                }
                for (uint32_t subblock = 0; subblock < num_subblocks; ++subblock) {
                    run_subblock(0, subblock, block + 1 == expert_blocks);
                }
                if constexpr (row_major_output) {
                    output_cb.push_back(n_tiles);
                }
                input_cb.pop_front(k_tiles);
            }
        }
        wait_weights(slot_tiles);
        weight_cb.pop_front(slot_tiles - released);
    }
    schedule_cb.pop_front(1);
}
