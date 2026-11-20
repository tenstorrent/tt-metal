// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/compute/gate_factor.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/compute/gate_matmul.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_e = get_compile_time_arg_val(0);
constexpr uint32_t cb_b = get_compile_time_arg_val(1);
constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
constexpr uint32_t cb_eout = get_compile_time_arg_val(3);
constexpr uint32_t kt = get_compile_time_arg_val(4);
// The same landed entry state as cb_ein, unpacked straight to DST (FP32, exact).
constexpr uint32_t cb_ein_exact = get_compile_time_arg_val(5);
// One BF16 identity tile: B enters the sum as I @ B.
constexpr uint32_t cb_identity = get_compile_time_arg_val(6);
// The final carry (see the reader): FP32 E and B of the final transition, and the relayed final state P in its matmul
// and exact views.
constexpr uint32_t cb_fe = get_compile_time_arg_val(7);
constexpr uint32_t cb_fb = get_compile_time_arg_val(8);
constexpr uint32_t cb_p = get_compile_time_arg_val(9);
constexpr uint32_t cb_p_exact = get_compile_time_arg_val(10);
// The reader's {holder, split, chunks, group chunks}.
constexpr uint32_t cb_ctl = get_compile_time_arg_val(11);
// The entry correction: the seeds, the chunk terms, their differences and the chunk outputs (FP32).
constexpr uint32_t cb_seed = get_compile_time_arg_val(12);
constexpr uint32_t cb_tail = get_compile_time_arg_val(13);
constexpr uint32_t cb_terms = get_compile_time_arg_val(14);
constexpr uint32_t cb_diff = get_compile_time_arg_val(15);
constexpr uint32_t cb_out = get_compile_time_arg_val(16);
constexpr uint32_t core_chunks = get_compile_time_arg_val(17);
constexpr bool streamed = get_compile_time_arg_val(18);
// Each core's index among the heads' cores (head * cores_per_head + its place in the head) is in the common args at
// core_map, by the core's place in the compute grid (y * grid_x + x).
constexpr uint32_t cores_per_head = get_compile_time_arg_val(19);
constexpr uint32_t grid_x = get_compile_time_arg_val(20);
constexpr uint32_t core_map = get_compile_time_arg_val(21);
// The gated output norm of each chunk row x: x / sqrt(mean(x^2) + epsilon) * g with g = sigmoid(gate) * weight. The
// squares, the mean's scaler, 1 / rms, the gate rows (FP32, all of the core's chunks), the weight (row 0 of its tiles)
// and broadcast down every row, each chunk's g (all in place, one streamed), and the result.
constexpr uint32_t cb_square = get_compile_time_arg_val(22);
constexpr uint32_t cb_scaler = get_compile_time_arg_val(23);
constexpr uint32_t cb_inv_rms = get_compile_time_arg_val(24);
constexpr uint32_t cb_gate = get_compile_time_arg_val(25);
constexpr uint32_t cb_weight = get_compile_time_arg_val(26);
constexpr uint32_t cb_weight_rows = get_compile_time_arg_val(27);
constexpr uint32_t cb_norm = get_compile_time_arg_val(28);
constexpr uint32_t cb_gate_factor = get_compile_time_arg_val(29);
constexpr uint32_t epsilon_bits = get_compile_time_arg_val(30);
// An FP32 output takes the FP32-accurate sigmoid; a BF16 one needs only the BF16-accurate exp and one reciprocal step.
constexpr bool fp32_output = get_compile_time_arg_val(31);
// The output gate x @ W of the core's chunks, on every core of a head but its first (a helper forms that one's): K
// blocks of x and of W, at the input projection's fidelity.
constexpr uint32_t cb_gin0 = get_compile_time_arg_val(32);
constexpr uint32_t cb_gin1 = get_compile_time_arg_val(33);
constexpr uint32_t gate_kb = get_compile_time_arg_val(34);
constexpr uint32_t gate_blocks = get_compile_time_arg_val(35);
constexpr auto gate_fidelity = static_cast<ckernel::MathFidelity>(get_compile_time_arg_val(36));
// The partial sums of the pair's other head, and the pair's core's of this core's head (the output gate's heads pair
// up, h with h + heads / 2, each core forming both over its head's half of K).
constexpr uint32_t cb_gate_pair = get_compile_time_arg_val(37);
constexpr uint32_t cb_gate_partial = get_compile_time_arg_val(38);
constexpr uint32_t gate_heads = get_compile_time_arg_val(39);
// 2 where the heads pair up, else 1 (each core its head over all of K).
constexpr uint32_t gate_ways = get_compile_time_arg_val(40);
constexpr uint32_t state_tiles = kt * kt;
constexpr uint32_t dst_tiles =
    ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
// A tile column of the next state is formed in one DST pass (the op checks K against the DST).
static_assert(kt <= dst_tiles);
// A DST pass of the carry or the norm takes `half_dst` tiles, with as many operand tiles beside them.
constexpr uint32_t half_dst = kda::largest_divisor_at_most(kt, dst_tiles / 2);
// The terms buffer holds a chunk row of F(0) at tile row, and of F(I) at fi_offset further: in place, the core's whole
// shard (core_chunks rows of F(0), then of F(I)); streamed, one chunk at a time.
constexpr uint32_t fi_offset = (streamed ? 1 : core_chunks) * kt;
constexpr uint32_t terms_tiles = 2 * fi_offset;

// The relay's hop on the head core: the next entry state S_out = A S_in + B for one head, from the complement form
// E = A - I: S_out = S_in + E S_in + B, summed in the FP32 DST. The exact copy of S_in unpacks to DST losslessly; E
// S_in accumulates on top at the affine prefix's fidelity, so the matmul's source-register copy of S_in reaches only
// the small E S_in term; B accumulates as I @ B, exact at HiFi2 and above (a matmul's in1 goes to srcA, whose HiFi2
// phases cover a BF16 mantissa, and 1.0 is exact in srcB). Per output tile column (column-major state): column j needs
// only column j of S_in, so it is computed as soon as that column lands.
void relay_hop() {
    pack_reconfig_data_format(cb_eout);
    cb_wait_front(cb_e, state_tiles);
    cb_wait_front(cb_b, state_tiles);
    cb_wait_front(cb_identity, 1);
    for (uint32_t j = 0; j < kt; ++j) {
        cb_reserve_back(cb_eout, kt);
        cb_wait_front(cb_ein, (j + 1) * kt);
        cb_wait_front(cb_ein_exact, (j + 1) * kt);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_ein_exact);
        copy_init(cb_ein_exact);
        for (uint32_t r = 0; r < kt; ++r) {
            copy_tile(cb_ein_exact, j * kt + r, r);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_e, cb_ein);
        matmul_block_init(cb_e, cb_ein, false, 1, kt, kt);
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(cb_e, cb_ein, k, j * kt + k, 0, false, 1, kt, kt);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_identity, cb_b);
        matmul_block_init(cb_identity, cb_b, false, 1, 1, 1);
        for (uint32_t r = 0; r < kt; ++r) {
            matmul_block(cb_identity, cb_b, 0, r * kt + j, r, false, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t r = 0; r < kt; ++r) {
            pack_tile(r, cb_eout);
        }
        tile_regs_release();
        cb_push_back(cb_eout, kt);
    }
    cb_pop_front(cb_e, state_tiles);
    cb_pop_front(cb_ein, state_tiles);
    cb_pop_front(cb_ein_exact, state_tiles);
    cb_pop_front(cb_b, state_tiles);
    cb_pop_front(cb_identity, 1);
}

// The final carry on the first rank: P + (E P + B) per column with the final transition when the interval ends in a
// separated tail, else P itself. B and P unpack to DST exactly and are added on the SFPU, E P accumulates on top.
void carry_pass(bool split) {
    pack_reconfig_data_format(cb_eout);
    if (split) {
        cb_wait_front(cb_fe, state_tiles);
        cb_wait_front(cb_fb, state_tiles);
    }
    for (uint32_t j = 0; j < kt; ++j) {
        cb_reserve_back(cb_eout, kt);
        cb_wait_front(cb_p, (j + 1) * kt);
        cb_wait_front(cb_p_exact, (j + 1) * kt);
        for (uint32_t r0 = 0; r0 < kt; r0 += half_dst) {
            tile_regs_acquire();
            if (split) {
                reconfig_data_format_srca(cb_fb);
                copy_init(cb_fb);
                for (uint32_t i = 0; i < half_dst; ++i) {
                    copy_tile(cb_fb, (r0 + i) * kt + j, i);
                }
                reconfig_data_format_srca(cb_p_exact);
                copy_init(cb_p_exact);
                for (uint32_t i = 0; i < half_dst; ++i) {
                    copy_tile(cb_p_exact, j * kt + r0 + i, half_dst + i);
                }
                add_binary_tile_init();
                for (uint32_t i = 0; i < half_dst; ++i) {
                    add_binary_tile(i, half_dst + i, i);
                }
                reconfig_data_format<SrcOrder::Reverse>(cb_fe, cb_p);
                matmul_block_init(cb_fe, cb_p, false, 1, half_dst, kt);
                for (uint32_t k = 0; k < kt; ++k) {
                    matmul_block(cb_fe, cb_p, r0 * kt + k, j * kt + k, 0, false, 1, half_dst, kt);
                }
            } else {
                reconfig_data_format_srca(cb_p_exact);
                copy_init(cb_p_exact);
                for (uint32_t i = 0; i < half_dst; ++i) {
                    copy_tile(cb_p_exact, j * kt + r0 + i, i);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < half_dst; ++i) {
                pack_tile(i, cb_eout);
            }
            tile_regs_release();
        }
        cb_push_back(cb_eout, kt);
    }
    cb_pop_front(cb_p, state_tiles);
    cb_pop_front(cb_p_exact, state_tiles);
    if (split) {
        cb_pop_front(cb_fe, state_tiles);
        cb_pop_front(cb_fb, state_tiles);
    }
}

// FI - F0 for tile rows [first, first + tiles) of the terms buffer, packed to the same rows of the differences.
void subtract(uint32_t first, uint32_t tiles) {
    DataflowBuffer terms(cb_terms);
    DataflowBuffer diff(cb_diff);
    constexpr auto term_rows = ckl::input(
        cb_terms,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset);
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(tiles).block_size(kt),
        ckl::BinaryFpu<ckl::BinaryFpuOp::Sub, term_rows, term_rows>{terms, terms, fi_offset + first, first},
        ckl::PackTile<ckl::output(cb_diff, ckl::ReservePolicy::None, ckl::PushPolicy::None)>{diff});
}

void broadcast_weight() { kda_gate::broadcast_weight<cb_weight, cb_weight_rows, kt>(); }

void gate_factors(uint32_t chunks) {
    kda_gate::gate_factors<cb_gate, cb_weight_rows, cb_gate_factor, kt, fp32_output>(chunks);
}

// The gated norm of the chunk row in cb_out, with the front g of cb_gate_factor, into cb_norm.
void normalize() {
    DataflowBuffer out(cb_out);
    DataflowBuffer square(cb_square);
    DataflowBuffer scaler(cb_scaler);
    DataflowBuffer inv_rms(cb_inv_rms);
    ckl::square<
        ckl::input(cb_out, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
        ckl::output(cb_square, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
        ckl::IterationShape::tiles(kt).block_size(kt), out, square);
    // The row means land in column 0, all the column broadcast reads: epsilon and rsqrt on that column only.
    ckl::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        cb_square,
        cb_scaler,
        cb_inv_rms,
        ckl::ReduceInputPolicy::BulkWaitBulkPop>(
        square,
        scaler,
        inv_rms,
        ckl::ReduceInputBlockShape::row(kt),
        ckl::ReduceInputMemoryLayout::contiguous(),
        ckl::NoAccumulation{},
        [](uint32_t dst) {
            binop_with_scalar_tile_init();
            add_unary_tile(dst, epsilon_bits, VectorMode::C);
            rsqrt_tile_init();
            rsqrt_tile(dst, VectorMode::C);
        });
    // Per pass: x / rms on the FPU, then times g beside it on the SFPU, in FP32.
    DataflowBuffer gate_factor(cb_gate_factor);
    DataflowBuffer norm(cb_norm);
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(kt).block_size(kt),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Mul,
            ckl::input(cb_out, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
            ckl::input(cb_inv_rms, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd)>{
            out, inv_rms},
        ckl::CopyTile<
            ckl::input(cb_gate_factor, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
            ckl::Dst::D1>{gate_factor},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(cb_norm, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{norm});
}

// Chunk outputs x = F0_c + (FI_c - F0_c) @ seed for this core's chunks [begin, end), a row at a time once the whole
// seed has landed: F0's row is copied into the FP32 DST (through srcA, so at TF32: cb_terms also feeds the FPU's
// differences, and TF32 is finer than the BF16 the model writes) and the K-sum of each column accumulates on top of it,
// then the row is normalized. In place the differences are already formed; streamed, each chunk's is formed just before
// its products, and its g with it.
void correct(uint32_t begin, uint32_t end, uint32_t seed) {
    if (begin != end) {
        cb_wait_front(seed, state_tiles);
    }
    for (uint32_t c = begin; c < end; ++c) {
        if constexpr (streamed) {
            cb_wait_front(cb_terms, terms_tiles);
            cb_reserve_back(cb_diff, kt);
            subtract(0, kt);
            cb_push_back(cb_diff, kt);
            cb_wait_front(cb_diff, kt);
            gate_factors(1);
        }
        const uint32_t row = streamed ? 0 : c * kt;
        cb_reserve_back(cb_out, kt);
        pack_reconfig_data_format(cb_out);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_terms);
        copy_init(cb_terms);
        for (uint32_t j = 0; j < kt; ++j) {
            copy_tile(cb_terms, row + j, j);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_diff, seed);
        matmul_block_init(cb_diff, seed, false, 1, 1, kt);
        for (uint32_t j = 0; j < kt; ++j) {
            for (uint32_t k = 0; k < kt; ++k) {
                matmul_block(cb_diff, seed, row + k, j * kt + k, j, false, 1, 1, kt);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < kt; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, kt);
        cb_wait_front(cb_out, kt);
        normalize();
        cb_pop_front(cb_out, kt);
        if constexpr (streamed) {
            cb_pop_front(cb_terms, terms_tiles);
            cb_pop_front(cb_diff, kt);
        }
    }
}

// Every core of a head corrects its chunks once their gate rows are formed: the head's other cores form them first,
// the head's first core runs the relay's hop instead and, on the first rank, the final carry (from P, which lands after
// the lap) before its chunks, whose rows its helper forms meanwhile (in place, their factors too).
void kernel_main() {
    const uint32_t core = get_common_arg_val<uint32_t>(core_map + get_absolute_logical_y() * grid_x + get_absolute_logical_x());
    const bool relay_head = core % cores_per_head == 0;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_terms, cb_seed, cb_out);
    cb_wait_front(cb_ctl, 1);
    const uint32_t holder = read_tile_value(cb_ctl, 0, 0);
    const uint32_t split = read_tile_value(cb_ctl, 0, 1);
    const uint32_t chunks = read_tile_value(cb_ctl, 0, 2);
    const uint32_t group_chunks = read_tile_value(cb_ctl, 0, 3);
    cb_pop_front(cb_ctl, 1);

    // In place, every chunk's difference: on a head core while its entry state is on its way, outside the first rank
    // (whose relay starts at once).
    const auto differences = [&]() {
        if constexpr (!streamed) {
            if (chunks != 0) {
                cb_wait_front(cb_terms, terms_tiles);
                cb_reserve_back(cb_diff, chunks * kt);
                subtract(0, chunks * kt);
                cb_push_back(cb_diff, chunks * kt);
                cb_wait_front(cb_diff, chunks * kt);
            }
        }
    };
    // The cores that form their chunks' factors: all but the head cores, whose helpers form them in place.
    const bool own_factors = streamed || !relay_head;
    if (relay_head) {
        if (!holder) {
            differences();
        }
        relay_hop();
        if (holder) {
            carry_pass(split);
            differences();
        }
    } else {
        const uint32_t head = core / cores_per_head;
        kda_gate::matmul<gate_fidelity>(
            cb_gin0,
            cb_gin1,
            cb_gate,
            cb_gate_pair,
            gate_ways == 1 || head % gate_heads < gate_heads / 2,
            gate_ways,
            core_chunks,
            kt,
            gate_kb,
            gate_blocks);
        if constexpr (gate_ways == 2) {
            kda_gate::add_pair(cb_gate_partial, cb_gate, core_chunks * kt);
        }
    }
    // The norm's weight rows and, in place, every chunk's g.
    if (own_factors && chunks != 0) {
        broadcast_weight();
    }
    if (!streamed && !relay_head) {
        gate_factors(chunks);
        differences();
    }
    if (chunks != 0) {
        cb_wait_front(cb_scaler, 1);
    }
    correct(0, group_chunks, cb_seed);
    correct(group_chunks, chunks, cb_tail);
    if (group_chunks != 0) {
        cb_pop_front(cb_seed, state_tiles);
    }
    if (chunks != group_chunks) {
        cb_pop_front(cb_tail, state_tiles);
    }
    if (!streamed && chunks != 0) {
        cb_pop_front(cb_terms, terms_tiles);
        cb_pop_front(cb_diff, chunks * kt);
    }
    if (own_factors && chunks != 0) {
        cb_pop_front(cb_weight_rows, kt);
    }
    if (chunks != 0) {
        cb_pop_front(cb_scaler, 1);
    }
}
