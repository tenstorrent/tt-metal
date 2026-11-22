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
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/compute/gate_matmul.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/correction_core.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_ctl = get_compile_time_arg_val(0);  // the reader's chunk counts
constexpr uint32_t kt = get_compile_time_arg_val(1);
// The entry correction: the seeds, the chunk terms, their differences and the chunk outputs (FP32).
constexpr uint32_t cb_seed = get_compile_time_arg_val(2);
constexpr uint32_t cb_tail = get_compile_time_arg_val(3);
constexpr uint32_t cb_terms = get_compile_time_arg_val(4);
constexpr uint32_t cb_diff = get_compile_time_arg_val(5);
constexpr uint32_t cb_out = get_compile_time_arg_val(6);
constexpr uint32_t core_chunks = get_compile_time_arg_val(7);
constexpr bool streamed = get_compile_time_arg_val(8);
// The cores' layout (correction_core.hpp).
constexpr uint32_t cores_per_head = get_compile_time_arg_val(9);
constexpr uint32_t heads_per_column = get_compile_time_arg_val(10);
constexpr uint32_t first_row = get_compile_time_arg_val(11);
// The gated output norm of each chunk row x: x / sqrt(mean(x^2) + epsilon) * g with g = sigmoid(gate) * weight. The
// squares, the mean's scaler, 1 / rms, the gate rows (FP32, all of the core's chunks), the weight (row 0 of its tiles)
// and broadcast down every row, each chunk's g (all in place, one streamed), and the result.
constexpr uint32_t cb_square = get_compile_time_arg_val(12);
constexpr uint32_t cb_scaler = get_compile_time_arg_val(13);
constexpr uint32_t cb_inv_rms = get_compile_time_arg_val(14);
constexpr uint32_t cb_gate = get_compile_time_arg_val(15);
constexpr uint32_t cb_weight = get_compile_time_arg_val(16);
constexpr uint32_t cb_weight_rows = get_compile_time_arg_val(17);
constexpr uint32_t cb_norm = get_compile_time_arg_val(18);
constexpr uint32_t cb_gate_factor = get_compile_time_arg_val(19);
constexpr uint32_t epsilon_bits = get_compile_time_arg_val(20);
// An FP32 output takes the FP32-accurate sigmoid; a BF16 one needs only the BF16-accurate exp and one reciprocal step.
constexpr bool fp32_output = get_compile_time_arg_val(21);
// The output gate x @ W of the core's chunks: K blocks of x and of W, at gate_fidelity.
constexpr uint32_t cb_gin0 = get_compile_time_arg_val(22);
constexpr uint32_t cb_gin1 = get_compile_time_arg_val(23);
constexpr uint32_t gate_kb = get_compile_time_arg_val(24);
constexpr uint32_t gate_blocks = get_compile_time_arg_val(25);
constexpr auto gate_fidelity = static_cast<ckernel::MathFidelity>(get_compile_time_arg_val(26));
// With gate_ways 2 the heads pair up, h with h ^ 1, each core forming both over its head's half of K: the partial sums
// of the pair's other head, and the pair's core's of this core's head (gate_matmul.hpp).
constexpr uint32_t cb_gate_pair = get_compile_time_arg_val(27);
constexpr uint32_t cb_gate_partial = get_compile_time_arg_val(28);
constexpr uint32_t gate_ways = get_compile_time_arg_val(29);
constexpr uint32_t state_tiles = kt * kt;
// A chunk row of the outputs is formed in one DST pass (the op checks K against the DST).
static_assert(kt <= compute_kernel_lib::DEST_AUTO_LIMIT);
// The terms buffer holds a chunk row of F(0) at tile row, and of F(I) at fi_offset further: in place, the core's whole
// shard (core_chunks rows of F(0), then of F(I)); streamed, one chunk at a time.
constexpr uint32_t fi_offset = (streamed ? 1 : core_chunks) * kt;
constexpr uint32_t terms_tiles = 2 * fi_offset;

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

// The norm weight's row 0 broadcast down every row, for the elementwise product with each output row.
void broadcast_weight() {
    DataflowBuffer weight(cb_weight);
    DataflowBuffer weight_rows(cb_weight_rows);
    ckl::unary_bcast<
        ckl::BroadcastDim::Row,
        ckl::input(cb_weight, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::output(cb_weight_rows, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
        ckl::IterationShape::tiles(kt), weight, weight_rows);
    cb_wait_front(cb_weight_rows, kt);
}

// Each of the next `chunks` gate rows into its chunk's g = sigmoid(gate) * weight.
void gate_factors(uint32_t chunks) {
    DataflowBuffer gate(cb_gate);
    DataflowBuffer weight_rows(cb_weight_rows);
    DataflowBuffer gate_factor(cb_gate_factor);
    constexpr auto gate_row =
        ckl::input(cb_gate, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block);
    constexpr auto weight_row =
        ckl::input(cb_weight_rows, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block);
    constexpr auto gate_factor_row = ckl::output(cb_gate_factor, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd);
    for (uint32_t c = 0; c < chunks; ++c) {
        if constexpr (fp32_output) {
            // The product on the SFPU, in FP32: the FPU's DEST reuse would round sigmoid(gate) to srcA's format.
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(kt).block_size(kt),
                ckl::CopyTile<gate_row, ckl::Dst::D0>{gate},
                ckl::Sigmoid<ckl::Dst::D0>{},
                ckl::CopyTile<weight_row, ckl::Dst::D1>{weight_rows},
                ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
                ckl::PackTile<gate_factor_row, ckl::Dst::D0>{gate_factor});
        } else {
            // A BF16 g needs only the BF16-accurate sigmoid, and the FPU product reusing DEST.
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(kt).block_size(kt),
                ckl::CopyTile<gate_row>{gate},
                ckl::Sigmoid<ckl::Dst::D0, ckl::Approx::Exact, ckl::SigmoidPrecision::Bf16>{},
                ckl::DestReuseBinary<ckl::BinaryFpuOp::Mul, weight_row, ckl::DestReuseType::DEST_TO_SRCA>{weight_rows},
                ckl::PackTile<gate_factor_row>{gate_factor});
        }
    }
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

// Every correction core forms its chunks' gate rows, then corrects and normalizes its chunks once their seeds land.
void kernel_main() {
    const uint32_t head = kda_relay::correction_core<heads_per_column, first_row, cores_per_head>().head;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_terms, cb_seed, cb_out);
    cb_wait_front(cb_ctl, 1);
    const uint32_t chunks = read_tile_value(cb_ctl, 0, 0);
    const uint32_t group_chunks = read_tile_value(cb_ctl, 0, 1);
    cb_pop_front(cb_ctl, 1);

    kda_gate::matmul<gate_fidelity>(
        cb_gin0,
        cb_gin1,
        cb_gate,
        cb_gate_pair,
        gate_ways == 1 || head % 2 == 0,
        gate_ways,
        core_chunks,
        kt,
        gate_kb,
        gate_blocks);
    if constexpr (gate_ways == 2) {
        kda_gate::add_pair(cb_gate_partial, cb_gate, core_chunks * kt);
    }
    // The norm's weight rows.
    if (chunks != 0) {
        cb_wait_front(cb_scaler, 1);
        broadcast_weight();
    }
    // In place, every chunk's g and difference while the seeds are on their way.
    if constexpr (!streamed) {
        gate_factors(chunks);
        if (chunks != 0) {
            cb_wait_front(cb_terms, terms_tiles);
            cb_reserve_back(cb_diff, chunks * kt);
            subtract(0, chunks * kt);
            cb_push_back(cb_diff, chunks * kt);
            cb_wait_front(cb_diff, chunks * kt);
        }
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
    if (chunks != 0) {
        cb_pop_front(cb_weight_rows, kt);
        cb_pop_front(cb_scaler, 1);
    }
}
