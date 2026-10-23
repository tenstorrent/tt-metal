// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm compute (compute cores) — op_design.md -> Block schedule / API Mapping. All kernel-lib
// helpers; the only raw calls are the CB waits / pops the caller-managed chain policies hand back to the caller.
//
// Per block (= wave) of R_b tile-rows of this core's W-slice (WS tiles per tile-row):
//   square_accumulate_block   eltwise_chain x*x, DEST-accumulated per tile-row -> cb_sumsq_partial. Interleaved:
//                             grid(R_b, WS), cumulative wait (pad tiles zeroed by the reader). Sharded (X_SHARD): the
//                             resident shard block, grid(R_b, V) strided by WS — only the slice's valid tiles (a ragged
//                             last shard's pad tiles are the caller's and never zeroed)
//   collapse_block            reduce<SUM, REDUCE_ROW>(R_b x 1) -> R1-X: cb_device_partial; R3-X: cb_partial_col
//   core_combine_block        R3-X owner: reduce<SUM, REDUCE_COL, AccumulateViaAdd, Skip>(G_w x n_k, row stride q)
//                             over cb_gather -> cb_device_partial (n_k = rows this owner holds; reader's ownership)
//   per C-tile chunk c (a shard-backed slice uses C | S, so WS = nch * C everywhere):
//     scale_by_gamma_block    x * γ (Row broadcast) -> cb_x_gamma — chunk 0 runs before the stats wait, so the γ
//                             pass fills the Fabric exchange's stall shadow
//     (c == 0) device_sum_finalize_block: Σ_{g=0..G-1} slot g (SFPU fp32 adds, fixed order), ×1/(G·W_local), +ε,
//                             rsqrt — on the slots' wire form (face 0 rows 0-1, 2 SFPU vectors), then transpose_dest +
//                             one SFPU lane move put inv in column 0 (no_col0_expand: the reader no longer expands
//                             slots; measured root gather 1.53 -> 1.04 us / wave, compute end -0.95 us on dev 0).
//                             R1-X and single-root R3-X (FWD_INV: the root forwards the raw slots, Perf 1):
//                             EVERY core, all R_b rows -> cb_inv_rms (bit-identical group-wide: same slots, same
//                             order). R3-RS: owner k, its n_k rows -> cb_inv_root (the reader all-gathers inv)
//     scale_by_inv_rms_block  (x·γ | x) * inv (Col broadcast) -> cb_output, one tile-row (C pages) per push; OUT_SHARD:
//                             packed strided into the output shard (reserved / pushed once per block, zero-copy)
//   release_block             pop cb_inv_rms (and cb_x_tiles without γ; with γ x is popped after its last γ chunk)
// Schedule: head(w) = square + collapse + core combine, tail(w) = the chunk loop + release. Back to back, or with the
// WAVE_PIPELINE look-ahead (reader order): head(0), then [head(w+1), tail(w)] — block w+1 is squared at page offset
// R * WS behind block w, which is still at the front of cb_x_tiles (the interleaved x ring holds every wave then, so
// the offset never wraps; a shard-backed x is the shard).

#include <stdint.h>

#include <tuple>
#include <utility>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "api/compute/transpose_dest.h"
#include "tools/profiler/kernel_profiler.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t cb_x_tiles = 0;
constexpr uint32_t cb_reduce_scaler = 1;
constexpr uint32_t cb_sumsq_partial = 2;
constexpr uint32_t cb_partial_col = 3;
constexpr uint32_t cb_gather = 4;
constexpr uint32_t cb_device_partial = 5;
constexpr uint32_t cb_stats_slots = 6;
constexpr uint32_t cb_inv_root = 7;
constexpr uint32_t cb_inv_rms = 8;
constexpr uint32_t cb_gamma_tiles = 9;
constexpr uint32_t cb_x_gamma = 10;
constexpr uint32_t cb_output = 16;

#ifdef TRISC_MATH
// ---- no_col0_expand (perf round 2): the slots arrive in their compact wire form, 32 contiguous fp32 words at the
// page base = tile face 0 rows 0-1 (element i at row i / 16, column i % 16). No RISC-V expansion to column 0 runs
// (the reader's tail_to_col0 / rd_slot_expand / rd_mbr_expand are gone): compute sums and finalizes the wire form
// directly — face 0 rows 0-1 are exactly SFPU vectors 0 (rows 0-3, even columns) and 1 (rows 0-3, odd columns), 2
// vectors per tile instead of column 0's 8 — then moves inv to column 0 once: transpose_dest (exact in both DEST
// widths: the 32-bit path moves hi / lo halves) takes (r, c) -> (c, r), so element i < 16 lands at face 0 column 0
// and element 16 + j at face 0 column 1 row j; one lane-aligned SFPU move (odd vector = columns 2m+1 -> even vector
// = columns 2m of face 2) puts the latter at face 2 column 0 (tile row 16 + j). Same per-lane fp32 ops in the same
// g order as the column-0 path: inv is bit-identical to it (and group-wide).
// Raw sfpi / LLK: no compute API or chain op scopes an SFPU op to two vectors, or moves vectors between faces.
sfpi_inline void add_row01_body(const uint32_t in0, const uint32_t in1, const uint32_t out) {
    constexpr uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 2
    for (int v = 0; v < 2; v++) {
        const sfpi::vFloat a = sfpi::dst_reg[in0 * dst_tile_size_sfpi + v];
        const sfpi::vFloat b = sfpi::dst_reg[in1 * dst_tile_size_sfpi + v];
        sfpi::vFloat sum = a + b;
        if constexpr (!DST_ACCUM_MODE) {
            sum = sfpi::convert<sfpi::vFloat16b>(sum, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[out * dst_tile_size_sfpi + v] = sum;
    }
}

sfpi_inline void finalize_row01_body(const uint32_t inv_w_bits, const uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int v = 0; v < 2; v++) {
        const sfpi::vFloat mean_sq = sfpi::dst_reg[0] * ckernel::sfpu::Converter::as_float(inv_w_bits) +
                                     ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(mean_sq);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg++;
    }
}

// After transpose_dest: face 0 column 1 (odd vectors 1, 3, 5, 7) -> face 2 column 0 (even vectors 16, 18, 20, 22).
sfpi_inline void col1_to_face2_body() {
#pragma GCC unroll 4
    for (int k = 0; k < 4; k++) {
        const sfpi::vFloat c1 = sfpi::dst_reg[2 * k + 1];
        sfpi::dst_reg[16 + 2 * k] = c1;
    }
}
#endif

template <ckl::Dst In0 = ckl::Dst::D0, ckl::Dst In1 = ckl::Dst::D1, ckl::Dst Out = ckl::Dst::D0>
struct AddRow01 : ckl::BinaryOp<AddRow01<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { add_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_binary_sfpu_params_(
            add_row01_body,
            ckl::to_u32(In0) + slot_offset,
            ckl::to_u32(In1) + slot_offset,
            ckl::to_u32(Out) + slot_offset,
            ckernel::VectorMode::RC_custom)));  // one call at the tile base: the body walks vectors 0-1 itself
    }
};

template <ckl::Dst Slot = ckl::Dst::D0>
struct FinalizeRow01 : ckl::UnaryOp<FinalizeRow01<Slot>, Slot> {
    uint32_t inv_w_bits;
    uint32_t eps_bits;
    constexpr FinalizeRow01(uint32_t inv_w, uint32_t eps) noexcept : inv_w_bits(inv_w), eps_bits(eps) {}
    static ALWI void init() { rsqrt_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            finalize_row01_body,
            ckl::to_u32(Slot) + slot_offset,
            ckernel::VectorMode::RC_custom,
            inv_w_bits,
            eps_bits)));
    }
};

// In-DEST 32x32 transpose (both DEST widths), then the face-2 lane move: inv's 32 values -> column 0 (faces 0 / 2),
// the only positions BroadcastDim::Col reads. One element: the transpose's MOP / ADDR_MOD init is re-done per tile by
// the (non-hoisted) chain, and the move's SFPU pass needs no init beyond the SFPU start / done of the params call.
template <ckl::Dst Slot = ckl::Dst::D0>
struct Row01ToCol0 : ckl::UnaryOp<Row01ToCol0<Slot>, Slot> {
    static ALWI void init() { transpose_dest_init<DST_ACCUM_MODE, true>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        transpose_dest<DST_ACCUM_MODE, true>(ckl::to_u32(Slot) + slot_offset);
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            col1_to_face2_body, ckl::to_u32(Slot) + slot_offset, ckernel::VectorMode::RC_custom)));
    }
};

// Slot g of tile-row r sits at page r * G + g (caller-managed: the root waits G * R_b once, pops after).
constexpr auto slots_input = ckl::input(
    cb_stats_slots,
    ckl::WaitPolicy::None,
    ckl::PopPolicy::None,
    ckl::InputTileMapping::Block,
    ckl::DataFormatReconfig::Enabled,
    ckl::TileAddressing::Strided);

template <uint32_t g, uint32_t G>
constexpr auto slot_add() {
    return std::make_tuple(
        ckl::CopyTile<slots_input, ckl::Dst::D1>{ckl::StridedTileRange{g, G}},
        AddRow01<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{});
}

template <uint32_t G, uint32_t... Is>
constexpr auto slot_adds(std::integer_sequence<uint32_t, Is...>) {
    return std::tuple_cat(slot_add<Is + 1, G>()...);
}

// device_sum_finalize_block: D0 = slot 0; D0 += slot g for g = 1..G-1 (same order on every device -> bit-identical
// inv group-wide); D0 = rsqrt(D0 * 1/(G*W_local) + eps) on the wire form (FinalizeRow01), then -> column 0.
template <uint32_t G, uint32_t cb_fin>
FORCE_INLINE void device_sum_finalize(uint32_t rows, uint32_t inv_w_bits, uint32_t eps_bits) {
    std::apply(
        [&](auto... adds) {
            ckl::eltwise_chain(
                ckl::IterationShape::grid(rows, 1),
                ckl::CopyTile<slots_input, ckl::Dst::D0>{ckl::StridedTileRange{0, G}},
                adds...,
                FinalizeRow01<ckl::Dst::D0>{inv_w_bits, eps_bits},
                Row01ToCol0<ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(cb_fin)>{});
        },
        slot_adds<G>(std::make_integer_sequence<uint32_t, G - 1>{}));
}

}  // namespace

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t num_waves = get_compile_time_arg_val(1);
    constexpr uint32_t WS = get_compile_time_arg_val(2);
    constexpr uint32_t C = get_compile_time_arg_val(3);
    constexpr uint32_t nch = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(6);
    constexpr uint32_t G_w = get_compile_time_arg_val(7);
    constexpr bool r3 = get_compile_time_arg_val(8) != 0;
    constexpr bool has_gamma = get_compile_time_arg_val(9) != 0;
    constexpr bool x_shard = get_compile_time_arg_val(18) != 0;
    constexpr bool out_shard = get_compile_time_arg_val(19) != 0;
    constexpr bool gamma_resident = get_compile_time_arg_val(20) != 0;
    constexpr uint32_t OWNERS = get_compile_time_arg_val(21);  // R3-RS owners (1 = a single root)
    constexpr uint32_t Q = R / OWNERS;                         // rows an owner holds per full block (OWNERS | R)
    constexpr bool pipelined = get_compile_time_arg_val(24) > 1;  // slot depth 2 <=> the WAVE_PIPELINE look-ahead

    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits = get_arg_val<uint32_t>(2);
    const bool is_root = get_arg_val<uint32_t>(3) != 0;
    const uint32_t V = get_arg_val<uint32_t>(4);  // valid tiles of this core's W-slice
    const uint32_t my_index = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_collapsed = r3 ? cb_partial_col : cb_device_partial;
    constexpr bool fwd_inv = r3 && OWNERS == 1;  // every member gets the slots and finalizes (reader FWD_INV)
    constexpr uint32_t cb_fin = (r3 && !fwd_inv) ? cb_inv_root : cb_inv_rms;

    compute_kernel_hw_startup(cb_x_tiles, cb_x_tiles, cb_sumsq_partial);

    using SquareAccumulate = ckl::BinaryFpu<
        ckl::BinaryFpuOp::Mul,
        ckl::input(cb_x_tiles, ckl::WaitPolicy::Cumulative, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
        ckl::input(cb_x_tiles, ckl::WaitPolicy::Cumulative, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
        ckl::Dst::D0,
        ckl::DestAccumulation::PerRow>;
    using SumsqPack = ckl::PackTile<ckl::output(
        cb_sumsq_partial,
        ckl::ReservePolicy::PerOuter,
        ckl::PushPolicy::PerOuter,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Direct,
        ckl::DestAccumulation::PerRow)>;
    // x tiles: caller-managed (waited by the square chain, popped at release), strided {c*C, WS}.
    constexpr auto x_strided_input = ckl::input(
        cb_x_tiles,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    constexpr auto gamma_input = ckl::input(
        cb_gamma_tiles,
        ckl::BroadcastDim::Row,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Row,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset);
    constexpr auto inv_input = ckl::input(
        cb_inv_rms, ckl::BroadcastDim::Col, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Col);
    constexpr auto x_gamma_input =
        ckl::input(cb_x_gamma, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block);
    constexpr auto row_output = [](uint32_t cb) {
        return ckl::output(cb, ckl::ReservePolicy::PerOuter, ckl::PushPolicy::PerOuter);
    };
    // Sharded square stage: the resident shard block, strided by WS, accumulated per tile-row.
    using SquareAccumulateStrided = ckl::
        BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, x_strided_input, ckl::Dst::D0, ckl::DestAccumulation::PerRow>;
    // Output shard: packed strided (chunk c of tile-row r -> page r * WS + c * C + j); the caller reserves / pushes.
    constexpr auto shard_output = ckl::output(
        cb_output,
        ckl::ReservePolicy::None,
        ckl::PushPolicy::None,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    // scale_by_inv_rms of one chunk: (x·γ | x) * inv -> the output stream (per tile-row) or the output shard.
    auto scale_by_inv = [&](uint32_t R_b, uint32_t cw, const ckl::StridedTileRange& x_chunk) {
        const auto shape = ckl::IterationShape::grid(R_b, cw);
        if constexpr (has_gamma && out_shard) {
            ckl::eltwise_chain(
                shape,
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_gamma_input, inv_input>{},
                ckl::PackTile<shard_output>{x_chunk});
        } else if constexpr (has_gamma) {
            ckl::eltwise_chain(
                shape,
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_gamma_input, inv_input>{},
                ckl::PackTile<row_output(cb_output)>{});
        } else if constexpr (out_shard) {
            ckl::eltwise_chain(
                shape,
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, inv_input>{x_chunk},
                ckl::PackTile<shard_output>{x_chunk});
        } else {
            ckl::eltwise_chain(
                shape,
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, inv_input>{x_chunk},
                ckl::PackTile<row_output(cb_output)>{});
        }
    };

    if constexpr (has_gamma && gamma_resident) {
        cb_wait_front(cb_gamma_tiles, WS);  // resident for the kernel
    }

    // Geometry of block (= wave) w: (R_b, my_first, my_rows) — rows this root owns: [my_index*q, +my_rows)
    // (reader / host owner_rows).
    struct Block {
        uint32_t R_b;
        uint32_t my_rows;   // rows this core owns (the root's core combine)
        uint32_t fin_rows;  // rows this core finalizes: all of them under FWD_INV, its own otherwise
    };
    auto block = [&](uint32_t w) {
        Block b;
        const uint32_t first = w * R;
        b.R_b = rows > first ? (rows - first < R ? rows - first : R) : 0;
        const uint32_t q = (b.R_b + OWNERS - 1) / OWNERS;
        const uint32_t my_first = my_index * q;
        b.my_rows = (is_root && my_first < b.R_b) ? (b.R_b - my_first < q ? b.R_b - my_first : q) : 0;
        b.fin_rows = fwd_inv ? b.R_b : b.my_rows;
        return b;
    };

    // ---- head(w): square_accumulate_block + collapse_block + core_combine_block ----
    // x_base: pages of cb_x_tiles still held ahead of block w (the look-ahead squares block w+1 while block w waits
    // for its stats: block w's R * WS pages are still at the front).
    auto head = [&](uint32_t w, uint32_t x_base) {
        const Block b = block(w);
        const uint32_t R_b = b.R_b;
        if (R_b == 0) {
            return;
        }
        if constexpr (x_shard || pipelined) {
            // The whole block (resident shard / prefetched), strided by WS from x_base. Sharded: only the slice's V
            // valid tiles (a ragged last shard's pad tiles are the caller's); interleaved: WS (pad tiles zeroed).
            cb_wait_front(cb_x_tiles, x_base + R_b * WS);
            ckl::eltwise_chain(
                ckl::IterationShape::grid(R_b, x_shard ? V : WS),
                SquareAccumulateStrided{ckl::StridedTileRange{x_base, WS}, ckl::StridedTileRange{x_base, WS}},
                SumsqPack{});
        } else {
            ckl::eltwise_chain(ckl::IterationShape::grid(R_b, WS), SquareAccumulate{}, SumsqPack{});
        }
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_sumsq_partial,
            cb_reduce_scaler,
            cb_collapsed,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(ckl::ReduceInputBlockShape::of(R_b, 1));

        // ---- core_combine_block (R3-X owner): its rows' G_w partials, contributor-major with row stride q ----
        if constexpr (r3) {
            if (b.my_rows > 0) {
                const uint32_t q = (R_b + OWNERS - 1) / OWNERS;
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_COL,
                    cb_gather,
                    cb_reduce_scaler,
                    cb_device_partial,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Fast,
                    ckl::ReduceAlgorithm::AccumulateViaAdd,
                    ckl::ReduceWithinTile::Skip>(
                    ckl::ReduceInputBlockShape::of(G_w, b.my_rows), ckl::ReduceInputMemoryLayout::with_row_stride(q));
            }
        }
    };

    // ---- tail(w): per chunk scale_by_gamma, device_sum_finalize (c == 0), scale_by_inv_rms; release ----
    auto tail = [&](uint32_t w) {
        const Block b = block(w);
        const uint32_t R_b = b.R_b;
        if (R_b == 0) {
            return;
        }
        const uint32_t fin_rows = b.fin_rows;
        if constexpr (out_shard) {
            cb_reserve_back(cb_output, R_b * WS);  // the block's pages of the output shard
        }
        for (uint32_t c = 0; c < nch; ++c) {
            const ckl::StridedTileRange x_chunk{c * C, WS};
            // ---- scale_by_gamma_block (chunk c) ----
            if constexpr (has_gamma) {
                if constexpr (!gamma_resident) {
                    cb_wait_front(cb_gamma_tiles, C);  // chunk c's γ (streamed by the writer, C pages)
                }
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(R_b, C),
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_strided_input, gamma_input>{
                        x_chunk, gamma_resident ? c * C : 0},
                    ckl::PackTile<row_output(cb_x_gamma)>{});
                if constexpr (!gamma_resident) {
                    cb_pop_front(cb_gamma_tiles, C);
                }
                // A ragged block (R_b < R) pads cb_x_gamma to a whole R*C cycle per chunk, so every chunk's block
                // starts at the ring base and never straddles its end (pack / unpack do not wrap inside a block).
                if (R_b < R) {
                    cb_reserve_back(cb_x_gamma, (R - R_b) * C);
                    cb_push_back(cb_x_gamma, (R - R_b) * C);
                }
                if (c + 1 == nch) {
                    cb_pop_front(cb_x_tiles, R_b * WS);  // x is no longer needed: free it for the next load
                }
            }
            if (c == 0) {
                // ---- device_sum_finalize_block (its fin_rows rows; the reader pushes a whole G*Q region) ----
                if (fin_rows > 0) {
                    {
                        DeviceZoneScopedN("cmp_wait_stats");
                        cb_wait_front(cb_stats_slots, G * Q);
                    }
                    DeviceZoneScopedN("cmp_finalize");
                    device_sum_finalize<G, cb_fin>(fin_rows, inv_w_bits, eps_bits);
                    cb_pop_front(cb_stats_slots, G * Q);
                }
                DeviceZoneScopedN("cmp_wait_inv");
                cb_wait_front(cb_inv_rms, R_b);
            }
            // ---- scale_by_inv_rms_block (chunk c) ----
            scale_by_inv(R_b, C, x_chunk);
            if constexpr (has_gamma) {
                if (R_b < R) {
                    cb_wait_front(cb_x_gamma, (R - R_b) * C);
                    cb_pop_front(cb_x_gamma, (R - R_b) * C);
                }
            }
        }
        if constexpr (out_shard) {
            cb_push_back(cb_output, R_b * WS);
        }
        // ---- release_block ----
        cb_pop_front(cb_inv_rms, R_b);
        if constexpr (!has_gamma) {
            cb_pop_front(cb_x_tiles, R_b * WS);
        }
    };

    if constexpr (pipelined) {
        // One-block look-ahead (reader order): block w+1's partials go out while block w waits for its exchange.
        head(0, 0);
        for (uint32_t w = 0; w < num_waves; ++w) {
            if (w + 1 < num_waves) {
                head(w + 1, block(w).R_b * WS);
            }
            tail(w);
        }
    } else {
        for (uint32_t w = 0; w < num_waves; ++w) {
            head(w, 0);
            tail(w);
        }
    }

    // reduce() waits the scaler but never pops it; the caller owns that page.
    cb_pop_front(cb_reduce_scaler, 1);
    if constexpr (has_gamma && gamma_resident) {
        cb_pop_front(cb_gamma_tiles, WS);
    }
}
