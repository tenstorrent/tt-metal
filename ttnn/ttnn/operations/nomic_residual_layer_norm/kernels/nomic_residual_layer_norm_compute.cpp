// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_residual_layer_norm — compute.
//
// Per row block (one block = the whole resident shard where it fits L1, block_rows x hidden_tiles tiles):
//   residual_add_block      y = x + r -> cb_y (fp32)                  eltwise_chain BinaryFpu Add
//   row_sums_block          S_k partial tiles (DEST)                   eltwise_chain BinaryFpu Add, PerRow
//                           even hidden_tiles: pairwise halves y_j + y_{j+W/2} (W/2 FPU steps a row);
//                           odd hidden_tiles (model_layout, 3): y_j + y_j over all W tiles -> 2*S_k, exact; the
//                           finalize scales S by fl(1/N)/2 (inv_sum_bits), so mean is bit-identical to S*fl(1/N).
//   row_square_sums_block   Q_k = sum y*y (DEST mul-accumulate)        eltwise_chain BinaryFpu Mul, PerRow
//   collapse + pack_stats   per-row S and Q partials -> DEST, SFPU row collapse (fp32) -> one "pair tile" per 2
//                           rows: row j's S in column slot j, its Q in slot 2 + j     custom block op
//   (exchange_block runs on the writer: the pair tiles of every group member land in cb_stats_gather)
//   finalize_stats_block    combine contributors in fixed order 0..G-1 (3 a pass), mean / var / rstd, and
//                           extract both rows' center / rstd column-0 tiles in the same DEST window   custom SFPU
//                           (center = mean; seeded normalize: -mean * rstd, with rstd rounded to tf32)
//   (multi-block plans run phase A of block b+1 before block b's finalize: lookahead, see kernel_main)
//   normalize_block         ((y - mean) * rstd) * gamma + beta -> cb_output        4 x eltwise_chain
//                           (host SEEDED_NORMALIZE knob, default off: 2 seeded-multiply chains instead, see below)
//
// Raw-LLK / custom SFPU justification (pack_* / finalize / extract):
//   * placing row k's column-0 statistic into column {0, 1, 16, 17} of a packed tile (and back) needs a
//     DEST->DEST vector copy with different source and destination vector offsets. copy_dest_values /
//     CopyDest copy a whole tile position-for-position (copy_dest_values.h:33) and cannot shift columns.
//     An FPU selector matmul would route the fp32 statistics through srcA/srcB (TF32 truncation).
//   * the finalize runs only on the DEST vectors that carry statistics (8 vectors a row). The chain's
//     AddBinary / MulUnary / Rsqrt elements are full-tile SFPU passes without a vector-mode parameter
//     (op_design § API Mapping, "Helpers considered and rejected"); the fused single pass is perf lamp L3's
//     alternative. The combine keeps the design's contract: contributors are added in member order
//     0..G-1 on every core, so mean / rstd are bit-identical across a row's hidden slices.
//   * the row collapse runs in DEST right before the placement (sfpu_reduce, raw compute API) instead of through
//     reduce<SUM, REDUCE_ROW, AccumulateViaAdd>: that helper always packs its column-0 result to an output CB,
//     and its only consumer here is the placement, which needs the tile back in DEST. Fusing removes block_rows
//     fp32 pack + unpack round trips and the per-row DEST windows per pass (measured: phase A -1.7 us at M=128).
//     With no reduce helper left, cb_rowstats and cb_reduce_scaler are gone too.
//   * normalize and the residual add run blocked (eltwise_block_tiles tiles per DEST window, PerBlockSize).
//   Gap to close in the helper library: copy_dest_vectors(src, dst, src_vec_offset, dst_vec_offset), and a
//   reduce variant that leaves its result in DEST.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/copy_dest_values.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "api/compute/compute_kernel_api.h"  // sfpu_reduce (row collapse in DEST)

// A packed statistics ("pair") tile carries ROWS <= 2 rows: row j's S in slot j, its Q in slot kPairQSlot + j.
constexpr uint32_t kPairQSlot = 2;

#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"

namespace ckernel::sfpu {

// SFPU DEST addressing (32x32 tile, fp32 DEST): a tile spans 64 address rows; face f starts at 16*f.
// One SFPLOAD at address a reads 4 rows x 8 columns: even columns at a, odd columns at a + 2. So column 0 of
// rows 0-31 lives in the even vectors {0, 4, 8, 12} of face 0 (a = 0) and face 2 (a = 32); column 1 in the
// odd vectors (+2); columns 16 / 17 in faces 1 / 3 (+16, +18). A packed statistics tile stores row k of its
// group of 4 in column {0, 1, 16, 17}[k % 4]: the four slots occupy disjoint vectors.
constexpr uint32_t kDestTileRows = 64;
constexpr uint32_t kFaceRows = 16;
constexpr uint32_t kRowsPerLoad = 4;
constexpr uint32_t kOddColumns = 2;

constexpr uint32_t slot_offset(uint32_t slot) { return (slot & 1u) * kOddColumns + (slot >> 1) * kFaceRows; }

// Column-0 vector v (0..7) of a tile: faces 0 and 2, row groups 0..3.
constexpr uint32_t col0_vector(uint32_t v) { return (v >> 2) * (2 * kFaceRows) + (v & 3u) * kRowsPerLoad; }

// sfpi dst_reg index of DEST address `a` of tile `tile` (dst_reg steps two address rows).
constexpr uint32_t dreg(uint32_t tile, uint32_t a) { return (tile * kDestTileRows + a) >> 1; }

// Every address below is a compile-time constant (template parameters + fully unrolled loops), so each
// load / store is one inline SFPU instruction rather than a RISC-built runtime instruction word.

// DST tile slot SLOT <- column 0 of SRC tile.
template <uint32_t SRC, uint32_t DST, uint32_t SLOT>
inline void stats_place(uint32_t, uint32_t, uint32_t) {
#pragma GCC unroll 8
    for (uint32_t v = 0; v < 8; ++v) {
        sfpi::dst_reg[dreg(DST, col0_vector(v) + slot_offset(SLOT))] =
            sfpi::vFloat(sfpi::dst_reg[dreg(SRC, col0_vector(v))]);
    }
}

// Column 0 of DST tile <- slot SLOT of SRC tile.
template <uint32_t SRC, uint32_t DST, uint32_t SLOT>
inline void stats_extract(uint32_t, uint32_t, uint32_t) {
#pragma GCC unroll 8
    for (uint32_t v = 0; v < 8; ++v) {
        sfpi::dst_reg[dreg(DST, col0_vector(v))] =
            sfpi::vFloat(sfpi::dst_reg[dreg(SRC, col0_vector(v) + slot_offset(SLOT))]);
    }
}

// ACC = ((ACC + IN0) + IN1) + ... over the NUM_IN tiles IN0, IN0 + 1, ... (left to right: the fixed contributor
// order), on the S and Q slots of the first ROWS rows of a pair tile. One pass loads each accumulator vector once.
template <uint32_t ACC, uint32_t IN0, uint32_t NUM_IN, uint32_t ROWS>
inline void stats_accumulate(uint32_t, uint32_t, uint32_t) {
#pragma GCC unroll 4
    for (uint32_t k = 0; k < 2 * ROWS; ++k) {
        const uint32_t slot = (k < ROWS) ? k : kPairQSlot + (k - ROWS);  // S slots, then Q slots
#pragma GCC unroll 8
        for (uint32_t v = 0; v < 8; ++v) {
            const uint32_t a = col0_vector(v) + slot_offset(slot);
            sfpi::vFloat acc = sfpi::dst_reg[dreg(ACC, a)];
#pragma GCC unroll 3
            for (uint32_t i = 0; i < NUM_IN; ++i) {
                acc = acc + sfpi::dst_reg[dreg(IN0 + i, a)];
            }
            sfpi::dst_reg[dreg(ACC, a)] = acc;
        }
    }
}

union Fp32Bits {
    uint32_t u;
    float f;
};

// Pair tile T, rows 0..ROWS-1: S slot -> center, Q slot -> rstd (in place).
// mean = S * inv_sum; var = max(Q * fl(1/N) - mean^2, 0); rstd = rsqrt(var + eps) (exact-mode rsqrt body).
// inv_sum = fl(1/N) / (scale of the packed S: 1 for the pairwise sum, 2 for the odd-width self-add sum).
// SEEDED (seeded normalize): rstd is rounded down to tf32 (the 19-bit srcB the normalize multiply sees it through
// anyway) and center = -mean * rstd, so the seed uses exactly the factor y is multiplied by: z = y * rstd + center =
// (y - mean) * rstd with no rstd mismatch between the two terms. Otherwise center = mean and rstd is full fp32.
constexpr uint32_t kTf32Mask = 0xFFFFE000u;  // fp32 sign + exponent + top 10 mantissa bits
template <uint32_t T, uint32_t ROWS, bool SEEDED>
inline void stats_finalize(
    uint32_t, uint32_t, uint32_t, uint32_t inv_n_bits, uint32_t eps_bits, uint32_t inv_sum_bits) {
    const Fp32Bits inv_n{inv_n_bits};
    const Fp32Bits eps{eps_bits};
    const Fp32Bits inv_sum{inv_sum_bits};
#pragma GCC unroll 2
    for (uint32_t j = 0; j < ROWS; ++j) {
#pragma GCC unroll 8
        for (uint32_t v = 0; v < 8; ++v) {
            const uint32_t s = col0_vector(v) + slot_offset(j);
            const uint32_t q = col0_vector(v) + slot_offset(kPairQSlot + j);
            // Constants are (re)loaded per vector so few values stay live across the rsqrt body.
            const sfpi::vFloat mean = sfpi::vFloat(sfpi::dst_reg[dreg(T, s)]) * sfpi::vFloat(inv_sum.f);
            if constexpr (!SEEDED) {
                sfpi::dst_reg[dreg(T, s)] = mean;
            }
            sfpi::vFloat var = sfpi::vFloat(sfpi::dst_reg[dreg(T, q)]) * sfpi::vFloat(inv_n.f) - mean * mean;
            v_if(var < 0.0f) { var = 0.0f; }
            v_endif;
            const sfpi::vFloat rstd = _calculate_sqrt_body_<false, true, false>(var + sfpi::vFloat(eps.f));
            if constexpr (SEEDED) {
                const sfpi::vFloat rstd_tf32 = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(rstd) & kTf32Mask);
                sfpi::dst_reg[dreg(T, q)] = rstd_tf32;
                sfpi::dst_reg[dreg(T, s)] = -(mean * rstd_tf32);
            } else {
                sfpi::dst_reg[dreg(T, q)] = rstd;
            }
        }
    }
}

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH

// ---- custom block-op wrappers (MATH thread only; the CB / dst-sync scaffolding is done by the caller) ----
ALWI void stats_sfpu_init() { copy_dest_values_init(); }
ALWI void stats_finalize_init() {
    copy_dest_values_init();
    MATH((ckernel::sfpu::sqrt_init<false>()));
}
template <uint32_t SRC, uint32_t DST, uint32_t SLOT>
ALWI void stats_place_tile() {
    MATH((_llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::stats_place<SRC, DST, SLOT>, SRC, SRC, DST, VectorMode::RC_custom)));
}
template <uint32_t SRC, uint32_t DST, uint32_t SLOT>
ALWI void stats_extract_tile() {
    MATH((_llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::stats_extract<SRC, DST, SLOT>, SRC, SRC, DST, VectorMode::RC_custom)));
}
template <uint32_t ACC, uint32_t IN0, uint32_t NUM_IN, uint32_t ROWS>
ALWI void stats_accumulate_tiles() {
    MATH((_llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::stats_accumulate<ACC, IN0, NUM_IN, ROWS>, ACC, IN0, ACC, VectorMode::RC_custom)));
}
template <uint32_t T, uint32_t ROWS, bool SEEDED>
ALWI void stats_finalize_tiles(uint32_t inv_n, uint32_t eps, uint32_t inv_sum) {
    MATH((_llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::stats_finalize<T, ROWS, SEEDED>, T, T, T, VectorMode::RC_custom, inv_n, eps, inv_sum)));
}

namespace ckl = compute_kernel_lib;

namespace {

// Host-owned constants (nomic_residual_layer_norm_program_descriptor.py: compute_ct), single source of truth.
constexpr uint32_t kStatSlots = get_compile_time_arg_val(8);  // rows a packed statistics (pair) tile holds
static_assert(kStatSlots >= 1 && kStatSlots <= kPairQSlot, "pair tile: S in slots 0..1, Q in slots 2..3");
constexpr uint32_t cb_x = get_compile_time_arg_val(9);
constexpr uint32_t cb_residual = get_compile_time_arg_val(10);
constexpr uint32_t cb_gamma = get_compile_time_arg_val(11);
constexpr uint32_t cb_beta = get_compile_time_arg_val(12);
constexpr uint32_t cb_y = get_compile_time_arg_val(13);
constexpr uint32_t cb_row_partials = get_compile_time_arg_val(14);
constexpr uint32_t cb_stats_local = get_compile_time_arg_val(15);
constexpr uint32_t cb_stats_gather = get_compile_time_arg_val(16);
constexpr uint32_t cb_center_col = get_compile_time_arg_val(17);  // per-row mean (seeded: -mean * rstd), col 0
constexpr uint32_t cb_rstd_col = get_compile_time_arg_val(18);
constexpr uint32_t cb_output = get_compile_time_arg_val(19);
constexpr uint32_t kInvSumBits = get_compile_time_arg_val(20);  // fl(1/N) / packed-S scale (see stats_finalize)
constexpr bool kLookahead = get_compile_time_arg_val(21) != 0;  // row-block software pipelining knob
constexpr bool kSeededNormalize = get_compile_time_arg_val(22) != 0;  // normalize as 2 seeded multiplies (knob)
constexpr bool kFusedRecompute = get_compile_time_arg_val(23) != 0;   // lookahead: y - mean as (x - mean) + r (knob)
constexpr uint32_t D0 = 0, D1 = 1, D2 = 2, D3 = 3;

// One pair tile (collapse + pack, fused): the S partials of rows first..first+ROWS-1 (cb_row_partials pages
// first + j) and their Q partials (pages block_rows + first + j) are copied into DEST (lossless fp32
// unpack-to-DEST), collapsed within the tile on the SFPU (row SUM in fp32 -> column 0 of each row), and placed
// into the pair tile in D0: S of row j in slot j, Q of row j in slot kPairQSlot + j. Row `first`'s S is collapsed
// in D0 itself: its column 0 already is slot 0.
template <uint32_t ROWS, uint32_t block_rows>
ALWI void collapse_pack_stats_tile(uint32_t first) {
    static_assert(ROWS >= 1 && ROWS <= kStatSlots);
    tile_regs_acquire();
    copy_init(cb_row_partials);
    copy_tile(cb_row_partials, first, D0);
    copy_tile(cb_row_partials, block_rows + first, D2);
    if constexpr (ROWS > 1) {
        copy_tile(cb_row_partials, first + 1, D1);
        copy_tile(cb_row_partials, block_rows + first + 1, D3);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(D0);
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(D2);
    if constexpr (ROWS > 1) {
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(D1);
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(D3);
    }
    stats_sfpu_init();
    stats_place_tile<D2, D0, kPairQSlot>();
    if constexpr (ROWS > 1) {
        stats_place_tile<D1, D0, 1>();
        stats_place_tile<D3, D0, kPairQSlot + 1>();
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(D0, cb_stats_local);
    tile_regs_release();
}

// Row collapse + pack_sums_block / pack_square_sums_block (fused): the block's block_rows S partial tiles, then
// its block_rows Q partial tiles -> stat_tiles pair tiles.
template <uint32_t block_rows, uint32_t stat_tiles>
ALWI void collapse_pack_stats_block(CircularBuffer& row_partials, CircularBuffer& stats_local) {
    constexpr uint32_t last_rows = block_rows - kStatSlots * (stat_tiles - 1);
    row_partials.wait_front(2 * block_rows);
    reconfig_data_format_srca(cb_row_partials);
    pack_reconfig_data_format(cb_stats_local);
    stats_local.reserve_back(stat_tiles);
    for (uint32_t p = 0; p + 1 < stat_tiles; ++p) {
        collapse_pack_stats_tile<kStatSlots, block_rows>(p * kStatSlots);
    }
    collapse_pack_stats_tile<last_rows, block_rows>((stat_tiles - 1) * kStatSlots);
    stats_local.push_back(stat_tiles);
    row_partials.pop_front(2 * block_rows);
}

// Contributors C.. of pair tile p into D0, up to kCombineWidth at a time (D1..D3), in member order.
constexpr uint32_t kCombineWidth = 3;
template <uint32_t C, uint32_t G, uint32_t ROWS, uint32_t stat_tiles>
ALWI void combine_contributors(uint32_t p) {
    if constexpr (C < G) {
        constexpr uint32_t n = (G - C < kCombineWidth) ? G - C : kCombineWidth;
        copy_tile(cb_stats_gather, C * stat_tiles + p, D1);
        if constexpr (n > 1) {
            copy_tile(cb_stats_gather, (C + 1) * stat_tiles + p, D2);
        }
        if constexpr (n > 2) {
            copy_tile(cb_stats_gather, (C + 2) * stat_tiles + p, D3);
        }
        stats_accumulate_tiles<D0, D1, n, ROWS>();
        combine_contributors<C + n, G, ROWS, stat_tiles>(p);
    }
}

// finalize_stats_block for one pair tile p (ROWS rows): combine the G contributors in member order, finalize
// center / rstd, and publish each row's column-0 center / rstd tiles, all in one DEST window: row 0's center is
// slot 0 (column 0) of the pair tile itself; row 0's rstd and row 1's center / rstd are extracted into D2, D1, D3.
template <uint32_t ROWS, uint32_t hidden_split, uint32_t stat_tiles>
ALWI void finalize_stats_tile(
    uint32_t p, uint32_t inv_hidden_bits, uint32_t eps_bits, CircularBuffer& center_cb, CircularBuffer& rstd_cb) {
    tile_regs_acquire();
    copy_init(cb_stats_gather);
    copy_tile(cb_stats_gather, p, D0);  // contributor 0
    stats_sfpu_init();
    combine_contributors<1, hidden_split, ROWS, stat_tiles>(p);  // ((c0 + c1) + c2) ... identically on every core
    stats_finalize_init();
    stats_finalize_tiles<D0, ROWS, kSeededNormalize>(inv_hidden_bits, eps_bits, kInvSumBits);
    stats_sfpu_init();
    stats_extract_tile<D0, D2, kPairQSlot>();  // rstd of row 0
    if constexpr (ROWS > 1) {
        stats_extract_tile<D0, D1, 1>();               // center of row 1
        stats_extract_tile<D0, D3, kPairQSlot + 1>();  // rstd of row 1
    }
    tile_regs_commit();
    tile_regs_wait();
    center_cb.reserve_back(ROWS);
    pack_tile(D0, cb_center_col);
    if constexpr (ROWS > 1) {
        pack_tile(D1, cb_center_col);
    }
    center_cb.push_back(ROWS);
    rstd_cb.reserve_back(ROWS);
    pack_tile(D2, cb_rstd_col);
    if constexpr (ROWS > 1) {
        pack_tile(D3, cb_rstd_col);
    }
    rstd_cb.push_back(ROWS);
    tile_regs_release();
}

}  // namespace

void kernel_main() {
    constexpr uint32_t block_rows = get_compile_time_arg_val(0);
    constexpr uint32_t hidden_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t stat_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t hidden_split = get_compile_time_arg_val(3);
    constexpr uint32_t num_row_blocks = get_compile_time_arg_val(4);
    constexpr bool has_residual = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t inv_hidden_bits = get_compile_time_arg_val(6);
    constexpr uint32_t eltwise_block_tiles = get_compile_time_arg_val(7);
    const uint32_t eps_bits = get_arg_val<uint32_t>(0);
    // Row-block software pipelining (host LOOKAHEAD knob; only meaningful with more than one block).
    constexpr bool lookahead = kLookahead && num_row_blocks > 1;
    // The sums address their source at a runtime block offset only in the lookahead schedule (block b+1 sits
    // behind block b in the resident x); otherwise at the front, with the offset compile-time-elided.
    constexpr auto kSrcAddressing = lookahead ? ckl::TileAddressing::Offset : ckl::TileAddressing::Direct;

    constexpr uint32_t block_tiles = block_rows * hidden_tiles;
    constexpr uint32_t half_hidden = hidden_tiles / 2;
    constexpr uint32_t gather_pages = stat_tiles * hidden_split;  // per contributor: its stat_tiles pair tiles
    constexpr bool pairwise_row_sum = hidden_tiles % 2 == 0;      // else the self-add sum (2*S)
    static_assert(stat_tiles * kStatSlots >= block_rows, "stat_tiles too small for block_rows");
    constexpr uint32_t last_stat_rows = block_rows - kStatSlots * (stat_tiles - 1);

    // y source for the sums: cb_y (x + r, fp32) or x itself (no residual; bf16 is exact in TF32).
    constexpr uint32_t cb_stats_src = has_residual ? cb_y : cb_x;
    constexpr uint32_t cb_second_in = has_residual ? cb_residual : cb_x;

    compute_kernel_hw_startup(cb_x, cb_second_in, cb_y);

    DataflowBuffer x_buf(cb_x);
    DataflowBuffer residual_buf(cb_second_in);
    DataflowBuffer y_buf(cb_y);
    DataflowBuffer partials_buf(cb_row_partials);
    DataflowBuffer center_buf(cb_center_col);
    DataflowBuffer rstd_buf(cb_rstd_col);
    DataflowBuffer gamma_buf(cb_gamma);
    DataflowBuffer beta_buf(cb_beta);
    DataflowBuffer out_buf(cb_output);
    DataflowBuffer& stats_src_buf = has_residual ? y_buf : x_buf;

    CircularBuffer stats_src_cb(cb_stats_src);
    CircularBuffer x_cb(cb_x);
    CircularBuffer residual_cb(cb_second_in);
    CircularBuffer y_cb(cb_y);
    CircularBuffer row_partials_cb(cb_row_partials);
    CircularBuffer stats_local_cb(cb_stats_local);
    CircularBuffer stats_gather_cb(cb_stats_gather);
    CircularBuffer center_cb(cb_center_col);
    CircularBuffer rstd_cb(cb_rstd_col);

    using namespace ckl;
    constexpr auto shape = IterationShape::grid(block_rows, hidden_tiles);
    // Tiles one DEST window carries in the streaming element-wise stages (residual add, normalize).
    constexpr auto norm_shape = IterationShape::grid(block_rows, hidden_tiles).block_size(eltwise_block_tiles);

    // ---------------- residual_add_block: y = x + r (block at the front of the resident x / residual) ----------------
    // Pops the block's x / residual tiles (they are dead after it, and in place their memory becomes the output).
    auto residual_add_block = [&]() __attribute__((always_inline)) {
        eltwise_chain(
            norm_shape,
            BinaryFpu<
                BinaryFpuOp::Add,
                input(cb_x, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                input(cb_residual, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block)>{
                x_buf, residual_buf},
            PackTile<output(cb_y, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{y_buf});
    };

    // ---------------- phase A: residual add (lookahead: recomputed later), S / Q row sums, pack ----------------
    // `base`: tile offset of this block in the resident x / residual CBs (block_tiles when the previous block is
    // still at their front: the lookahead schedule), 0 otherwise.
    auto phase_a = [&](uint32_t base) __attribute__((always_inline)) {
        uint32_t src_base = base;
        if constexpr (has_residual) {
            if constexpr (lookahead) {
                // x / residual stay resident (normalize re-adds them), so read them in place at `base`.
                x_cb.wait_front(base + block_tiles);
                residual_cb.wait_front(base + block_tiles);
                eltwise_chain(
                    norm_shape,
                    BinaryFpu<
                        BinaryFpuOp::Add,
                        input(
                            cb_x,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        input(
                            cb_residual,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset)>{x_buf, residual_buf, base, base},
                    PackTile<output(cb_y, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{y_buf});
            } else {
                residual_add_block();
            }
            src_base = 0;  // the sums read cb_y
        }

        // The sums read the resident block in place (no pop).
        stats_src_cb.wait_front(src_base + block_tiles);

        // ---------------- row_sums_block: S_k (pairwise halves) or 2*S_k (odd width, self-add) ----------------
        if constexpr (pairwise_row_sum) {
            eltwise_chain(
                IterationShape::grid(block_rows, half_hidden),
                BinaryFpu<
                    BinaryFpuOp::Add,
                    input(
                        cb_stats_src,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Block,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Strided),
                    input(
                        cb_stats_src,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Block,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Strided),
                    Dst::D0,
                    DestAccumulation::PerRow>{
                    stats_src_buf,
                    stats_src_buf,
                    StridedTileRange{src_base, hidden_tiles},
                    StridedTileRange{src_base + half_hidden, hidden_tiles}},
                PackTile<output(
                    cb_row_partials,
                    ReservePolicy::PerOuter,
                    PushPolicy::PerOuter,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Direct,
                    DestAccumulation::PerRow)>{partials_buf});
        } else {
            eltwise_chain(
                shape,
                BinaryFpu<
                    BinaryFpuOp::Add,
                    input(
                        cb_stats_src,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Block,
                        DataFormatReconfig::Enabled,
                        kSrcAddressing),
                    input(
                        cb_stats_src,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Block,
                        DataFormatReconfig::Enabled,
                        kSrcAddressing),
                    Dst::D0,
                    DestAccumulation::PerRow>{stats_src_buf, stats_src_buf, src_base, src_base},
                PackTile<output(
                    cb_row_partials,
                    ReservePolicy::PerOuter,
                    PushPolicy::PerOuter,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Direct,
                    DestAccumulation::PerRow)>{partials_buf});
        }

        // ---------------- row_square_sums_block: Q_k ----------------
        eltwise_chain(
            shape,
            BinaryFpu<
                BinaryFpuOp::Mul,
                input(
                    cb_stats_src,
                    WaitPolicy::None,
                    PopPolicy::None,
                    InputTileMapping::Block,
                    DataFormatReconfig::Enabled,
                    kSrcAddressing),
                input(
                    cb_stats_src,
                    WaitPolicy::None,
                    PopPolicy::None,
                    InputTileMapping::Block,
                    DataFormatReconfig::Enabled,
                    kSrcAddressing),
                Dst::D0,
                DestAccumulation::PerRow>{stats_src_buf, stats_src_buf, src_base, src_base},
            PackTile<output(
                cb_row_partials,
                ReservePolicy::PerOuter,
                PushPolicy::PerOuter,
                DataFormatReconfig::Enabled,
                TileAddressing::Direct,
                DestAccumulation::PerRow)>{partials_buf});

        // ---------------- row collapse + pack_sums_block / pack_square_sums_block (fused, pair tiles) ----------
        collapse_pack_stats_block<block_rows, stat_tiles>(row_partials_cb, stats_local_cb);

        if constexpr (has_residual && lookahead) {
            y_cb.pop_front(block_tiles);  // y is recomputed by normalize; cb_y is free for the next block
        }
    };

    // Lookahead (cb_y holds one block, so normalize cannot reuse phase A's y): normalize stage 1 recomputes the
    // centered y from the resident shards. Fused (host FUSED_RECOMPUTE knob): one chain (x - mean) + r, the DEST-reuse
    // pre-add of the library's layernorm, instead of a residual-add pass y = x + r followed by stage 1's y - mean.
    constexpr bool fused_recompute = has_residual && lookahead && kFusedRecompute && !kSeededNormalize;

    // ---------------- finalize_stats_block + normalize_block (block at the front of x / residual) ----------------
    auto finalize_normalize = [&]() __attribute__((always_inline)) {
        // finalize_stats_block (after the writer's exchange_block)
        stats_gather_cb.wait_front(gather_pages);

        reconfig_data_format_srca(cb_stats_gather);
        pack_reconfig_data_format(cb_center_col);
        for (uint32_t p = 0; p + 1 < stat_tiles; ++p) {
            finalize_stats_tile<kStatSlots, hidden_split, stat_tiles>(p, inv_hidden_bits, eps_bits, center_cb, rstd_cb);
        }
        finalize_stats_tile<last_stat_rows, hidden_split, stat_tiles>(
            stat_tiles - 1, inv_hidden_bits, eps_bits, center_cb, rstd_cb);
        stats_gather_cb.pop_front(gather_pages);

        if constexpr (has_residual && lookahead && !fused_recompute) {
            residual_add_block();  // recompute y = x + r (bit-identical to phase A's)
        }
        if constexpr (kSeededNormalize) {
            // Seeded normalize (knob, default off): ELWMUL accumulates into DEST (its MOP never clears it), so a
            // UnaryBcast seed in the lane followed by a BinaryFpu Mul into the same lane yields seed + a * b; the
            // chain runs element-major within a DEST window, so each stage pays two inits a window, not a tile.
            // Measured a regression (Refinement 2; host SEEDED_NORMALIZE comment). center = -mean * rstd must reach
            // DEST in full fp32 (the host puts cb_center_col on UnpackToDestFp32, so unary_bcast unpacks it straight to
            // DEST): rows with |mean| >> std cancel y * rstd against it.
            // 1) z = center + y * rstd = (y - mean) * rstd -> cb_y. Without a residual the y source is x.
            eltwise_chain(
                norm_shape,
                UnaryBcast<
                    BroadcastDim::Col,
                    input(cb_center_col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col)>{center_buf},
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(cb_stats_src, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
                    input(
                        cb_rstd_col, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col)>{
                    stats_src_buf, rstd_buf},
                PackTile<output(cb_y, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{y_buf});
            // 2) out = beta + z * gamma -> the output shard (beta bf16 row-bcast seed, exact in DEST).
            eltwise_chain(
                norm_shape,
                UnaryBcast<
                    BroadcastDim::Row,
                    input(cb_beta, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row)>{beta_buf},
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(cb_y, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
                    input(cb_gamma, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row)>{
                    y_buf, gamma_buf},
                PackTile<output(cb_output, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{out_buf});
        } else {
            // 1) y = y - mean (col bcast). Without a residual the source is x and the result lands in cb_y.
            if constexpr (fused_recompute) {
                // (x - mean) + r: x and mean through srcA / srcB, the fp32 difference moved DEST -> srcB for the add.
                // Pops the block's x / residual tiles, as residual_add_block does.
                eltwise_chain(
                    norm_shape,
                    BinaryFpu<
                        BinaryFpuOp::Sub,
                        input(cb_x, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                        input(
                            cb_center_col,
                            BroadcastDim::Col,
                            WaitPolicy::Upfront,
                            PopPolicy::AtEnd,
                            InputTileMapping::Col)>{x_buf, center_buf},
                    DestReuseBinary<
                        BinaryFpuOp::Add,
                        input(cb_residual, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                        DestReuseType::DEST_TO_SRCB>{residual_buf},
                    PackTile<output(cb_y, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{y_buf});
            } else {
                eltwise_chain(
                    norm_shape,
                    BinaryFpu<
                        BinaryFpuOp::Sub,
                        input(cb_stats_src, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
                        input(
                            cb_center_col,
                            BroadcastDim::Col,
                            WaitPolicy::Upfront,
                            PopPolicy::AtEnd,
                            InputTileMapping::Col)>{stats_src_buf, center_buf},
                    PackTile<output(cb_y, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{y_buf});
            }
            // 2) y = y * rstd (col bcast), in place.
            eltwise_chain(
                norm_shape,
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(cb_y, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
                    input(
                        cb_rstd_col, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col)>{
                    y_buf, rstd_buf},
                PackTile<output(cb_y, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{y_buf});
            // 3) y = y * gamma (row bcast), in place.
            eltwise_chain(
                norm_shape,
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(cb_y, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
                    input(cb_gamma, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row)>{
                    y_buf, gamma_buf},
                PackTile<output(cb_y, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{y_buf});
            // 4) out = y + beta (row bcast) -> the output shard.
            eltwise_chain(
                norm_shape,
                BinaryFpu<
                    BinaryFpuOp::Add,
                    input(cb_y, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
                    input(cb_beta, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row)>{
                    y_buf, beta_buf},
                PackTile<output(cb_output, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{out_buf});
        }
    };

    // Row-block schedule, one call site per stage. Without lookahead: phase A, finalize, normalize per block. With
    // lookahead (software pipeline, perf lamp L2): phase A of block b+1 runs before block b's finalize, so the
    // writer's exchange of block b overlaps it; block b stays at the front of x / residual until its normalize,
    // which puts block b+1 at offset block_tiles.
    constexpr uint32_t kLag = lookahead ? 1 : 0;
    for (uint32_t step = 0; step < num_row_blocks + kLag; ++step) {
        if (step < num_row_blocks) {
            phase_a(lookahead && step > 0 ? block_tiles : 0);
        }
        if (step >= kLag) {
            finalize_normalize();
        }
    }

    // Balance the persistent CBs (gamma / beta are read for the whole call and never popped inside).
    CircularBuffer gamma_cb(cb_gamma);
    CircularBuffer beta_cb(cb_beta);
    gamma_cb.pop_front(hidden_tiles);
    beta_cb.pop_front(hidden_tiles);
}
