// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post compute.
//
// Helper substitution (documented per the helper-first policy): the per-term SFPU mix (MulBinary + n x
// Addcmul, one SFPU pass per term) is replaced by ONE custom SFPU pass, `WeightedSum` below, that
// computes out = sum_t coef_t * data_t over a group of DEST-resident terms. No kernel_lib element
// expresses an n-term weighted sum, and the per-term chain paid, per output tile, n+1 SFPU passes
// that each re-load / re-store the fp32 accumulator plus a copy-init / SFPU-init / unpack-reconfig
// switch per element (measured: that switching, not the copies, dominated). WeightedSum reads each
// coefficient vector once for two data faces (the half-packed layout in mhc_post_common.hpp) and keeps
// the accumulator in SFPU registers. Numerics are those of the old chain: acc = d_0 * c_0 (one rounding),
// then acc = d_t * c_t + acc (fused MAD, one rounding) for t = 1 .. in the same term order.
// Everything else (copies, packs, the DEST sync window, reconfig) stays in eltwise_chain; WeightedSum is a
// DEST-only chain element (UnaryOp CRTP) so the chain owns its init and the dst-sync window.
//
// DEST window (SyncFull: the host sets dst_full_sync_en, DEST_AUTO_LIMIT = 8 fp32 slots), one
// eltwise_chain iteration per window:
//   D0 .. D(P-1)   the P = ceil((n+1)/2) half-packed coefficient tiles of output stream j
//   then data      per window column c: F[c], X_0[c] .. X_{n-1}[c] (n+1 slots); WeightedSum writes the
//                  column's output over its F slot, which is packed to out slot j*B + c.
// "fits"   regime (P + n + 1 <= DEST_AUTO_LIMIT, n <= 4): K = (DEST_AUTO_LIMIT - P) / (n+1) columns per
//          window, all copies first, then K WeightedSums, then K packs.
// "grouped" regime (n = 5): one column per window; terms are loaded in groups that fit the free slots
//          and accumulated in the first data slot (a WeightedSum with Accumulate).
// All fp32 CBs read via copy_tile are tagged UnpackToDestFp32 on the host (lossless); no FPU math.
// CB lifecycles are caller-managed (None, None) with TileAddressing::Offset bases: waited / reserved once
// per block (coefficients once per segment), popped / pushed with nominal counts.

#include <cstdint>
#include <utility>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/ternary/ternary.hpp"
#include "perf_instrumentation.hpp"
#include "mhc_post_common.hpp"

using namespace compute_kernel_lib;

namespace {

constexpr uint32_t n = get_compile_time_arg_val(0);
constexpr uint32_t col_tiles_per_row = get_compile_time_arg_val(1);
constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);
constexpr uint32_t cb_sublayer_tiles = get_compile_time_arg_val(3);
constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(4);
constexpr uint32_t cb_coef_bcast = get_compile_time_arg_val(5);
constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(6);
constexpr uint32_t coef_tiles_per_stream = get_compile_time_arg_val(7);  // P = ceil((n+1)/2)

// ---- DEST window layout (single source: DEST_AUTO_LIMIT) ----
constexpr uint32_t NUM_TERMS = n + 1;  // post term + n comb terms
constexpr uint32_t DST_DATA0 = coef_tiles_per_stream;
static_assert(2 * coef_tiles_per_stream >= NUM_TERMS, "mhc_post: coefficient tiles hold two terms each");
static_assert(DST_DATA0 + 2 <= DEST_AUTO_LIMIT, "mhc_post: coefficients + two data slots exceed DEST");
constexpr uint32_t DATA_SLOTS = DEST_AUTO_LIMIT - DST_DATA0;
constexpr bool TERMS_FIT = NUM_TERMS <= DATA_SLOTS;
constexpr uint32_t WINDOW_FIT = TERMS_FIT ? DATA_SLOTS / NUM_TERMS : 1;
constexpr uint32_t WINDOW_COL_TILES = WINDOW_FIT < block_col_tiles ? WINDOW_FIT : block_col_tiles;

constexpr Dst slot(uint32_t s) { return static_cast<Dst>(s); }

#ifdef TRISC_MATH
// out = [out +] sum_{k < NumTerms} coef(FirstTerm + k) * data(DataSlot0 + k), fp32, one tile.
// Coefficient of term t: slot coef_tile_in_stream(t), face 2h + coef_half(t) (mhc_post_common.hpp);
// it serves data faces 2h and 2h+1 (same tile rows). One SFPU row = 2 DEST rows; 8 per face, 32 per tile.
template <uint32_t NumTerms, uint32_t FirstTerm, uint32_t DataSlot0, uint32_t OutSlot, bool Accumulate>
struct WeightedSumSfpu {
    static constexpr int TILE_ROWS = 32;
    static constexpr int FACE_ROWS = 8;

    template <uint32_t K>
    static inline void term(sfpi::vFloat& acc_a, sfpi::vFloat& acc_b) {
        constexpr uint32_t t = FirstTerm + K;
        constexpr int coef_off =
            int(mhc_post::coef_tile_in_stream(t)) * TILE_ROWS + int(mhc_post::coef_half(t)) * FACE_ROWS;
        constexpr int data_off = int(DataSlot0 + K) * TILE_ROWS;
        const sfpi::vFloat c = sfpi::dst_reg[coef_off];
        const sfpi::vFloat xa = sfpi::dst_reg[data_off];
        const sfpi::vFloat xb = sfpi::dst_reg[data_off + FACE_ROWS];
        if constexpr (K == 0 && !Accumulate) {
            acc_a = xa * c;
            acc_b = xb * c;
        } else {
            acc_a = xa * c + acc_a;
            acc_b = xb * c + acc_b;
        }
    }

    template <uint32_t... Ks>
    static inline void row(std::integer_sequence<uint32_t, Ks...>) {
        constexpr int out_off = int(OutSlot) * TILE_ROWS;
        sfpi::vFloat acc_a;
        sfpi::vFloat acc_b;
        if constexpr (Accumulate) {
            acc_a = sfpi::dst_reg[out_off];
            acc_b = sfpi::dst_reg[out_off + FACE_ROWS];
        }
        (term<Ks>(acc_a, acc_b), ...);
        sfpi::dst_reg[out_off] = acc_a;
        sfpi::dst_reg[out_off + FACE_ROWS] = acc_b;
    }

    static inline void run() {
#pragma GCC unroll 0
        for (int h = 0; h < 2; ++h) {  // tile rows 0-15 (faces 0/1), 16-31 (faces 2/3)
#pragma GCC unroll 8
            for (int d = 0; d < FACE_ROWS; ++d) {
                row(std::make_integer_sequence<uint32_t, NumTerms>{});
                sfpi::dst_reg++;
            }
            // Next row half = face 2. The face increment rebases on the carry register (the face start),
            // discarding the in-loop dst_reg++ steps, so two increments move face 0 -> face 2.
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
    }
};
#endif

// DEST-only chain element wrapping WeightedSumSfpu (lane width = highest slot touched + 1).
template <uint32_t NumTerms, uint32_t FirstTerm, uint32_t DataSlot0, uint32_t OutSlot, bool Accumulate>
struct WeightedSum
    : UnaryOp<WeightedSum<NumTerms, FirstTerm, DataSlot0, OutSlot, Accumulate>, slot(DataSlot0 + NumTerms - 1)> {
    static_assert(OutSlot < DEST_AUTO_LIMIT && DataSlot0 + NumTerms <= DEST_AUTO_LIMIT);
    static ALWI void init() { MATH((llk_math_eltwise_ternary_sfpu_init<SfpuType::addcmul>())); }
    static ALWI void exec_impl(uint32_t /*slot_offset*/) {
        MATH(
            (_llk_math_eltwise_sfpu_start_(0),
             WeightedSumSfpu<NumTerms, FirstTerm, DataSlot0, OutSlot, Accumulate>::run(),
             _llk_math_eltwise_sfpu_done_()));
    }
};

template <uint32_t S, DataFormatReconfig R = DataFormatReconfig::Enabled>
using CoefCopy = CopyTile<
    input(cb_coef_bcast, WaitPolicy::None, PopPolicy::None, InputTileMapping::Scalar, R, TileAddressing::Offset),
    slot(S)>;
template <uint32_t S, DataFormatReconfig R = DataFormatReconfig::Enabled>
using SublayerCopy = CopyTile<
    input(cb_sublayer_tiles, WaitPolicy::None, PopPolicy::None, InputTileMapping::Scalar, R, TileAddressing::Offset),
    slot(S)>;
template <uint32_t S, DataFormatReconfig R = DataFormatReconfig::Enabled>
using ResidualCopy = CopyTile<
    input(cb_residual_tiles, WaitPolicy::None, PopPolicy::None, InputTileMapping::Scalar, R, TileAddressing::Offset),
    slot(S)>;
// The output CB is the kernel's only pack target and compute_kernel_hw_startup configured the packer
// for it, so the per-window pack reconfig (unconditional at a chain's entry) is disabled.
template <uint32_t S>
using PackOutput = PackTile<
    output(
        cb_output_tiles, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Disabled, TileAddressing::Offset),
    slot(S)>;

// CB tile of data term t (0 = F, 1+i = X_i) of block column c (sublayer CB for t = 0, residual CB else).
constexpr uint32_t residual_tile(uint32_t t, uint32_t c) { return (t - 1) * block_col_tiles + c; }

// ======================= "fits" regime: K columns per window =======================
// Copies are grouped by source CB so the srcA format / unpack-mode switch (coefficients: fp32
// UnpackToDestFp32; data: its own format) happens once inside a window. Consecutive windows alternate
// the group order — coefficient-first (coef, F, X) then data-first (X, F, coef) — so each window starts
// on the CB the previous window ended on; that first copy's reconfig (unconditional at a chain's entry)
// is disabled. The walk starts data-first: compute_kernel_hw_startup leaves srcA on cb_residual_tiles.
// Then K WeightedSums, then K packs.
constexpr uint32_t data_slot(uint32_t c, uint32_t t) { return DST_DATA0 + c * NUM_TERMS + t; }

template <uint32_t K, uint32_t C, class... Es>
ALWI void fit_packs(uint32_t out_tile0, Es... elts) {
    if constexpr (C == K) {
        eltwise_chain(IterationShape::one_tile(), elts...);
    } else {
        fit_packs<K, C + 1>(out_tile0, elts..., PackOutput<data_slot(C, 0)>{out_tile0 + C});
    }
}

template <uint32_t K, uint32_t C, class... Es>
ALWI void fit_sums(uint32_t out_tile0, Es... elts) {
    if constexpr (C == K) {
        fit_packs<K, 0>(out_tile0, elts...);
    } else {
        fit_sums<K, C + 1>(out_tile0, elts..., WeightedSum<NUM_TERMS, 0, data_slot(C, 0), data_slot(C, 0), false>{});
    }
}

// Coefficient copies: P tiles of stream j into D0 .. D(P-1).
template <DataFormatReconfig First, uint32_t... Ps, class... Es>
ALWI auto with_coefs(uint32_t j, std::integer_sequence<uint32_t, Ps...>, Es... elts) {
    return [=](auto next) {
        next(elts..., CoefCopy<Ps, (Ps == 0 ? First : DataFormatReconfig::Enabled)>{j * coef_tiles_per_stream + Ps}...);
    };
}

// Data copies of one window, term-major: F then X_0.. X_{n-1}, or (XFirst) X_0.. X_{n-1} then F. The first copy
// emitted takes reconfig `First`.
template <uint32_t K, bool XFirst, DataFormatReconfig First, uint32_t Step, class Next, class... Es>
ALWI void data_copies(uint32_t col0, Next next, Es... elts) {
    constexpr uint32_t STEPS = NUM_TERMS * K;
    if constexpr (Step == STEPS) {
        next(elts...);
    } else {
        // Step -> (term, column): X terms 1..n then F (XFirst) or F then X terms 1..n.
        constexpr uint32_t ordinal = Step / K;
        constexpr uint32_t c = Step % K;
        constexpr uint32_t t = XFirst ? (ordinal + 1) % NUM_TERMS : ordinal;
        constexpr DataFormatReconfig R = Step == 0 ? First : DataFormatReconfig::Enabled;
        if constexpr (t == 0) {
            data_copies<K, XFirst, First, Step + 1>(col0, next, elts..., SublayerCopy<data_slot(c, 0), R>{col0 + c});
        } else {
            data_copies<K, XFirst, First, Step + 1>(
                col0, next, elts..., ResidualCopy<data_slot(c, t), R>{residual_tile(t, col0 + c)});
        }
    }
}

template <uint32_t K, bool DataFirst>
ALWI void fit_window(uint32_t j, uint32_t col0) {
    const uint32_t out_tile0 = j * block_col_tiles + col0;
    auto finish = [=](auto... elts) { fit_sums<K, 0>(out_tile0, elts...); };
    constexpr auto coef_seq = std::make_integer_sequence<uint32_t, coef_tiles_per_stream>{};
    if constexpr (DataFirst) {
        // X.., F.., then the coefficients
        data_copies<K, true, DataFormatReconfig::Disabled, 0>(
            col0, [=](auto... elts) { with_coefs<DataFormatReconfig::Enabled>(j, coef_seq, elts...)(finish); });
    } else {
        // coefficients, then F.., X..
        with_coefs<DataFormatReconfig::Disabled>(j, coef_seq)(
            [=](auto... elts) { data_copies<K, false, DataFormatReconfig::Enabled, 0>(col0, finish, elts...); });
    }
}

// ======================= "grouped" regime: one column, term groups =======================
// Group 0 loads terms [0, DATA_SLOTS) into D(P).., sums them over D(P). Group g >= 1 loads up to
// DATA_SLOTS - 1 terms into D(P+1).. and accumulates onto D(P).
constexpr uint32_t group_first(uint32_t g) { return g == 0 ? 0 : DATA_SLOTS + (g - 1) * (DATA_SLOTS - 1); }
constexpr uint32_t group_capacity(uint32_t g) { return g == 0 ? DATA_SLOTS : DATA_SLOTS - 1; }
constexpr uint32_t group_count(uint32_t g) {
    return NUM_TERMS - group_first(g) < group_capacity(g) ? NUM_TERMS - group_first(g) : group_capacity(g);
}
constexpr uint32_t group_slot0(uint32_t g) { return g == 0 ? DST_DATA0 : DST_DATA0 + 1; }

template <uint32_t G, uint32_t K, class... Es>
ALWI void grouped_terms(uint32_t j, uint32_t col, Es... elts) {
    if constexpr (group_first(G) >= NUM_TERMS) {
        eltwise_chain(IterationShape::one_tile(), elts..., PackOutput<DST_DATA0>{j * block_col_tiles + col});
    } else if constexpr (K == group_count(G)) {
        grouped_terms<G + 1, 0>(
            j, col, elts..., WeightedSum<group_count(G), group_first(G), group_slot0(G), DST_DATA0, (G > 0)>{});
    } else {
        constexpr uint32_t t = group_first(G) + K;
        if constexpr (t == 0) {
            grouped_terms<G, K + 1>(j, col, elts..., SublayerCopy<group_slot0(G) + K>{col});
        } else {
            grouped_terms<G, K + 1>(j, col, elts..., ResidualCopy<group_slot0(G) + K>{residual_tile(t, col)});
        }
    }
}

template <uint32_t... Ps>
ALWI void grouped_window(uint32_t j, uint32_t col, std::integer_sequence<uint32_t, Ps...>) {
    grouped_terms<0, 0>(j, col, CoefCopy<Ps>{j * coef_tiles_per_stream + Ps}...);
}

// ======================= block schedule =======================
// `data_first` is the window-order parity of the "fits" regime; it advances once per window across the
// whole kernel (blocks, streams, segments), so every window starts on the CB its predecessor ended on.
template <uint32_t K>
ALWI void mix_window(uint32_t j, uint32_t col0, bool& data_first) {
    if constexpr (TERMS_FIT) {
        if (data_first) {
            fit_window<K, true>(j, col0);
        } else {
            fit_window<K, false>(j, col0);
        }
        data_first = !data_first;
    } else {
        grouped_window(j, col0, std::make_integer_sequence<uint32_t, coef_tiles_per_stream>{});
    }
}

// Ragged tail window (rem < WINDOW_COL_TILES), dispatched to its compile-time width.
template <uint32_t K>
ALWI void mix_tail_window(uint32_t j, uint32_t col0, uint32_t rem, bool& data_first) {
    if constexpr (K > 0) {
        if (rem == K) {
            mix_window<K>(j, col0, data_first);
        } else {
            mix_tail_window<K - 1>(j, col0, rem, data_first);
        }
    }
}

// mix_block: all n output streams of one block.
ALWI void mix_block(uint32_t valid_col_tiles, bool& data_first) {
    for (uint32_t j = 0; j < n; ++j) {
        // load_coefficients pushes the set one stream at a time (P tiles each); stream j needs streams
        // 0..j. Cumulative, so a no-op after the segment's first block.
        {
            MaybeDeviceZoneScope("compute_wait_coef");  // unpack: starved on the coefficient expander
            cb_wait_front(cb_coef_bcast, (j + 1) * coef_tiles_per_stream);
        }
        uint32_t col0 = 0;
        for (; col0 + WINDOW_COL_TILES <= valid_col_tiles; col0 += WINDOW_COL_TILES) {
            mix_window<WINDOW_COL_TILES>(j, col0, data_first);
        }
        mix_tail_window<WINDOW_COL_TILES - 1>(j, col0, valid_col_tiles - col0, data_first);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t num_coef_tiles = n * coef_tiles_per_stream;
    constexpr uint32_t residual_block_tiles = n * block_col_tiles;
    constexpr uint32_t output_block_tiles = n * block_col_tiles;

    const uint32_t start_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_residual_tiles, cb_coef_bcast, cb_output_tiles);

    bool data_first = true;  // srcA starts on cb_residual_tiles (hw_startup), see "fits" regime
    mhc_post::SegmentWalker walker(start_unit, num_units, col_tiles_per_row);
    while (!walker.done()) {
        const mhc_post::Segment seg = walker.next();
        const uint32_t blocks = mhc_post::num_blocks(seg.col_tiles, block_col_tiles);

        for (uint32_t block_idx = 0; block_idx < blocks; ++block_idx) {
            const uint32_t valid = mhc_post::block_valid_col_tiles(seg.col_tiles, block_col_tiles, block_idx);
            {
                MaybeDeviceZoneScope("compute_wait_in");  // unpack: starved on the reader
                cb_wait_front(cb_sublayer_tiles, block_col_tiles);
                cb_wait_front(cb_residual_tiles, residual_block_tiles);
            }
            {
                MaybeDeviceZoneScope("compute_reserve_out");  // pack: back-pressure from the writer
                cb_reserve_back(cb_output_tiles, output_block_tiles);
            }
            {
                MaybeDeviceZoneScope("compute_mix");  // math: occupancy (wait + work), see attribution doc
                mix_block(valid, data_first);
            }

            cb_push_back(cb_output_tiles, output_block_tiles);
            cb_pop_front(cb_residual_tiles, residual_block_tiles);
            cb_pop_front(cb_sublayer_tiles, block_col_tiles);
        }
        cb_pop_front(cb_coef_bcast, num_coef_tiles);  // release_coefficients
    }
}
