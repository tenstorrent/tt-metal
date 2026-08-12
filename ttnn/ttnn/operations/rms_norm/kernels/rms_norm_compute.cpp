// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm compute (3 TRISCs).
//
// Per block (block_row_tiles x core_w_tiles tiles, resident in cb_input_tiles
// from sumsq_block through scale_block — which is what keeps the input to ONE
// DRAM crossing):
//
//   tilize_in_block   (ROW_MAJOR input only)
//   sumsq_block       Sum_c x[r,c]^2 accumulated ELEMENTWISE across the block's
//                     hidden tiles inside a persistent fp32 DEST accumulator
//                     (DestAccumulation::PerRow) -> one tile per tile-row.
//                     No block-sized x^2 scratch buffer exists.
//   mask_tail_block   only when this core owns the tensor's last hidden tile and
//                     W % 32 != 0: (x_tail * wmask)^2 -> the second stat column.
//                     mask BEFORE squaring, so a finite poison value is
//                     annihilated instead of overflowing.
//   reduce_stat_block reduce<SUM, REDUCE_ROW> folds the within-tile columns of
//                     the 1..2 stat columns into a column-0-valid tile.
//   combine (root)    the G per-core partials gathered by the writer are summed
//                     in a second DEST accumulation, then finalized
//                     x -> rsqrt(x + eps) into the multicast source. The 1/W_true
//                     divisor is NOT applied here: it rides in the reduce scaler
//                     (see the reader), so each partial is already a share of the
//                     MEAN and the gathered sum is the mean itself.
//   scale_block       x * rstd            (BroadcastDim::Col)
//   gamma_block       normed * gamma      (BroadcastDim::Row), elided w/o gamma
//   untilize_out_block (ROW_MAJOR output only)
//
// UNIFORM BLOCK WIDTH. Every CB moves CB_W_TILES pages per tile-row on every
// core of a reduction group (required both by the peer-addressed cb_rstd /
// cb_stat_gather L1 map and by the CB "capacity is a multiple of the quantum"
// rule, dataflow_api.h:216-221). A core's VALID hidden slice is the runtime
// `core_w <= CB_W_TILES`; the statistics phases walk `core_w` columns at row
// stride CB_W_TILES, so the trailing pad tiles never enter the reduction. The
// apply phases DO cover the pad columns (uniform, contiguous, no stride) — their
// output is simply never written to DRAM by the writer.
//
// HIDDEN-AXIS CHUNKING (op_design.md regime R3, Refinement 2b). CB_CHUNK_TILES
// (WC) is the block extent along `hidden` of every buffer that only STREAMS over
// it — cb_gamma_tiles, cb_normed, cb_output_*, cb_input_rm. At WC == CB_W_TILES
// (NUM_CHUNKS == 1, the default for every interleaved geometry) the loops below
// run exactly once and this file is the unchunked schedule. When a resident
// shard PINS both G and C — a HEIGHT shard hands one core the tensor's whole
// hidden slice — those buffers are what overruns L1, so the schedule walks the
// hidden axis in NUM_CHUNKS = ceil(C / WC) chunks instead:
//   * sumsq_block runs once per chunk and packs that chunk's partial Sum x^2
//     into its OWN stat column, so reduce_stat_block (which already sums the
//     `nc` columns of a tile-row) folds them together for free — no L1
//     read-modify-write accumulator, and no change to the combine.
//   * the apply pass runs once per chunk against the chunk's gamma slice.
// The input block stays whole-resident either way (that is what keeps the input
// to one crossing), so chunking never re-reads it.
//
// BLOCK LAYOUT of cb_input_tiles. A TILE input — interleaved (reader-written) or
// a pinned shard — is row-major at row stride CB_W_TILES. A ROW_MAJOR input is
// staged by `tilize<CB_CHUNK_TILES>` per chunk, which emits chunks
// back-to-back, so it is CHUNK-major: tile (r, g) sits at
// (g / WC) * rows * WC + r * WC + g % WC. `in_ref()` is the one place that knows
// which; every phase addresses the block through it. The two coincide at
// NUM_CHUNKS == 1.
//
// PER-STAGE INSTRUMENTATION (PERMANENT — see perf_instrumentation.hpp's
// durability contract). Every zone below is recorded THREE times, once per TRISC
// (unpack / math / pack); read each as that thread's OCCUPANCY, not as one
// "stage cost". Only unpack blocks on tiles arriving and only pack on space, so
// the waits that can be hoisted out of a phase are hoisted into their own
// `*_wait` zone (device-zone-scope-attribution.md §2-§3) — notably the combine's
// gather wait, which is a peer rendezvous and would otherwise be
// indistinguishable from the combine's arithmetic.
//
// Helper substitutions: none. Every phase is a kernel_lib helper call
// (eltwise_chain / eltwise_convenience / reduce / tilize / untilize). The
// caller-managed (None, None) CB policies on sumsq_block and mask_tail_block are
// NOT a substitution — TileOffset::Strided *requires* them
// (eltwise_chain.inl:1169-1172), and strided addressing is what lets both
// phases index the block's hidden tiles at row stride core_w_tiles without
// copying them into a contiguous scratch buffer.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// Perf 1, idea I3: the column-valid finalize elements below hand a raw sfpi body to
// the SAME wrapper the stock SFPU ops use. `_calculate_sqrt_body_` lives here.
#if defined(TRISC_MATH) && !defined(ARCH_QUASAR)
#include "ckernel_sfpu_sqrt.h"
#endif

// TEMPORARY ablation switch (Refinement 4 measurement) — MUST be 0 in committed
// code. 1 drops the apply pass's MATH (scale_block + gamma_block) while keeping
// the CB handshakes on cb_output_tiles / cb_gamma_tiles, so the diff against the
// baseline is the apply pass's compute contribution.
#define RMSN_ABLATE_APPLY 0
// TEMPORARY ablation switch (Refinement 5 measurement) — MUST be 0 in committed
// code. 1 folds only ONE of the leader's W_GROUP_SIZE gathered partials instead
// of all of them, keeping the gather CB's wait/pop quantum intact, so the diff
// against the baseline is the root's fp32 tile-add chain.
#define RMSN_ABLATE_COMBINE_MATH 0
// TEMPORARY ablation switch (Refinement 5 measurement) — MUST be 0 in committed
// code. 1 keeps only the copy + pack of the root's finalize chain (no SFPU at
// all); 2 keeps the Rsqrt but drops MulUnary/AddUnary. Both keep every CB
// quantum, so the diffs split the finalize into its SFPU elements.
#define RMSN_ABLATE_FINALIZE 0

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_wmask = 3;
constexpr uint32_t cb_zero_tile = 4;
constexpr uint32_t cb_stat_sq = 5;
constexpr uint32_t cb_stat_partial = 7;
constexpr uint32_t cb_stat_gather = 8;
constexpr uint32_t cb_stat_gather2 = 15;  // two-stage combine only (root)
constexpr uint32_t cb_branch_sum = 18;    // two-stage combine only (row leaders)
constexpr uint32_t cb_stat_sum = 9;
constexpr uint32_t cb_rstd_send = 10;
constexpr uint32_t cb_rstd = 11;
constexpr uint32_t cb_gamma_rm = 12;
constexpr uint32_t cb_gamma_tiles = 13;
constexpr uint32_t cb_normed = 14;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
}  // namespace

namespace ckl = compute_kernel_lib;

// ---------------------------------------------------------------------------
// COLUMN-VALID SFPU chain elements (Refinement 5).
//
// The finalize chain runs on a `reduce<SUM, REDUCE_ROW>` result, which is
// COLUMN-0-VALID: 31 of every tile's 32 columns are structurally garbage and are
// never read (the apply consumes rstd as an `OperandKind::Col` broadcast, i.e.
// column 0 alone). Column 0 lives in faces 0 and 2, and the SFPU wrapper already
// has a mode for exactly that pair — `VectorMode::C` walks Face0 + Face2 and skips
// the other two (llk_math_eltwise_sfpu_common.h), i.e. HALF the work.
//
// `ckl::Rsqrt` / `ckl::AddUnary` cannot express it: the compute-API entry points
// they call (`rsqrt_tile`, `add_unary_tile`) HARDCODE `VectorMode::RC`. So these
// are the same ops as the kernel_lib ones with the one template argument the
// helper does not thread — built as `ckl::UnaryOp` chain ELEMENTS, so
// eltwise_chain keeps owning the DEST window, the CB lifecycle, the init and the
// format reconfig. This is a missing block operation built, not a helper bypassed.
//
// MEASURED (blackhole_p150b, `(1,1,8192,1024)` BLOCK_SHARDED [1024,128] on (8,8),
// the pinned perf geometry): the finalize chain was 31.1 us of a 76.9 us wall —
// 23.1 us of it the Rsqrt alone, at ~720 ns per one-tile-row stat tile — because
// it is on the ROOT and every other core in the group waits for it.
//
// ---------------------------------------------------------------------------
// PERF 1 (idea I3): `VectorMode::C` IS NOT THE FLOOR — the EVEN-PARITY VECTORS ARE.
//
// RAW-LLK JUSTIFICATION (do not "fix" this back to a plain `SFPU_UNARY_CALL`; that
// undoes a measured win). `VectorMode::C` still runs BOTH column parities of faces
// 0 and 2, but one SFPU vector op covers 4 rows x 8 STRIDE-2 columns and a face is
// walked `[rg0-even, rg0-odd, rg1-even, ...]` — column 0 lives ONLY in the
// even-parity vectors. A column-0-valid result therefore needs 4 vectors per face,
// not 8. `ITERATIONS` cannot express it (it truncates the OUTER, row-group axis
// contiguously); parity is the INNER axis, so the only handle is the DEST address
// stride, which NO wrapper exposes — hence the hand-written `sfpi` body handed to
// `_llk_math_eltwise_unary_sfpu_params_`, the same wrapper `SFPU_UNARY_CALL` itself
// uses, so DEST addressing, the STALLWAIT and the face stepping are unchanged.
// The body reuses `_calculate_sqrt_body_<APPROX, true, false>` — the exact body the
// stock non-legacy `calculate_rsqrt` calls, at the same precision — so this changes
// WHICH vectors run, never the arithmetic inside one.
// NET `dst_reg` ADVANCE MUST STAY +8 (== the stock `ITERATIONS = 8` == one face) or
// `VectorMode::C`'s face0 -> face2 stepping desynchronizes: these bodies do 4 x (+2).
//
// MEASURED (isolated bake-off, blackhole_p150b @1350 MHz, bf16 / HiFi2 /
// fp32_dest_acc_en=False, `cp_finalize` TRISC_1 occupancy per chain call):
//   vectors/tile   64 (`::RC`, pre-R5)   32 (`::C`, R5)   16 (this)
//   rows_t = 1          1011 ns               415 ns        377 ns  (1.10x)
//   rows_t = 16       14475 ns              7568 ns       4409 ns  (1.72x)
//   rows_t = 32       28837 ns             15156 ns       8715 ns  (1.74x)
// With the copy+pack floor removed the cost is exactly proportional to the vector
// count (64:32:16 -> 13405:6498:3339 ns). Whole-kernel at rows_t=16: 7705 -> 4546 ns.
// Column 0 is BIT-IDENTICAL to the `VectorMode::C` chain in BOTH `fp32_dest_acc_en`
// legs, over 11 magnitudes (0, 1e-30 ... 1e18) plus hostile `-1/0/+-inf/nan/+-3.4e38`
// garbage in columns 1..31, with no faults and no leakage into column 0.
// NOT taken: the same skip with the `+eps` and the rsqrt FUSED into one element is a
// further 1.90x at rows_t=16 but a MEASURED REGRESSION at rows_t=1 (0.84x: 8 long
// dependent vectors pipeline worse than 16 short independent ones), and rows_t=1 is
// the decode geometry — so the two elements stay separate.
namespace {
// Even-parity walk of a `VectorMode::C` face pair: 4 vectors instead of 8, net +8.
#if defined(TRISC_MATH) && !defined(ARCH_QUASAR)
template <int STEP, int ITER>
sfpi_inline void rsqrt_col0_body() {
    for (int i = 0; i < ITER; i++) {
        sfpi::vFloat t =
            ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STEP;
    }
}

template <int STEP, int ITER>
sfpi_inline void add_col0_body(uint32_t param) {
    const sfpi::vFloat parameter = ckernel::sfpu::Converter::as_float(param);
    for (int i = 0; i < ITER; i++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = val + parameter;
        sfpi::dst_reg += STEP;
    }
}
#endif

template <ckl::Dst Slot = ckl::Dst::D0>
struct RsqrtColValid : ckl::UnaryOp<RsqrtColValid<Slot>, Slot> {
    static ALWI void init() { ckernel::rsqrt_tile_init<false>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        [[maybe_unused]] const uint32_t slot = ckl::to_u32(Slot) + slot_offset;
#if defined(ARCH_QUASAR)
        // CARVE-OUT (inexpressible, not slower): Quasar has no `_calculate_sqrt_body_`
        // — its `calculate_sqrt` is a different implementation — so the parity-strided
        // body cannot be built there. Keep the `VectorMode::C` pair.
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_rsqrt,
            (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE, false /* FAST_APPROX */, false /* legacy */),
            slot,
            VectorMode::C));
#else
        MATH((_llk_math_eltwise_unary_sfpu_params_((rsqrt_col0_body<2, 4>), slot, VectorMode::C)));
#endif
    }
};

template <ckl::Dst Slot = ckl::Dst::D0>
struct AddUnaryColValid : ckl::UnaryOp<AddUnaryColValid<Slot>, Slot> {
    uint32_t param;
    constexpr explicit AddUnaryColValid(uint32_t p) noexcept : param(p) {}
    static ALWI void init() { ckernel::binop_with_scalar_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        [[maybe_unused]] const uint32_t slot = ckl::to_u32(Slot) + slot_offset;
#if defined(ARCH_QUASAR)
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binop_with_scalar,
            (APPROX, ckernel::ADD_UNARY, 8 /* ITERATIONS */),
            slot,
            VectorMode::C,
            param));
#else
        MATH((_llk_math_eltwise_unary_sfpu_params_((add_col0_body<2, 4>), slot, VectorMode::C, param)));
#endif
    }
};
}  // namespace

void kernel_main() {
    constexpr uint32_t CB_W_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t W_GROUP_SIZE = get_compile_time_arg_val(1);
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(2) != 0;
    constexpr bool IS_RM_IN = get_compile_time_arg_val(3) != 0;
    constexpr bool IS_RM_OUT = get_compile_time_arg_val(4) != 0;
    // Carried by the REDUCE SCALER since Refinement 5 (see the reader), not by a
    // MulUnary on the finalize chain. Kept as a CT arg so the host arg layout and
    // every downstream index are unchanged.
    [[maybe_unused]] constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(5);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(6);
    constexpr bool IS_RM_GAMMA = get_compile_time_arg_val(7) != 0;
    // See the reader: the ROW_MAJOR gamma stick is staged in cb_input_rm when the
    // two CBs share a page format (disjoint lifetimes, same producer + consumer).
    constexpr bool ALIAS_GAMMA_RM = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t cb_gamma_stage = ALIAS_GAMMA_RM ? cb_input_rm : cb_gamma_rm;
    // Block extent along `hidden` of the buffers that only STREAM over it. Equal
    // to CB_W_TILES (one chunk) unless a resident shard pinned C too wide to fit.
    constexpr uint32_t CB_CHUNK_TILES = get_compile_time_arg_val(9);
    // Level-2 fan-in of the combine tree (see the writer). 1 == the flat
    // root-gather: W_GROUP_SIZE is then the whole group, is_leader == is_root, and
    // every `if constexpr (TWO_STAGE)` below compiles out.
    constexpr uint32_t STAGE2_SPAN = get_compile_time_arg_val(10);
    constexpr bool TWO_STAGE = STAGE2_SPAN > 1;
    constexpr uint32_t NUM_CHUNKS = (CB_W_TILES + CB_CHUNK_TILES - 1) / CB_CHUNK_TILES;
    constexpr bool CHUNKED = NUM_CHUNKS > 1;
    // The only chunked configuration with a TILE output is a PINNED output shard
    // (the host gates chunking on a resident, uniform-width shard), whose layout
    // is the shard's own row-major-C one. The apply therefore packs into it at a
    // strided offset under a caller-managed reserve/push, instead of streaming
    // chunk-major pages the writer would then have to un-permute.
    constexpr bool OUT_STRIDED = CHUNKED && !IS_RM_OUT;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t block_row_tiles = get_arg_val<uint32_t>(1);
    const uint32_t last_block_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t core_w = get_arg_val<uint32_t>(3);
    const uint32_t has_tail = get_arg_val<uint32_t>(4);
    const uint32_t is_root = get_arg_val<uint32_t>(5);
    const uint32_t is_leader = get_arg_val<uint32_t>(6);

    {
        MaybeDeviceZoneScope("cp_hw_startup");
        compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_stat_sq);
    }

    // A mcast-box FILLER core: inside a reduction group's broadcast rectangle (a
    // physical shard grid is not always a rectangle) but owning no shard, so it
    // carries no work. It stays a program core only so the group's L1 map stays
    // uniform for the peer-addressed cb_rstd / cb_stat_gather.
    if (num_blocks == 0) {
        return;
    }

    // Hidden tiles handled by the bulk accumulation, and the stat-column layout.
    // ONE stat column per bulk chunk (plus the tail's): reduce_stat_block already
    // sums a tile-row's `nc` columns, so the per-chunk partials fold there instead
    // of through an L1 read-modify-write accumulator. Unchunked this is the
    // Phase 0 expression (bulk_cols = c_full > 0).
    const uint32_t c_full = core_w - has_tail;
    const uint32_t bulk_cols = (c_full + CB_CHUNK_TILES - 1) / CB_CHUNK_TILES;
    const uint32_t nc = bulk_cols + has_tail;  // stat columns per tile-row
    const uint32_t tail_col = bulk_cols;

    // Tile (r, g) of the resident input block, as (base, row_stride) — the ONE
    // place that knows whether the block is row-major (TILE input, stride
    // CB_W_TILES) or chunk-major (ROW_MAJOR input, staged chunk by chunk by
    // tilize<CB_CHUNK_TILES>). Identical at NUM_CHUNKS == 1.
    auto in_ref = [](uint32_t g, uint32_t rows_t) -> ckl::StridedTileRange {
        if constexpr (IS_RM_IN) {
            const uint32_t k = g / CB_CHUNK_TILES;
            return ckl::StridedTileRange{k * rows_t * CB_CHUNK_TILES + (g - k * CB_CHUNK_TILES), CB_CHUNK_TILES};
        } else {
            return ckl::StridedTileRange{g, CB_W_TILES};
        }
    };

    // ---- gamma (RM): tilize the stick into the row-0-valid tile form ----
    // Unchunked, gamma is read ONCE for the whole kernel and never popped. Chunked,
    // one chunk's slice is resident at a time, so the reader re-feeds it per chunk
    // inside the block loop and `apply_chunk` pops it there.
    if constexpr (HAS_GAMMA && IS_RM_GAMMA && !CHUNKED) {
        MaybeDeviceZoneScope("cp_gamma_tilize");
        ckl::tilize<CB_W_TILES, cb_gamma_stage, cb_gamma_tiles>(1);
    }

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;

        // ---------------- tilize_in_block (RM input only) ----------------
        // One tilize per chunk: cb_input_rm holds one chunk's width, and the
        // chunks land back-to-back in cb_input_tiles (the chunk-major layout
        // `in_ref` decodes). NUM_CHUNKS == 1 is the single unchunked call.
        if constexpr (IS_RM_IN) {
            MaybeDeviceZoneScope("cp_tilize_in");
            for (uint32_t k = 0; k < NUM_CHUNKS; ++k) {
                ckl::tilize<CB_CHUNK_TILES, cb_input_rm, cb_input_tiles>(rows_t);
            }
        }

        // The whole block stays resident: waited here, read again by every apply
        // chunk, popped exactly once at the end of the block.
        const uint32_t in_block_pages = rows_t * (IS_RM_IN ? (NUM_CHUNKS * CB_CHUNK_TILES) : CB_W_TILES);
        {
            // STARVATION, not work: how long unpack sat waiting for the reader's
            // block. This is the reader's number, read from the consumer side.
            MaybeDeviceZoneScope("cp_wait_in");
            cb_wait_front(cb_input_tiles, in_block_pages);
        }

        // ---------------- sumsq_block + mask_tail_block ----------------
        {
            MaybeDeviceZoneScope("cp_sumsq");
            cb_reserve_back(cb_stat_sq, rows_t * nc);
            for (uint32_t k = 0; k < bulk_cols; ++k) {
                const uint32_t chunk_base = k * CB_CHUNK_TILES;
                const uint32_t cols = (c_full - chunk_base < CB_CHUNK_TILES) ? (c_full - chunk_base) : CB_CHUNK_TILES;
                const ckl::StridedTileRange src = in_ref(chunk_base, rows_t);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(rows_t, cols),
                    ckl::BinaryFpu<
                        ckl::input(
                            cb_input_tiles,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileOffset::Strided),
                        ckl::input(
                            cb_input_tiles,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileOffset::Strided),
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None,
                        ckl::Dst::D0,
                        ckl::DestAccumulation::PerRow>{src, src},
                    ckl::PackTile<ckl::output(
                        cb_stat_sq,
                        ckl::ReservePolicy::None,
                        ckl::PushPolicy::None,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::PackRelu::Disabled,
                        ckl::L1Accumulation::Disabled,
                        ckl::DestAccumulation::PerRow,
                        ckl::TileOffset::Strided)>{ckl::StridedTileRange{k, nc}});
            }
            if (has_tail) {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(rows_t, 1),
                    ckl::BinaryFpu<
                        ckl::input(
                            cb_input_tiles,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileOffset::Strided),
                        ckl::input(cb_wmask, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Scalar),
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Row,
                        ckl::Dst::D0,
                        ckl::DestAccumulation::Disabled>{in_ref(core_w - 1, rows_t)},
                    ckl::Square<>{},
                    ckl::PackTile<ckl::output(
                        cb_stat_sq,
                        ckl::ReservePolicy::None,
                        ckl::PushPolicy::None,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::PackRelu::Disabled,
                        ckl::L1Accumulation::Disabled,
                        ckl::DestAccumulation::Disabled,
                        ckl::TileOffset::Strided)>{ckl::StridedTileRange{tail_col, nc}});
            }
            cb_push_back(cb_stat_sq, rows_t * nc);
        }

        // ---------------- reduce_stat_block ----------------
        // Scaler is a plain 1.0: the hidden padding was already zeroed above, so
        // no partial scaler is needed here.
        {
            MaybeDeviceZoneScope("cp_reduce_stat");
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_stat_sq,
                cb_scaler,
                cb_stat_partial,
                ckl::ReduceInputPolicy::BulkWaitBulkPop>(ckl::ReduceInputBlockShape::of(rows_t, nc, 1));
        }

        // ---------------- combine_stat_block (root side) ----------------
        // The writer has gathered the group's W_GROUP_SIZE partials into
        // cb_stat_gather at slot r * G + member. Summing across slots is one
        // more DEST accumulation over grid(rows_t, G); cb_zero_tile is the
        // identity B operand (BinaryFpu needs two CB inputs).
        // Combine level 1, on every row LEADER: fold this row's W_GROUP_SIZE
        // gathered partials with one DEST accumulation. Flat (STAGE2_SPAN == 1) has
        // leader == root and W_GROUP_SIZE == G, so this IS the Phase 0 chain and it
        // packs straight into cb_stat_sum.
        if (is_leader) {
#if RMSN_ABLATE_COMBINE_MATH
            cb_wait_front(cb_stat_gather, rows_t * W_GROUP_SIZE);
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(rows_t, 1),
                ckl::BinaryFpu<
                    ckl::input(cb_stat_gather, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block),
                    ckl::input(cb_zero_tile, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Scalar),
                    ckl::BinaryFpuOp::Add,
                    ckl::BroadcastDim::None,
                    ckl::Dst::D0,
                    ckl::DestAccumulation::PerRow>{},
                ckl::PackTile<ckl::output(
                    TWO_STAGE ? cb_branch_sum : cb_stat_sum,
                    ckl::ReservePolicy::PerOuter,
                    ckl::PushPolicy::PerOuter,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::PackRelu::Disabled,
                    ckl::L1Accumulation::Disabled,
                    ckl::DestAccumulation::PerRow)>{});
            cb_pop_front(cb_stat_gather, rows_t * W_GROUP_SIZE);
#else
            {
                // The PEER RENDEZVOUS, hoisted out of the chain's own
                // WaitPolicy::Upfront so the group's gather latency is a separate
                // number from the root's tile-add arithmetic. Waiting for the same
                // count twice is idempotent — the helper's wait is then satisfied.
                MaybeDeviceZoneScope("cp_gather_wait");
                cb_wait_front(cb_stat_gather, rows_t * W_GROUP_SIZE);
            }
            MaybeDeviceZoneScope("cp_combine_l1");
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(rows_t, W_GROUP_SIZE),
                ckl::BinaryFpu<
                    ckl::input(
                        cb_stat_gather, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                    ckl::input(cb_zero_tile, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Scalar),
                    ckl::BinaryFpuOp::Add,
                    ckl::BroadcastDim::None,
                    ckl::Dst::D0,
                    ckl::DestAccumulation::PerRow>{},
                ckl::PackTile<ckl::output(
                    TWO_STAGE ? cb_branch_sum : cb_stat_sum,
                    ckl::ReservePolicy::PerOuter,
                    ckl::PushPolicy::PerOuter,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::PackRelu::Disabled,
                    ckl::L1Accumulation::Disabled,
                    ckl::DestAccumulation::PerRow)>{});
#endif
        }

        // Combine level 2, on the ROOT: fold the STAGE2_SPAN row totals its writer
        // gathered. The same chain one level up, with a much smaller fan-in — this
        // is what turns the root's O(G) tile-add walk into O(nx) + O(ny), the O(nx)
        // half running in parallel on every leader.
        if constexpr (TWO_STAGE) {
            if (is_root) {
                {
                    // Level-2 peer rendezvous, split from the arithmetic for the
                    // same reason as level 1.
                    MaybeDeviceZoneScope("cp_gather2_wait");
                    cb_wait_front(cb_stat_gather2, rows_t * STAGE2_SPAN);
                }
                MaybeDeviceZoneScope("cp_combine_l2");
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(rows_t, STAGE2_SPAN),
                    ckl::BinaryFpu<
                        ckl::input(
                            cb_stat_gather2, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                        ckl::input(
                            cb_zero_tile, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Scalar),
                        ckl::BinaryFpuOp::Add,
                        ckl::BroadcastDim::None,
                        ckl::Dst::D0,
                        ckl::DestAccumulation::PerRow>{},
                    ckl::PackTile<ckl::output(
                        cb_stat_sum,
                        ckl::ReservePolicy::PerOuter,
                        ckl::PushPolicy::PerOuter,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::PackRelu::Disabled,
                        ckl::L1Accumulation::Disabled,
                        ckl::DestAccumulation::PerRow)>{});
            }
        }

        if (is_root) {
            // finalize: rsqrt(sum * (1/W_true) + eps) -> the multicast source.
            // The 1/W uses the TRUE, unpadded W.
            MaybeDeviceZoneScope("cp_finalize");
#if RMSN_ABLATE_FINALIZE == 1
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<ckl::input(cb_stat_sum)>{},
                ckl::PackTile<ckl::output(cb_rstd_send)>{});
#elif RMSN_ABLATE_FINALIZE == 2
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<ckl::input(cb_stat_sum)>{},
                ckl::Rsqrt<>{},
                ckl::PackTile<ckl::output(cb_rstd_send)>{});
#else
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<ckl::input(cb_stat_sum)>{},
                AddUnaryColValid<>{EPS_BITS},
                RsqrtColValid<>{},
                ckl::PackTile<ckl::output(cb_rstd_send)>{});
#endif
        }

        // ---------------- scale_block (+ gamma_block), once per chunk ---------
        // cb_rstd is filled by THIS core's writer (multicast landing buffer on
        // every group member, root included via INCLUDE_SRC loopback). It and the
        // resident input block serve EVERY chunk, so both are caller-managed here:
        // waited once, popped once, after the last chunk.
        {
            // The whole cross-core round trip, seen from the consumer: gather +
            // combine + finalize + multicast. On a NON-root member this is the
            // single number that says how much of the wall the combine costs.
            MaybeDeviceZoneScope("cp_wait_rstd");
            cb_wait_front(cb_rstd, rows_t);
        }
        if constexpr (OUT_STRIDED) {
            cb_reserve_back(cb_output_tiles, rows_t * CB_W_TILES);
        }
        for (uint32_t k = 0; k < NUM_CHUNKS; ++k) {
            const uint32_t chunk_base = k * CB_CHUNK_TILES;
            // A strided pack into the row-major output block must stop at column
            // CB_W_TILES or it would spill into the next tile-row; a chunk-major
            // streaming output carries the pad columns of the last chunk, exactly
            // as the unchunked apply carries `CB_W_TILES - core_w`.
            const uint32_t cols =
                OUT_STRIDED ? ((CB_W_TILES - chunk_base < CB_CHUNK_TILES) ? (CB_W_TILES - chunk_base) : CB_CHUNK_TILES)
                            : CB_CHUNK_TILES;
            const ckl::StridedTileRange src = in_ref(chunk_base, rows_t);
            constexpr auto in_spec = ckl::input(
                cb_input_tiles,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::OperandKind::Block,
                ckl::DataFormatReconfig::Enabled,
                ckl::TileOffset::Strided);
            constexpr auto rstd_spec =
                ckl::input(cb_rstd, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Col);
            constexpr auto gamma_spec =
                ckl::input(cb_gamma_tiles, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Row);
            // Perf 2, idea I11 — DEST-lane BLOCKING for the apply pass.
            // The apply is the op's dominant compute stage on every multi-tile-row
            // geometry (`cp_apply_scale` math = 27880 ns of a 40657 ns kernel span on
            // BLOCK_SHARDED (1,1,8192,1024)). Processing `APPLY_BLK` tiles per DEST-sync
            // window instead of one collapses the per-tile acquire/commit overhead.
            // MEASURED (blackhole_p150b @1350 MHz, bf16/TILE/HiFi2/fp32_dest_acc_en=False,
            // ns per apply chunk, PCC identical to the per-tile baseline to six digits —
            // this REORDERS identical FPU work and touches no precision knob):
            //   streaming out (PerChunk + blocking): (1,3) 567 -> 503 (1.13x),
            //     (16,4) 6527 -> 4091 (1.60x), (32,4) 12763 -> 7784 (1.64x),
            //     (1,112) 10666 -> 6804 (1.57x)
            //   OUT_STRIDED (blocking only; its reserve is already caller-hoisted):
            //     (32,4) 11199 -> 7786 (1.44x), (16,4) 5733 -> 4096 (1.40x),
            //     (1,112) 9526 -> 6814 (1.40x)
            // `PerChunk` (not a bulk output) is deliberate AND is the fastest form
            // measured: it reserves/pushes exactly `APPLY_BLK` pages per window, so the
            // writer is never delayed by more than a block, and `cb_output_tiles` keeps
            // its OUTPUT_CB_DEPTH sizing (a bulk output would need rows_t*WC pages — a
            // 16x L1 blow-up on the block-sharded leg, and a deadlock at depth 2).
            // 8 is the measured knee; the chain clamps it to what DEST holds, and to
            // `cols` via the min below, so a narrow chunk simply runs unblocked.
            // HARD CONTRACT: a chain-owned output at blk > 1 MUST be PerChunk.
            // `ReservePolicy/PushPolicy::PerTile` reserves one page per block iteration
            // but packs `inner_count` tiles — a CB overrun that HANGS the device with no
            // static_assert (confirmed on device in round 1 and again here).
            const uint32_t apply_blk = cols < 8u ? cols : 8u;
            constexpr auto out_strided = ckl::output(
                cb_output_tiles,
                ckl::ReservePolicy::None,
                ckl::PushPolicy::None,
                ckl::DataFormatReconfig::Enabled,
                ckl::PackRelu::Disabled,
                ckl::L1Accumulation::Disabled,
                ckl::DestAccumulation::Disabled,
                ckl::TileOffset::Strided);

#if RMSN_ABLATE_APPLY
            // Payload stubbed, synchronization scaffolding intact.
            if constexpr (HAS_GAMMA) {
                cb_wait_front(cb_gamma_tiles, CB_CHUNK_TILES);
            }
            if constexpr (!OUT_STRIDED) {
                // Per tile-row, matching the streaming quantum the writer waits on
                // (cb_output_tiles holds only OUTPUT_CB_DEPTH * WC pages).
                for (uint32_t r = 0; r < rows_t; ++r) {
                    cb_reserve_back(cb_output_tiles, cols);
                    cb_push_back(cb_output_tiles, cols);
                }
            }
#else
            if constexpr (HAS_GAMMA) {
                if constexpr (IS_RM_GAMMA && CHUNKED) {
                    // This chunk's gamma slice, re-fed by the reader per chunk.
                    ckl::tilize<CB_CHUNK_TILES, cb_gamma_stage, cb_gamma_tiles>(1);
                }
                {
                    MaybeDeviceZoneScope("cp_wait_gamma");
                    cb_wait_front(cb_gamma_tiles, CB_CHUNK_TILES);
                }
                {
                    // Pass 1 of the apply: x * rstd (Col broadcast) -> cb_normed.
                    MaybeDeviceZoneScope("cp_apply_scale");
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::grid(rows_t, cols, apply_blk),
                        ckl::BinaryFpu<in_spec, rstd_spec, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col>{src},
                        ckl::PackTile<ckl::output(
                            cb_normed, ckl::ReservePolicy::PerChunk, ckl::PushPolicy::PerChunk)>{});
                }

                MaybeDeviceZoneScope("cp_apply_gamma");
                if constexpr (OUT_STRIDED) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::grid(rows_t, cols, apply_blk),
                        ckl::BinaryFpu<
                            ckl::input(
                                cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                            gamma_spec,
                            ckl::BinaryFpuOp::Mul,
                            ckl::BroadcastDim::Row>{},
                        ckl::PackTile<out_strided>{ckl::StridedTileRange{chunk_base, CB_W_TILES}});
                } else {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::grid(rows_t, cols, apply_blk),
                        ckl::BinaryFpu<
                            ckl::input(
                                cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                            gamma_spec,
                            ckl::BinaryFpuOp::Mul,
                            ckl::BroadcastDim::Row>{},
                        ckl::PackTile<ckl::output(
                            cb_output_tiles, ckl::ReservePolicy::PerChunk, ckl::PushPolicy::PerChunk)>{});
                }
                if constexpr (CHUNKED) {
                    cb_pop_front(cb_gamma_tiles, CB_CHUNK_TILES);
                }
            } else if constexpr (OUT_STRIDED) {
                MaybeDeviceZoneScope("cp_apply_scale");
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(rows_t, cols, apply_blk),
                    ckl::BinaryFpu<in_spec, rstd_spec, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col>{src},
                    ckl::PackTile<out_strided>{ckl::StridedTileRange{chunk_base, CB_W_TILES}});
            } else {
                MaybeDeviceZoneScope("cp_apply_scale");
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(rows_t, cols, apply_blk),
                    ckl::BinaryFpu<in_spec, rstd_spec, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col>{src},
                    ckl::PackTile<ckl::output(
                        cb_output_tiles, ckl::ReservePolicy::PerChunk, ckl::PushPolicy::PerChunk)>{});
            }
#endif

            // ---------------- untilize_out_block (RM output only) ------------
            if constexpr (IS_RM_OUT) {
                MaybeDeviceZoneScope("cp_untilize_out");
                ckl::untilize<CB_CHUNK_TILES, cb_output_tiles, cb_output_rm>(rows_t);
            }
        }
        if constexpr (OUT_STRIDED) {
            cb_push_back(cb_output_tiles, rows_t * CB_W_TILES);
        }
        cb_pop_front(cb_rstd, rows_t);
        cb_pop_front(cb_input_tiles, in_block_pages);
    }
}
