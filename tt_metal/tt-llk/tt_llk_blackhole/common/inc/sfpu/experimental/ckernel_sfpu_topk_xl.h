// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
//  Top-K XL — SFPU bitonic sort / merge / rebuild for K = 512, 1024, 2048
// =============================================================================
//
// This header implements the SFPU primitives that the higher-level top-k LLK
// API (`llk_math_eltwise_unary_sfpu_topk_xl_*`) dispatches to. The core sort
// phases are:
//
//   _topk_xl_local_sort_  : sort K elements in place.
//   _topk_xl_merge_       : bitonic-merge two adjacent K-element sorted
//                           runs into a single K-element top-K run.
//   _topk_xl_rebuild_     : restore the bitonic property of a K-element
//                           run after a merge so the next merge can run.
//
// The "Public API" section at the bottom holds these, their init, the index
// injection / extraction phases and the MATH/PACK split steps. Everything above
// it is an internal helper.
//
// Data layout
// -----------
// Every operated-on word is 32 bits in DST. In the "fused" mode used by the
// distributed top-k op, each word carries two pieces of data:
//
//      [ bf16 value (high 16) | u16 index (low 16) ]
//
// Because the bf16 value sits in the high bits of the 32-bit DST word, the
// fused compare acts as a value compare with the index as a deterministic
// tie-breaker. Since that fused word is an opaque sort key, fused load/store
// paths move it with INT32 modes so FP32 store denormal flushing cannot erase
// low index bits. (`fused=false` keeps values and indices in two distinct DST
// regions.)
//
// Bitonic sort, briefly
// ---------------------
// A bitonic sequence is one that first ascends then descends (or vice
// versa). Sorting two such sequences of length N takes log2(N) "merge"
// passes of compare-exchange at stride N/2, N/4, ... 1. We build a fully
// sorted sequence of length 2*N from two sorted halves by a single bitonic
// merge — that's exactly the loop structure here.
//
// `window_sort<WINDOW, FROM, TO, ORDER>` runs the steps of one window held in
// LREG0-7, strides FROM down to TO; a full sort of 32 runs the windows 2, 4, 8,
// 16 and 32 in turn (see the "Window sort" section).
//
// SFPU execution model
// --------------------
// Each `LREG` holds 32 elements (4 rows × 8 lanes), and `SFPSWAP`
// compare-exchanges two LREGs across all 32 at once. `SFPTRANSP` rotates between two views of the data (rows
// ↔ columns of the 4-element lane block), and `MOVD2B / MOVB2A / MOVA2D /
// MOVB2D` shuttle 4-row strips between Dst, SrcA, and SrcB so that we can
// transpose larger faces (the `transpose_dest_face_32b` / `transpose_8_faces`
// path used for strides > 256).
//
// Replay buffer + MOP expander
// ----------------------------
// The math thread can record a window of up to 32 instructions into the
// "replay buffer" and re-issue any contiguous sub-range with `lltt::replay`.
// On top of that, the MOP expander can fire a 5-slot template per
// `TTI_MOP` issue, where each slot can itself be a REPLAY(start, len).
// Small loop bodies replay directly; the merge inner loop and the length-2048
// rebuild go through the MOP expander (see `topk_mop_config` and
// `topk_rebuild_build2048_mop_config`).
//
// Address-mod recipe (ADDR_MOD_0..7)
// ----------------------------------
// The kernel programs several ADDR_MODs at init time so that SFPSTORE can
// advance Dst by configurable strides without a separate INCRWC issue:
//
//   ADDR_MOD_7 : zero advance (used by SFPLOAD/SFPSTORE that don't move Dst).
//                On MATH it arrives via `_llk_math_eltwise_unary_sfpu_init_`,
//                and PACK gets it through `_topk_xl_remove_msb_values_init_`.
//   ADDR_MOD_6 : +32 (two faces)
//   ADDR_MOD_5 : +16 (one face)
//   ADDR_MOD_4 : +8     (unfused) / +16 (add_lsb_indices reuses this slot)
//   ADDR_MOD_3 : +40 = 32 + 8     (folds trailing +8 into the last store)
//   ADDR_MOD_2 : +24 = 16 + 8     (same idea at 16-stride)
//   ADDR_MOD_1 : +48 = 32 + 16    (folds trailing +16 into the last store)
//   ADDR_MOD_0 : +2 (one 32-lane group, which is the two Dest rows a single
//                SFPU load/store covers). Used by the post-reduction phases
//                `remove_msb_values` and `separate_indices`.
//
// "Folding" means the trailing INCRWC(0, X, ...) advance that would normally
// follow a store gets absorbed into the store's own ADDR_MOD increment,
// saving one math-thread issue per inner loop iter.
//
// Post-reduction phases (`add_lsb_indices`, `remove_msb_values`,
// `separate_indices`) reprogram a subset of slots for their own use — but
// they all run AFTER the last merge/rebuild has consumed the bitonic
// ADDR_MODs above, so the reprogramming is invisible to the hot loops.

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include "ckernel.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "lltt.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "sfpu/experimental/ckernel_sfpu_set_dst_write_addr_offset.h"

// SFPLOADMACRO acceleration of the UNFUSED merge / rebuild bodies (TOPK_XL_UNFUSED_MACRO) and of the fused sort,
// merge and rebuild step groups (TOPK_XL_FUSED_MACRO); see "Unfused bodies" in the SFPLOADMACRO section below
// for the unfused design and the test-only mutation hook. DISABLE_TOPK_XL_SFPLOADMACRO
// and TOPK_XL_BLAZE_COMPAT turn both off. Defined here (before `topk_mop_config`) because the merge MOP body length
// keys on it.
#if !defined(TOPK_XL_BLAZE_COMPAT) && !defined(DISABLE_TOPK_XL_SFPLOADMACRO) && !defined(DISABLE_SFPLOADMACRO)
#define TOPK_XL_UNFUSED_MACRO 1
#define TOPK_XL_FUSED_MACRO   1
#else
#define TOPK_XL_UNFUSED_MACRO 0
#define TOPK_XL_FUSED_MACRO   0
#endif

namespace ckernel
{
namespace sfpu
{

// #############################################################################
//  Internal Helpers
//
//  Building blocks for the Public API below. Only enter/leave_transpose_cfg_block is also used outside this
//  header, by ckernel_sfpu_topk_xl_split.h.
// #############################################################################

constexpr std::uint32_t FACE_INC         = 16;
constexpr std::uint32_t HALF_FACE_INC    = FACE_INC / 2; // 8
constexpr std::uint32_t QUARTER_FACE_INC = FACE_INC / 4; // 4
constexpr std::uint32_t EIGHTH_FACE_INC  = FACE_INC / 8; // 2

constexpr std::uint32_t FACE_AND_HALF_INC     = FACE_INC + HALF_FACE_INC;     // 24
constexpr std::uint32_t TWO_FACE_INC          = 2 * FACE_INC;                 // 32
constexpr std::uint32_t TWO_FACE_AND_HALF_INC = TWO_FACE_INC + HALF_FACE_INC; // 40
constexpr std::uint32_t THREE_FACE_INC        = 3 * FACE_INC;                 // 48

// Defined with the face-transpose helpers below; declared here for detail::transpose_32b_dest_faces.
template <int dst_offset, std::uint32_t last_addr_mod = ADDR_MOD_7>
inline void transpose_dest_face_32b();

// Internal helper functions for TopK
namespace detail
{
template <std::uint32_t ADDR_MODE, std::uint32_t INC>
inline __attribute__((always_inline)) void set_dst_addr_mode()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = INC},
    }
        .set(ADDR_MODE);
}

constexpr std::uint16_t lower16(std::uint32_t x)
{
    return x & 0xFFFFu;
}

constexpr std::uint16_t upper16(std::uint32_t x)
{
    return x >> 16;
}

/**
 * @brief Distance between the two compared elements, in sequence positions.
 *
 * Must be at most half of the @ref WindowSize. ONE and TWO run after SFPTRANSP;
 * FOUR, EIGHT and SIXTEEN run before it.
 */
enum class Stride : std::uint32_t
{
    ONE     = 1,
    TWO     = 2,
    FOUR    = 4,
    EIGHT   = 8,
    SIXTEEN = 16,
};

/**
 * @brief Length of one bitonic window, the run of elements sorted in one direction.
 *
 * Neighbouring windows alternate between max-first and min-first. THIRTY_TWO covers
 * all 32 elements held in LREG0-7, so there is a single window and no alternation.
 */
enum class WindowSize : std::uint32_t
{
    TWO        = 2,
    FOUR       = 4,
    EIGHT      = 8,
    SIXTEEN    = 16,
    THIRTY_TWO = 32,
};

/**
 * @brief Sort direction of the first window; following windows alternate from it.
 *
 * ASCENDING puts the max in the lower position, matching the `ascending` flag
 * of @ref window_sort.
 */
enum class Order : std::uint32_t
{
    DESCENDING,
    ASCENDING,
};

/** @brief Issue the swaps directly, or install them as SFPLOADMACRO templates. */
enum class UseLoadMacro : std::uint32_t
{
    No,
    Yes,
};

/** @brief Whether all of LREG0-7 hold sort keys, or only LREG0-3 with indices in LREG4-7. */
enum class Fused : std::uint32_t
{
    No,
    Yes,
};

/** @brief Whether a local sort stops once each 64-row Dest column is sorted on its own, before anything crosses a column. */
enum class EarlyExit : std::uint32_t
{
    No,
    Yes,
};

/**
 * @brief SFPCONFIG lane mask that flips the swap direction of alternating groups of SFPU lanes, so one sort runs
 * neighbouring groups in opposite directions. Lane c is bit 2c of the mask.
 */
enum class LaneFlip : std::uint32_t
{
    NONE          = 0x0000,
    EVERY_LANE    = 0x4444, // lanes 1, 3, 5, 7
    EVERY_2_LANES = 0x5050, // lanes 2-3, 6-7
    EVERY_4_LANES = 0x5500, // lanes 4-7
};

/**
 * @brief Which of the two optional SFPTRANSPs around the stride-2 / stride-1 steps a @ref window_sort emits.
 *
 * Strides 2 and 1 only run transposed, so a window sort that reaches them wraps them in SFPTRANSPs (T):
 *
 *     BOTH         T · 2 1 · T
 *     NO_LEADING       2 1 · T     the data is already transposed (the previous call used NO_TRAILING)
 *     NO_TRAILING  T · 2 1         the next call continues transposed (it uses NO_LEADING)
 *     NONE             2 1         the caller issues both, for example from a MOP template
 *
 * The leading T is the first instruction of the call, so it exists only when the call starts at stride 2 or 1.
 * A call that starts at stride 4 or above has its T between steps instead (16 8 4 · T · 2 1 · T), and that one is
 * always emitted: NO_LEADING and NONE do not compile there. The trailing T exists only when the call ends at stride 2
 * or 1, so NO_TRAILING and NONE do not compile for a call that stops at stride 4 or above. BOTH is always valid and
 * emits every T the steps have, which is none for a call that stays at stride 4 or above (16 8 4).
 */
enum class Transposes : std::uint32_t
{
    BOTH,
    NO_LEADING,
    NO_TRAILING,
    NONE,
};

/** @brief Whether a replayed body is recorded by this call (and executed while recording) or replayed from earlier. */
enum class ReplayMode : std::uint32_t
{
    RECORD,
    REPLAY,
};

/**
 * @brief Dest distance between the two row groups a load puts in the two halves of its LREGs.
 *
 * load16_rows_x2: LREG0-3 read rows base + {0, 4, 8, 12}; LREG4-7 read the same four rows GroupOffset further on.
 * load8_rows_x2_unfused: LREG0-1 read rows base + {0, 4}; LREG2-3 read the same two rows GroupOffset further on.
 * It is also the distance the first compare step crosses. A Dest address counts rows of 16 columns and a face is 16
 * rows, except bit 1, which picks the odd columns: ODD_COLUMNS (+2) reads the odd columns of the same rows, the other
 * column of each SFPU lane.
 */
enum class GroupOffset : std::uint32_t
{
    ODD_COLUMNS = EIGHTH_FACE_INC, // 2: same rows, odd columns
    HALF_FACE   = HALF_FACE_INC,   // 8
    ONE_FACE    = FACE_INC,        // 16: the next 16 rows
    TWO_FACES   = TWO_FACE_INC,    // 32
    FOUR_FACES  = 4 * FACE_INC,    // 64
};

/**
 * @brief Dst advance folded into the last SFPSTORE of a store helper, so no separate INCRWC is needed.
 *
 * Each value is one ADDR_MOD the topk_xl inits program: NONE is ADDR_MOD_7 (+0), HALF_FACE ADDR_MOD_4 (+8 rows, unfused
 * inits only), ONE_FACE ADDR_MOD_5 (+16), TWO_FACES ADDR_MOD_6 (+32) and THREE_FACES ADDR_MOD_1 (+48).
 */
/**
 * @brief Dest distance from the unfused value region to its index region. A tile is 64 Dest rows (4 faces).
 */
enum class IndicesOffset : std::uint32_t
{
    NONE       = 0,             // fused: indices are packed into the values
    ONE_TILE   = 4 * FACE_INC,  // 64
    TWO_TILES  = 8 * FACE_INC,  // 128
    FOUR_TILES = 16 * FACE_INC, // 256
};

enum class DstAdvance : std::uint32_t
{
    NONE        = 0,
    HALF_FACE   = HALF_FACE_INC,
    ONE_FACE    = FACE_INC,
    TWO_FACES   = TWO_FACE_INC,
    THREE_FACES = THREE_FACE_INC,
};

/**
 * @brief The ADDR_MOD that advances Dst by INC rows on the load or store it is attached to.
 *
 * Each one is programmed by the topk_xl inits: +0 ADDR_MOD_7, +8 ADDR_MOD_4 (unfused), +16 ADDR_MOD_5, +24 ADDR_MOD_2,
 * +32 ADDR_MOD_6, +40 ADDR_MOD_3, +48 ADDR_MOD_1.
 *
 * @tparam INC: Dst advance in rows, values = <0/8/16/24/32/40/48>
 */
template <int INC>
constexpr std::uint32_t dst_advance_addr_mod()
{
    static_assert(INC == 0 || INC == 8 || INC == 16 || INC == 24 || INC == 32 || INC == 40 || INC == 48, "no ADDR_MOD advances Dst by INC");
    return INC == 0    ? ADDR_MOD_7
           : INC == 8  ? ADDR_MOD_4
           : INC == 16 ? ADDR_MOD_5
           : INC == 24 ? ADDR_MOD_2
           : INC == 32 ? ADDR_MOD_6
           : INC == 40 ? ADDR_MOD_3
                       : ADDR_MOD_1;
}

/**
 * @brief Call `body` with the runtime direction as a compile-time value: std::true_type when `ascending` is set,
 * std::false_type otherwise. The body reads it as `decltype(direction)::value`, so a template taking the direction as a
 * bool template argument is instantiated for both and picked at run time.
 */
template <typename Body>
inline void with_direction(const bool ascending, Body body)
{
    if (ascending)
    {
        body(std::true_type {});
    }
    else
    {
        body(std::false_type {});
    }
}

// Strides 1 and 2 run after SFPTRANSP, so neighbouring positions sit in neighbouring LREGs.
constexpr bool is_after_transpose(Stride stride)
{
    return stride == Stride::ONE || stride == Stride::TWO;
}

// LREG distance of a sequence stride. Before SFPTRANSP one LREG holds 4 consecutive positions.
constexpr std::uint32_t lreg_stride(Stride stride)
{
    const std::uint32_t positions = static_cast<std::uint32_t>(stride);
    return is_after_transpose(stride) ? positions : positions / 4;
}

// LREG index bit that reverses a pair's direction (0: none).
constexpr std::uint32_t lreg_flip_bit(Stride stride, WindowSize window)
{
    if (window == WindowSize::TWO || (window == WindowSize::EIGHT && !is_after_transpose(stride)))
    {
        return 2;
    }
    if (window == WindowSize::SIXTEEN)
    {
        return 4;
    }
    return 0;
}

// SFPSWAP mode: after SFPTRANSP, windows of 4 and 8 alternate across the rows of each LREG.
constexpr std::uint32_t swap_mode(Stride stride, WindowSize window)
{
    if (is_after_transpose(stride) && window == WindowSize::FOUR)
    {
        return p_sfpswap::ROWS_02_MAX;
    }
    if (is_after_transpose(stride) && window == WindowSize::EIGHT)
    {
        return p_sfpswap::ROWS_01_MAX;
    }
    return p_sfpswap::ALL_ROWS_MAX;
}

template <Stride STRIDE, WindowSize WINDOW_SIZE, Order ORDER, UseLoadMacro USE_LOADMACRO, std::uint32_t PAIR>
inline __attribute__((always_inline)) void element_swap_pair()
{
    constexpr std::uint32_t stride = lreg_stride(STRIDE);
    constexpr std::uint32_t lo     = (PAIR / stride) * 2 * stride + PAIR % stride;
    constexpr std::uint32_t hi     = lo + stride;
    constexpr std::uint32_t flip   = lreg_flip_bit(STRIDE, WINDOW_SIZE);
    // VD gets the max; ASCENDING gives it to the lower LREG unless this pair is flipped.
    constexpr bool lo_is_vd    = (ORDER == Order::ASCENDING) != ((lo & flip) != 0);
    constexpr std::uint32_t vc = lo_is_vd ? hi : lo;
    constexpr std::uint32_t vd = USE_LOADMACRO == UseLoadMacro::Yes ? 12 + PAIR : (lo_is_vd ? lo : hi);
    TTI_SFPSWAP(0, vc, vd, swap_mode(STRIDE, WINDOW_SIZE));
}

template <Stride STRIDE, WindowSize WINDOW_SIZE, Order ORDER, UseLoadMacro USE_LOADMACRO, std::uint32_t... PAIRS>
inline __attribute__((always_inline)) void element_swap_pairs(std::integer_sequence<std::uint32_t, PAIRS...>)
{
    (element_swap_pair<STRIDE, WINDOW_SIZE, ORDER, USE_LOADMACRO, PAIRS>(), ...);
}

/**
 * @brief Issue one bitonic compare-exchange layer across LREG0-7.
 *
 * @tparam STRIDE: Distance between compared elements, values = <ONE/TWO/FOUR/EIGHT/SIXTEEN>
 * @tparam WINDOW_SIZE: Length of one sorted window, values = <TWO/FOUR/EIGHT/SIXTEEN/THIRTY_TWO>
 * @tparam ORDER: Direction of the first window, values = <ASCENDING/DESCENDING>
 * @tparam FUSED: Swap all of LREG0-7 (4 pairs) or only LREG0-3 (2 pairs), values = <No/Yes>
 * @tparam USE_LOADMACRO: Install the swaps as SFPLOADMACRO templates instead of issuing them, values = <No/Yes>
 * @note With UseLoadMacro::Yes, pair k is written to InstructionTemplate[k] holding only VC; the
 *       matching SFPLOADMACRO k must load the pair's other LREG, which becomes VD.
 */
template <Stride STRIDE, WindowSize WINDOW_SIZE, Order ORDER, Fused FUSED = Fused::Yes, UseLoadMacro USE_LOADMACRO = UseLoadMacro::No>
inline __attribute__((always_inline)) void element_swap()
{
    static_assert(2 * static_cast<std::uint32_t>(STRIDE) <= static_cast<std::uint32_t>(WINDOW_SIZE), "STRIDE must be at most half of WINDOW_SIZE");
    static_assert(FUSED == Fused::Yes || lreg_stride(STRIDE) < 4, "Unfused swaps stay within LREG0-3");
    constexpr std::uint32_t pairs = FUSED == Fused::Yes ? 4 : 2;
    element_swap_pairs<STRIDE, WINDOW_SIZE, ORDER, USE_LOADMACRO>(std::make_integer_sequence<std::uint32_t, pairs> {});
}

template <std::uint32_t BASE, std::uint32_t... FACE>
inline __attribute__((always_inline)) void transpose_32b_dest_faces_impl(std::integer_sequence<std::uint32_t, FACE...>)
{
    (transpose_dest_face_32b<BASE + FACE_INC * FACE>(), ...);
}

// Transpose NUMBER_OF_FACES consecutive faces of 32-bit Dst data, starting at `BASE`.
template <std::uint32_t BASE, std::uint32_t NUMBER_OF_FACES>
inline __attribute__((always_inline)) void transpose_32b_dest_faces()
{
    transpose_32b_dest_faces_impl<BASE>(std::make_integer_sequence<std::uint32_t, NUMBER_OF_FACES> {});
}

} // namespace detail

// The Blaze pipeline owns SrcB validity, destination zero flags, and the shared
// SFPU/replay state across MATH and PACK. Select this protocol consistently on
// all three TRISCs. The default remains the standalone Metal protocol.
#ifdef TOPK_XL_BLAZE_COMPAT
constexpr bool topk_xl_blaze_compat = true;
#else
constexpr bool topk_xl_blaze_compat = false;
#endif

// =============================================================================
//  Load / store helpers
// =============================================================================
//
// All load/store helpers fill or drain 8 LREGs (LREG0..LREG7) per call.
// The "x2" variants split the 8 LREGs into two groups of 4 separated by
// `group_2_offset` so a single call covers two interleaved 4-row strips
// (the natural unit at every level of the bitonic sort).
//
// The fused / unfused split:
//   * `load16_rows_x2 / store16_rows_x2` — fused path. All 8 LREGs hold the
//     packed [value | index] INT32 payload.
//   * `load8_rows_x2_unfused / store8_rows_x2_unfused` — unfused path. The
//     first 4 LREGs hold FP32 values; the last 4 hold the matching INT32
//     indices read from a parallel DST region at `indices_offset`.
//
// `inc_dst_addr` selects which ADDR_MOD the final SFPSTORE uses, so the
// caller can fold a trailing INCRWC into the last store. See "Address-mod
// recipe" at the top of the file.

// set_dst_write_addr_offset (defined in ckernel_sfpu_set_dst_write_addr_offset.h,
// shared with deepseek_top32_rm) rebases the Dst write pointer for subsequent
// SFPSTOREs; here it switches between the even and odd columns of the two-tile
// DST region (offsets +0 and +2 from the tile base).

/**
 * @brief Load two 16-row groups into LREG0-7 (fused path): LREG0-3 from base + {0, 4, 8, 12}, LREG4-7 from
 * base + group_2_offset + {0, 4, 8, 12}.
 *
 * @tparam group_2_offset: Dest distance of the second group. An int rather than @ref detail::GroupOffset because the
 *         merges load runs whole tiles apart (any multiple of 64 below 1024)
 */
template <int group_2_offset = 16>
inline void load16_rows_x2()
{
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 8);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 12);
}

/**
 * @brief Store LREG0-6 the way @ref store16_rows_x2 does, leaving out the LREG7 store.
 *
 * The LREG7 store is the one that carries the Dst advance, so callers that need a different advance per iteration
 * (the rebuild's split-record stride-32 / stride-16 loops) replay these 7 stores and issue that store inline.
 *
 * @tparam group_2_offset: Dest distance of the second group, as in @ref load16_rows_x2
 */
template <int group_2_offset = 16>
inline void store_first_7_rows_x2()
{
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, group_2_offset + 8);
}

/**
 * @brief Store LREG0-7 back to Dst (fused path), the mirror of @ref load16_rows_x2.
 *
 * @tparam group_2_offset: Dest distance of the second group, as in @ref load16_rows_x2
 * @tparam advance: Dst advance folded into the last store, values = <NONE/ONE_FACE/TWO_FACES/THREE_FACES>
 */
template <int group_2_offset = 16, detail::DstAdvance advance = detail::DstAdvance::NONE>
inline void store16_rows_x2()
{
    store_first_7_rows_x2<group_2_offset>();
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, detail::dst_advance_addr_mod<static_cast<int>(advance)>(), group_2_offset + 12);
}

/**
 * @brief Store only LREG0-3 (offsets 0, 4, 8, 12): the per-pair max half of a fused merge.
 *
 * `_topk_xl_merge_` leaves the per-pair min in LREG4-7, which is dead by construction: the rebuild only reads the value
 * faces at offsets 0..112 (Dst tiles 0..1), and the next merge stage overwrites tiles 2..3 via
 * `topk_xl_copy_tile(recv, 2, ...)`. Skipping those 4 stores saves 4 instructions per merge body (20 → 16).
 *
 * @tparam inc_dst_addr: Dst advance folded into the LREG3 store, see @ref detail::dst_advance_addr_mod
 */
template <int inc_dst_addr = 0>
inline void store4_rows_top_only()
{
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, detail::dst_advance_addr_mod<inc_dst_addr>(), 12);
}

/**
 * @brief Load two 8-row groups for the unfused path: FP32 values into LREG0-3, their INT32 indices from the parallel
 * Dst region into LREG4-7.
 *
 * @tparam indices_offset: Dest distance from the value region to the index region
 * @tparam group_2_offset: Dest distance of the second group
 */
template <int indices_offset = 256, int group_2_offset = 16>
inline void load8_rows_x2_unfused()
{
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, 4);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + group_2_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + group_2_offset + 4);
}

/**
 * @brief Store LREG0-6 the way @ref store8_rows_x2_unfused does, leaving out the LREG7 store (see
 * @ref store_first_7_rows_x2).
 *
 * @tparam indices_offset: Dest distance from the value region to the index region
 * @tparam group_2_offset: Dest distance of the second group
 */
template <int indices_offset = 256, int group_2_offset = 16>
inline void store_first_7_rows_x2_unfused()
{
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, 4);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + group_2_offset + 0);
}

/**
 * @brief Store LREG0-7 back for the unfused path, the mirror of @ref load8_rows_x2_unfused.
 *
 * @tparam indices_offset: Dest distance from the value region to the index region
 * @tparam group_2_offset: Dest distance of the second group
 * @tparam inc_dst_addr: Dst advance folded into the LREG7 (index) store, see @ref detail::dst_advance_addr_mod
 */
template <int indices_offset = 256, int group_2_offset = 16, int inc_dst_addr = 0>
inline void store8_rows_x2_unfused()
{
    store_first_7_rows_x2_unfused<indices_offset, group_2_offset>();
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, detail::dst_advance_addr_mod<inc_dst_addr>(), indices_offset + group_2_offset + 4);
}

// =============================================================================
//  Window sort
// =============================================================================
//
// One bitonic window inside LREG0-7: the compare-exchange steps at strides FROM, FROM/2, ..., TO, each one
// `detail::element_swap` layer (4 SFPSWAPs fused, 2 unfused). WindowSize sets the direction pattern (which pairs
// give the max to the lower position), Stride sets which LREGs pair up. Strides 16, 8 and 4 compare LREGs in the
// layout SFPLOAD leaves; strides 2 and 1 need the SFPTRANSP layout, so they run between two SFPTRANSPs (T):
//
//     WINDOW      FROM -> TO    emits
//     THIRTY_TWO  16 -> 1       16 8 4 · T · 2 1 · T
//     THIRTY_TWO   8 -> 1          8 4 · T · 2 1 · T
//     SIXTEEN      8 -> 1          8 4 · T · 2 1 · T
//     EIGHT        4 -> 1            4 · T · 2 1 · T
//     FOUR         2 -> 1                T · 2 1 · T
//     TWO          1 -> 1                T · 1   · T
//     THIRTY_TWO  16 -> 16      16
//     THIRTY_TWO   8 -> 4          8 4
//
// A full sort of 32 runs the windows TWO, FOUR, EIGHT, SIXTEEN, THIRTY_TWO in that order. A call can start below
// WINDOW/2 when the larger strides already ran elsewhere (SFPLOADMACRO-hosted first layer, Dest strips), and stop
// above 1 for a single cross-strip layer or a MOP-split body. See @ref detail::Transposes for leaving out the
// leading or trailing T.

namespace detail
{

constexpr std::uint32_t log2_u32(std::uint32_t x)
{
    return x <= 1 ? 0 : 1 + log2_u32(x >> 1);
}

constexpr bool skips_leading(Transposes transposes)
{
    return transposes == Transposes::NO_LEADING || transposes == Transposes::NONE;
}

constexpr bool skips_trailing(Transposes transposes)
{
    return transposes == Transposes::NO_TRAILING || transposes == Transposes::NONE;
}

// Compile-time shape of one window_sort call: validates the arguments and counts what it emits.
template <WindowSize WINDOW, Stride FROM, Stride TO, Transposes TRANSPOSES, Fused FUSED>
struct WindowSortShape
{
    static constexpr std::uint32_t window = static_cast<std::uint32_t>(WINDOW);
    static constexpr std::uint32_t from   = static_cast<std::uint32_t>(FROM);
    static constexpr std::uint32_t to     = static_cast<std::uint32_t>(TO);

    static_assert(2 * from <= window, "FROM must be at most half of WINDOW: both elements of a pair have to sit in one window");
    static_assert(to <= from, "TO must not exceed FROM: the steps run from the larger stride down");

    // T · 2 1 · T: the leading T is the call's first instruction only when it starts at stride 2 or 1.
    static constexpr bool has_leading  = from <= 2;
    static constexpr bool has_trailing = to <= 2;
    static_assert(
        has_leading || !skips_leading(TRANSPOSES),
        "NO_LEADING / NONE need a leading SFPTRANSP, so the call must start at stride 2 or 1 (T · 2 1 · T); from stride 4 or above "
        "the T before stride 2 sits between steps (16 8 4 · T · 2 1 · T) and is always emitted");
    static_assert(
        has_trailing || !skips_trailing(TRANSPOSES), "NO_TRAILING / NONE need a trailing SFPTRANSP, so the call must end at stride 2 or 1 (... 2 1 · T)");

    static constexpr std::uint32_t steps = log2_u32(from) - log2_u32(to) + 1;
    // The T into the transposed layout exists whenever stride 2 or 1 runs; it is the leading one when FROM <= 2.
    static constexpr std::uint32_t transposes =
        (to <= 2 ? 1u : 0u) - (has_leading && skips_leading(TRANSPOSES) ? 1u : 0u) + (has_trailing && !skips_trailing(TRANSPOSES) ? 1u : 0u);
    static constexpr std::uint32_t length = steps * (FUSED == Fused::Yes ? 4u : 2u) + transposes;
};

// One step at stride S; the SFPTRANSP into the transposed layout goes before the first of stride 2 or 1 the call runs.
template <WindowSize WINDOW, Order ORDER, Transposes TRANSPOSES, Fused FUSED, std::uint32_t FROM, std::uint32_t S>
inline void window_sort_step()
{
    if constexpr (S == 2 || (S == 1 && FROM == 1))
    {
        if constexpr (!(S == FROM && skips_leading(TRANSPOSES)))
        {
            TTI_SFPTRANSP(0, 0, 0, 0);
        }
    }
    element_swap<static_cast<Stride>(S), WINDOW, ORDER, FUSED>();
}

template <WindowSize WINDOW, Order ORDER, Transposes TRANSPOSES, Fused FUSED, std::uint32_t FROM, std::uint32_t... K>
inline void window_sort_steps(std::integer_sequence<std::uint32_t, K...>)
{
    (window_sort_step<WINDOW, ORDER, TRANSPOSES, FUSED, FROM, (FROM >> K)>(), ...);
}

} // namespace detail

/**
 * @brief Run the bitonic steps of one window inside LREG0-7, strides FROM down to TO, with the SFPTRANSPs they need.
 *
 * Strides 16, 8 and 4 run in the loaded layout; strides 2 and 1 run transposed, wrapped as T · 2 1 · T. Examples:
 * <THIRTY_TWO, SIXTEEN, ONE> emits 16 8 4 · T · 2 1 · T, <EIGHT, TWO, ONE> emits T · 2 1 · T, <THIRTY_TWO, SIXTEEN,
 * SIXTEEN> emits 16.
 *
 * @tparam WINDOW: Length of one sorted window, sets the direction pattern, values = <TWO/FOUR/EIGHT/SIXTEEN/THIRTY_TWO>
 * @tparam FROM: First stride, at most WINDOW / 2, values = <ONE/TWO/FOUR/EIGHT/SIXTEEN>
 * @tparam TO: Last stride, at most FROM, values = <ONE/TWO/FOUR/EIGHT/SIXTEEN>
 * @tparam ORDER: Direction of the first window, values = <ASCENDING/DESCENDING>
 * @tparam TRANSPOSES: Which of the leading / trailing SFPTRANSP to emit, values = <BOTH/NO_LEADING/NO_TRAILING/NONE>
 * @tparam FUSED: Swap LREG0-7 (4 swaps per step) or only LREG0-3 (2 swaps per step), values = <Yes/No>
 * @note Combinations without the SFPTRANSP that TRANSPOSES leaves out do not compile; see @ref detail::Transposes.
 *       Emits @ref window_sort_len instructions.
 */
template <
    detail::WindowSize WINDOW,
    detail::Stride FROM,
    detail::Stride TO,
    detail::Order ORDER,
    detail::Transposes TRANSPOSES = detail::Transposes::BOTH,
    detail::Fused FUSED           = detail::Fused::Yes>
inline void window_sort()
{
    using shape = detail::WindowSortShape<WINDOW, FROM, TO, TRANSPOSES, FUSED>;
    detail::window_sort_steps<WINDOW, ORDER, TRANSPOSES, FUSED, shape::from>(std::make_integer_sequence<std::uint32_t, shape::steps> {});
    if constexpr (shape::has_trailing && !detail::skips_trailing(TRANSPOSES))
    {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

/**
 * @brief @ref window_sort with ORDER chosen at runtime.
 *
 * @param ascending: ORDER as a runtime value, true = ASCENDING
 */
template <
    detail::WindowSize WINDOW,
    detail::Stride FROM,
    detail::Stride TO,
    detail::Transposes TRANSPOSES = detail::Transposes::BOTH,
    detail::Fused FUSED           = detail::Fused::Yes>
inline void window_sort(const bool ascending)
{
    if (ascending)
    {
        window_sort<WINDOW, FROM, TO, detail::Order::ASCENDING, TRANSPOSES, FUSED>();
    }
    else
    {
        window_sort<WINDOW, FROM, TO, detail::Order::DESCENDING, TRANSPOSES, FUSED>();
    }
}

/**
 * @brief Number of Tensix instructions the matching @ref window_sort emits, for replay planning.
 *
 * steps × (4 fused / 2 unfused) + the SFPTRANSPs emitted, e.g. 22 for <THIRTY_TWO, SIXTEEN, ONE>.
 */
template <
    detail::WindowSize WINDOW,
    detail::Stride FROM,
    detail::Stride TO,
    detail::Transposes TRANSPOSES = detail::Transposes::BOTH,
    detail::Fused FUSED           = detail::Fused::Yes>
constexpr std::uint32_t window_sort_len()
{
    return detail::WindowSortShape<WINDOW, FROM, TO, TRANSPOSES, FUSED>::length;
}

/**
 * @brief Sort the next 32 Dest rows of 8 columns (one per SFPU lane) into one run in direction `dir`, then advance
 * Dst by 32 rows.
 *
 * Loads the 32 rows into LREG0-7, runs window_sort over windows 2, 4, 8, 16 and 32 and stores them back:
 *
 *     load · T 1 · 2 1 T · 4 T 2 1 T · [8 4 T] [2 1 T] · 16 [8 4 T] [2 1 T] · store
 *              W2   W4       W8          W16                 W32
 *
 * In W16 and W32 the bracketed halves emit the same SFPSWAPs (after SFPTRANSP the same LREG pairs give strides 2 and 1
 * that gave 8 and 4 before it), so each second half replays the first. Replay slots:
 *
 *     [0, 14)   W8 sort      recorded by RECORD, replayed by REPLAY
 *     [14, 23)  W16 half     recorded by RECORD, replayed by every call for the second half
 *     [23, 32)  W32 half     recorded by every call, since `dir` changes between calls, then replayed once
 *
 * @tparam MODE: RECORD for the first block of a phase, REPLAY for the blocks after it, values = <RECORD/REPLAY>
 * @param ascending: ORDER of the W4 and W8 sorts; it does not change the sorted result
 * @param dir: Direction of the 32-row run, true = max first
 * @note Overwrites replay slots [0, 32). Call it with REPLAY only after a RECORD call with the same `ascending` and no
 *       other replay recording in between.
 */
template <detail::ReplayMode MODE>
inline void sort_32_rows(const bool ascending, const bool dir)
{
    using detail::Order;
    using detail::Stride;
    using detail::Transposes;
    using detail::WindowSize;
    constexpr std::uint32_t w8_len   = window_sort_len<WindowSize::EIGHT, Stride::FOUR, Stride::ONE>();
    constexpr std::uint32_t w16_half = window_sort_len<WindowSize::SIXTEEN, Stride::EIGHT, Stride::FOUR>() + 1;    // 8 4 · T
    constexpr std::uint32_t w32_half = window_sort_len<WindowSize::THIRTY_TWO, Stride::EIGHT, Stride::FOUR>() + 1; // 8 4 · T
    constexpr std::uint32_t w16_slot = w8_len;
    constexpr std::uint32_t w32_slot = w16_slot + w16_half;
    static_assert(w32_slot + w32_half <= 32, "the W8 sort and the W16 and W32 halves must fit the 32-slot replay buffer");

    load16_rows_x2<16>();
    window_sort<WindowSize::TWO, Stride::ONE, Stride::ONE, Order::DESCENDING, Transposes::NO_TRAILING>();
    window_sort<WindowSize::FOUR, Stride::TWO, Stride::ONE, Transposes::NO_LEADING>(ascending);
    if constexpr (MODE == detail::ReplayMode::RECORD)
    {
        load_replay_buf<Exec>(
            0,
            w32_slot,
            [ascending]
            {
                window_sort<WindowSize::EIGHT, Stride::FOUR, Stride::ONE>(ascending);
                window_sort<WindowSize::SIXTEEN, Stride::EIGHT, Stride::FOUR, Order::DESCENDING>();
                TTI_SFPTRANSP(0, 0, 0, 0);
            });
    }
    else
    {
        lltt::replay(0, w32_slot);
    }
    lltt::replay(w16_slot, w16_half);
    window_sort<WindowSize::THIRTY_TWO, Stride::SIXTEEN, Stride::SIXTEEN>(dir);
    load_replay_buf<Exec>(
        w32_slot,
        w32_half,
        [dir]
        {
            window_sort<WindowSize::THIRTY_TWO, Stride::EIGHT, Stride::FOUR>(dir);
            TTI_SFPTRANSP(0, 0, 0, 0);
        });
    lltt::replay(w32_slot, w32_half);
    store16_rows_x2<16, detail::DstAdvance::TWO_FACES>();
}

namespace detail
{

// A W16 or W32 sort that runs from stride 8 or above down to 1 ends in 8 4 · T · 2 1 · T, and the two halves emit
// the same SFPSWAPs (after SFPTRANSP the LREG pairs of strides 8 and 4 give strides 2 and 1).
template <WindowSize WINDOW, Stride FROM, Stride TO>
constexpr bool has_repeated_half = static_cast<std::uint32_t>(WINDOW) >= 16 && static_cast<std::uint32_t>(FROM) >= 8 && TO == Stride::ONE;

/**
 * @brief Run `iterations` × (load · window_sort<WINDOW, FROM, TO>(dir) · store) from the replay buffer.
 *
 * The first iteration records the body while executing it; the others replay it. The layout is chosen at compile time:
 *
 *     body fits 32 slots:            [load][sort][store]                         1 issue per later iteration
 *     otherwise, a repeated half:    [load][strides above 8][8 4 · T][store]     2 issues per later iteration
 *
 * In the second layout each iteration replays [load .. 8 4 · T], then [8 4 · T][store] again, which issues 2 1 · T.
 *
 * @tparam LOAD_LEN: Instructions `load` emits
 * @tparam STORE_LEN: Instructions `store` emits
 * @tparam MODE: RECORD to record the body on the first iteration, REPLAY to replay every iteration from a RECORD call
 *               with the same template arguments and `dir`, values = <RECORD/REPLAY>
 * @param iterations: Bodies to run, at least 1
 * @param dir: Direction of every window_sort; it must not change between iterations
 * @note Overwrites the replay slots it uses, from slot 0.
 */
template <
    WindowSize WINDOW,
    Stride FROM,
    Stride TO,
    std::uint32_t LOAD_LEN,
    std::uint32_t STORE_LEN,
    Fused FUSED     = Fused::Yes,
    ReplayMode MODE = ReplayMode::RECORD,
    typename Load,
    typename Store>
inline void window_sort_replay_loop(const int iterations, const bool dir, Load load, Store store)
{
    constexpr std::uint32_t whole = LOAD_LEN + window_sort_len<WINDOW, FROM, TO, Transposes::BOTH, FUSED>() + STORE_LEN;
    constexpr int first_replay    = MODE == ReplayMode::RECORD ? 1 : 0;
    if constexpr (whole <= 32)
    {
        if constexpr (MODE == ReplayMode::RECORD)
        {
            load_replay_buf<Exec>(
                0 /*start*/,
                whole,
                [&]
                {
                    load();
                    window_sort<WINDOW, FROM, TO, Transposes::BOTH, FUSED>(dir);
                    store();
                });
        }
        for (int i = first_replay; i < iterations; i++)
        {
            lltt::replay(0 /*start*/, whole);
        }
    }
    else
    {
        static_assert(has_repeated_half<WINDOW, FROM, TO>, "the body does not fit 32 replay slots and has no repeated half to replay");
        constexpr std::uint32_t half      = window_sort_len<WINDOW, Stride::EIGHT, Stride::FOUR, Transposes::BOTH, FUSED>() + 1; // 8 4 · T
        constexpr std::uint32_t first     = LOAD_LEN + window_sort_len<WINDOW, FROM, Stride::FOUR, Transposes::BOTH, FUSED>() + 1;
        constexpr std::uint32_t half_slot = first - half;
        static_assert(first + STORE_LEN <= 32, "load, the sort up to its first half and store must fit 32 replay slots");
        if constexpr (MODE == ReplayMode::RECORD)
        {
            load_replay_buf<Exec>(
                0 /*start*/,
                first,
                [&]
                {
                    load();
                    window_sort<WINDOW, FROM, Stride::FOUR, Transposes::BOTH, FUSED>(dir);
                    TTI_SFPTRANSP(0, 0, 0, 0);
                });
            lltt::replay(half_slot, half);
            load_replay_buf<Exec>(first, STORE_LEN, [&] { store(); });
        }
        for (int i = first_replay; i < iterations; i++)
        {
            lltt::replay(0 /*start*/, first);
            lltt::replay(half_slot, half + STORE_LEN);
        }
    }
}

} // namespace detail

/**
 * @brief Run `iterations` × (load · window_sort<WINDOW, FROM, TO>(dir) · store), recorded once and replayed after.
 *
 * Fused: load16_rows_x2 / store16_rows_x2, each LREG holding values with their packed indices.
 * Unfused: load8_rows_x2_unfused / store8_rows_x2_unfused, values in LREG0-3 and their indices, read from INDICES
 * further on in Dest, in LREG4-7.
 *
 * @tparam GROUP_OFFSET: Dest distance between the two row groups of a load (fused LREG0-3 to LREG4-7, unfused LREG0-1
 *                       to LREG2-3), values = <ODD_COLUMNS/HALF_FACE/ONE_FACE/TWO_FACES/FOUR_FACES>
 * @tparam ADVANCE: Dst advance folded into the last store, values = <NONE/HALF_FACE/ONE_FACE/TWO_FACES/THREE_FACES>
 * @tparam FUSED: Fused or unfused layout, values = <Yes/No>
 * @tparam INDICES: Dest distance from the value region to the index region, unfused only, values =
 *                  <NONE/ONE_TILE/TWO_TILES/FOUR_TILES>
 * @tparam MODE: see @ref detail::window_sort_replay_loop, values = <RECORD/REPLAY>
 * @param iterations: Bodies to run, at least 1
 * @param dir: Direction of every window_sort; it must not change between iterations
 * @note See @ref detail::window_sort_replay_loop for the replay layout. Overwrites the replay slots it uses, from 0.
 */
template <
    detail::WindowSize WINDOW,
    detail::Stride FROM,
    detail::Stride TO,
    detail::GroupOffset GROUP_OFFSET,
    detail::DstAdvance ADVANCE,
    detail::Fused FUSED,
    detail::IndicesOffset INDICES = detail::IndicesOffset::NONE,
    detail::ReplayMode MODE       = detail::ReplayMode::RECORD>
inline void window_sort_rows(const int iterations, const bool dir)
{
    static_assert((FUSED == detail::Fused::Yes) == (INDICES == detail::IndicesOffset::NONE), "only the unfused layout has a separate index region");
    constexpr int group_offset          = static_cast<int>(GROUP_OFFSET);
    constexpr int indices_offset        = static_cast<int>(INDICES);
    constexpr std::uint32_t rows_x2_len = 8; // every load / store helper here: one SFPLOAD / SFPSTORE per LREG
    if constexpr (FUSED == detail::Fused::Yes)
    {
        detail::window_sort_replay_loop<WINDOW, FROM, TO, rows_x2_len, rows_x2_len, FUSED, MODE>(
            iterations, dir, [] { load16_rows_x2<group_offset>(); }, [] { store16_rows_x2<group_offset, ADVANCE>(); });
    }
    else
    {
        detail::window_sort_replay_loop<WINDOW, FROM, TO, rows_x2_len, rows_x2_len, FUSED, MODE>(
            iterations,
            dir,
            [] { load8_rows_x2_unfused<indices_offset, group_offset>(); },
            [] { store8_rows_x2_unfused<indices_offset, group_offset, static_cast<int>(ADVANCE)>(); });
    }
}

// =============================================================================
//  SFPLOADMACRO
// =============================================================================
//
// The macro configuration (`loadmacro_configure`) and the load / sort / store bodies that ride it, for the
// unfused and the fused paths.

// -----------------------------------------------------------------------------
//  Configuration
// -----------------------------------------------------------------------------

#if TOPK_XL_UNFUSED_MACRO || TOPK_XL_FUSED_MACRO

// Sequence and Misc encodings shared by the unfused and the fused macros.
namespace detail
{

// Simple byte: selector 4+m -> InstructionTemplate[m], delay 0. 0x80 SET ->
// Insn.VB = macroVD, leaving Insn.VC at the template's value (with it clear
// the macro would assign Insn.VC = macroVD and the SFPSWAP degenerates to a
// same-register compare — which silicon resolves as garbage, not a no-op).
// 0x40 CLEAR -> Insn.VD = macroVD.
constexpr std::uint32_t seq_simple(std::uint32_t m, std::uint32_t delay = 0)
{
    return 0x80u | (delay << 3) | (4u + m);
}

// MAD byte: SFPNOP at the same delay — required whenever SFPSWAP is scheduled
// to the Simple sub-unit (SFPLOADMACRO.md:11 footnote ‡). Round: nothing.
constexpr std::uint32_t seq_mad_nop(std::uint32_t delay = 0)
{
    return (delay << 3) | 2u;
}

// Store byte: selector 3 = the built-in SFPSTORE, delay 2 — the SFPSWAP
// writes macroVD on its second cycle (macro+2), the store fires at macro+3.
// 0x40/0x80 clear -> Insn.VD = macroVD.
constexpr std::uint32_t seq_store(std::uint32_t delay = 2)
{
    return (delay << 3) | 3u;
}

// Misc (SFPLOADMACRO.md:53-57): 0xF0 -> every macro's store inherits its
// LOAD's Mod0 (FP32 here — the value regs are genuine FP32, same mode the
// shipping software stores use). 0xB00 -> Simple/MAD/Store on
// WaitForElapsedInstructions so the delay chain cannot slide if the frontend
// bubbles.
constexpr std::uint32_t MISC_WORD = 0xB00u | 0xF0u;

// TTI_SFPCONFIG(imm16, dest, mod1) fields (SFPCONFIG.md):
//   dest  — target config register: SequenceReg[i] sits at 4+i (:52-55),
//           the Misc register at 8 (:56).
//   mod1  — 0: load the 32-bit word staged in LReg[0] into `dest` (:100);
//           SFPCFG_IMM16_IS_VALUE (1): write imm16 directly (:108).
constexpr std::uint32_t SFPCFG_DEST_SEQUENCE_BASE = 4; // dest of SequenceReg[0]; [1] at +1
constexpr std::uint32_t SFPCFG_DEST_MISC          = 8;
constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE     = 1; // SFPCONFIG.md:108

/**
 * @brief Sequence word of SFPLOADMACRO m.
 *
 * Schedules InstructionTemplate[m] on Simple, an SFPNOP on MAD at the same delay, and a store of macroVD back to its
 * load address. Fused: swap delay m and store delays {2, 4, 5, 6}, so four back-to-back macros swap one per two slots
 * and each store lands after its swap has written. Unfused: swap delay 0 and store delay 2.
 *
 * @tparam FUSED: Four fused macros or two unfused ones, values = <Yes/No>
 * @param m: Macro index, 0-3 fused, 0-1 unfused
 * @return The 32-bit word for SequenceReg[m]; 0 (schedule nothing) under the test-only mutation hooks.
 */
template <Fused FUSED>
constexpr std::uint32_t sequence_word([[maybe_unused]] std::uint32_t m)
{
    if constexpr (FUSED == Fused::Yes)
    {
#if defined(TOPK_XL_FUSED_MACRO_MUTATE)
        return 0u;
#else
        return (seq_store(m == 0 ? 2 : m + 3) << 24) | (seq_mad_nop(m) << 8) | seq_simple(m, m);
#endif
    }
    else
    {
#if defined(TOPK_XL_MACRO_MUTATE)
        return 0u;
#else
        return (seq_store() << 24) | (seq_mad_nop() << 8) | seq_simple(m);
#endif
    }
}

/**
 * @brief Write SequenceReg[M...] from @ref sequence_word.
 *
 * Each word is staged through LREG0, since the Store byte does not fit the imm16 path.
 *
 * @tparam FUSED: Four fused macros or two unfused ones, values = <Yes/No>
 * @tparam M: Macro indices to program
 */
template <Fused FUSED, std::uint32_t... M>
inline void program_sequences(std::integer_sequence<std::uint32_t, M...>)
{
    (_sfpu_load_config32_(SFPCFG_DEST_SEQUENCE_BASE + M, upper16(sequence_word<FUSED>(M)), lower16(sequence_word<FUSED>(M))), ...);
}

/**
 * @brief Write every Sequence word and Misc, then wait until an SFPLOADMACRO can read them.
 *
 * @tparam FUSED: Four fused macros or two unfused ones, values = <Yes/No>
 * @note Clobbers LREG0.
 */
template <Fused FUSED>
inline void program_sequences_and_misc()
{
    constexpr std::uint32_t macros = FUSED == Fused::Yes ? 4 : 2;
    program_sequences<FUSED>(std::make_integer_sequence<std::uint32_t, macros> {});
    TTI_SFPCONFIG(MISC_WORD, SFPCFG_DEST_MISC, SFPCFG_IMM16_IS_VALUE);
    // An SFPLOADMACRO issued sooner reads the stale Sequence/Misc. Two NOPs is the measured minimum on BH silicon:
    // the mutation-control suite diverges with either one removed.
    TTI_SFPNOP;
    TTI_SFPNOP;
}

} // namespace detail

/**
 * @brief Program SFPLOADMACRO to host one compare-exchange layer: its swap templates, the Sequence words and Misc.
 *
 * Template m holds pair m of the layer with only VC; macro m loads the pair's other LREG, which becomes VD and
 * receives the max.
 *
 * @tparam STRIDE: Distance between compared elements, values = <FOUR/EIGHT/SIXTEEN>
 * @tparam WINDOW_SIZE: Length of one sorted window, values = <EIGHT/SIXTEEN/THIRTY_TWO>
 * @tparam ORDER: Direction of the first window, values = <ASCENDING/DESCENDING>
 * @tparam FUSED: Four macros over LREG0-7, or two over LREG0-3 with indices in LREG4-7, values = <Yes/No>
 * @note Clobbers LREG0. Templates and Sequence words are shared by the three Tensix threads: no other thread may
 *       issue SFPLOADMACRO while they are reprogrammed.
 */
template <detail::Stride STRIDE, detail::WindowSize WINDOW_SIZE, detail::Order ORDER, detail::Fused FUSED>
inline void loadmacro_configure()
{
    detail::element_swap<STRIDE, WINDOW_SIZE, ORDER, FUSED, detail::UseLoadMacro::Yes>();
    detail::program_sequences_and_misc<FUSED>();
}

/**
 * @brief @ref loadmacro_configure with ORDER chosen at runtime; only the templates depend on it.
 *
 * @param ascending: ORDER as a runtime value, true = ASCENDING
 */
template <detail::Stride STRIDE, detail::WindowSize WINDOW_SIZE, detail::Fused FUSED>
inline void loadmacro_configure(const bool ascending)
{
    if (ascending)
    {
        detail::element_swap<STRIDE, WINDOW_SIZE, detail::Order::ASCENDING, FUSED, detail::UseLoadMacro::Yes>();
    }
    else
    {
        detail::element_swap<STRIDE, WINDOW_SIZE, detail::Order::DESCENDING, FUSED, detail::UseLoadMacro::Yes>();
    }
    detail::program_sequences_and_misc<FUSED>();
}

#endif

// -----------------------------------------------------------------------------
//  Unfused bodies
// -----------------------------------------------------------------------------
//
// The unfused merge body and the unfused rebuild's stride-64/32/16 bodies all
// have the same single-level shape:
//
//     load8_rows_x2_unfused + bitonic_sort_len_k<false> + store8_rows_x2_unfused
//
// with every store going back to the address it was loaded from, so the two
// SFPSWAPs ride SFPLOADMACRO Simple slots (delay 0) and the store of each macro's own
// register rides its Store slot (delay 2, address latched at load time —
// SFPLOADMACRO.md:140). Under SFPU index-tracking mode (LaneConfig bit [2],
// which the unfused path enables) a macro-scheduled SFPSWAP still performs the
// companion swap of LReg[4+VC] <-> LReg[4+VD].
//
// The macro body is 16 issues, against 18 issues plus two 2-cycle SFPSWAPs
// (and their index-tracking stalls) without macros:
//
//     i0..i3   4x SFPLOAD  (INT32)  index regs LREG4..7 — MUST precede the
//                                    macros: the companion swap reads/writes
//                                    LREG4..7 the cycle after each macro.
//     i4       SFPLOAD  (FP32)      first VC provider
//     i5       SFPLOADMACRO m0      loads first macroVD; Simple: SFPSWAP,
//                                    MAD: SFPNOP (footnote ‡), Store: delay 2
//     i6       SFPLOAD  (FP32)      second VC provider  — the plain load
//                                    between macros IS the 2-slot separation
//                                    rule (a macro-scheduled SFPSWAP owns the
//                                    Simple sub-unit for 2 cycles and is
//                                    exempt from the automatic stall)
//     i7       SFPLOADMACRO m1
//     i8       SFPNOP               m0's store fires here — keep it store-free
//     i9       SFPSTORE (FP32)      first VC register
//     i10      SFPNOP               m1's store fires here
//     i11      SFPSTORE (FP32)      second VC register
//     i12..i14 3x SFPSTORE (INT32)  LREG4/5/6
//     i15      SFPSTORE (INT32)     LREG7 — carries the Dst-advance fold
//                                    (kept in software so every inc_dst_addr
//                                    variant works)
//
// Both scheduled stores retire INSIDE the body (delay 2 from i5/i7 fires at
// i8/i10), so no trailing SFPNOPs and no cross-call pending-store state.
//
// Direction maps to which half is macroVD (the macro always overrides Insn.VD
// with the register its own load writes):
//   ascending  (sort_k ASC:  SWAP(VC=LREG2, VD=LREG0)) -> macroVD = LREG0/1,
//              templates' VC = LREG2/3
//   descending (sort_k DESC: SWAP(VC=LREG0, VD=LREG2)) -> macroVD = LREG2/3,
//              templates' VC = LREG0/1
// The merge always takes the descending form (it calls sort_k with false).
// Merge and rebuild run `loadmacro_configure` at entry: the two
// InstructionTemplates for their direction plus Sequence[0/1] and Misc, so they
// carry no cross-call contract with the fused phases, which reprogram the same
// registers.
//
// Index loads/stores stay in software: only 4 macros exist, two of them carry
// the swaps, and LREG7's store carries the ADDR_MOD fold.
//
// TOPK_XL_MACRO_MUTATE is a test-only hook: it zeroes the Sequence words so
// every SFPLOADMACRO degenerates into a plain SFPLOAD, which silently drops the
// compare-exchanges and the macro stores. The correctness suites must catch it.

#if TOPK_XL_UNFUSED_MACRO

namespace topk_xl_unfused_macro
{

// SFPLOADMACRO field packing (ckernel_ops.h:689, SFPLOADMACRO.md:20-26,45):
//   lreg_ind = (MacroIndex << 2) | (VD & 3), dest_reg_addr = (Imm9 << 1) | (VD >> 2).
// VD is u3 and the address low bit is unused (SFPLOAD.md:83), so VD in 0..7
// never perturbs the address.
#define TOPK_XL_UNFUSED_LOADMACRO(macro_idx, vd, off) \
    TTI_SFPLOADMACRO(((macro_idx) << 2) | ((vd) & 3u), InstrModLoadStore::FP32, ADDR_MOD_7, (off) | ((vd) >> 2))

// The body minus the LREG7 tail — 15 issues (see the section comment's
// timeline). Kept separate so the split-record loops can emit a per-iter
// LREG7 ADDR_MOD exactly like the shipping `store_first_7_rows_x2_unfused`
// pattern.
template <int indices_offset, int group_2_offset, bool ascending>
inline void ce_first15()
{
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + group_2_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + group_2_offset + 4);
    if constexpr (ascending)
    {
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 0);
        TOPK_XL_UNFUSED_LOADMACRO(0u, p_sfpu::LREG0, 0);
        TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 4);
        TOPK_XL_UNFUSED_LOADMACRO(1u, p_sfpu::LREG1, 4);
        TTI_SFPNOP; // m0's store (LREG0 -> 0) fires here
        TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 0);
        TTI_SFPNOP; // m1's store (LREG1 -> 4) fires here
        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 4);
    }
    else
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
        TOPK_XL_UNFUSED_LOADMACRO(0u, p_sfpu::LREG2, group_2_offset + 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, 4);
        TOPK_XL_UNFUSED_LOADMACRO(1u, p_sfpu::LREG3, group_2_offset + 4);
        TTI_SFPNOP; // m0's store (LREG2 -> group_2_offset) fires here
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
        TTI_SFPNOP; // m1's store (LREG3 -> group_2_offset + 4) fires here
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, 4);
    }
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, indices_offset + group_2_offset + 0);
}

// The LREG7 tail — same inc_dst_addr -> ADDR_MOD mapping as
// `store8_rows_x2_unfused`'s last store, so every shipping fold survives.
template <int indices_offset, int group_2_offset, int inc_dst_addr>
inline void ce_tail()
{
    constexpr int off = indices_offset + group_2_offset + 4;
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, detail::dst_advance_addr_mod<inc_dst_addr>(), off);
}

// Full 16-issue body.
template <int indices_offset, int group_2_offset, int inc_dst_addr, bool ascending>
inline void ce_full()
{
    ce_first15<indices_offset, group_2_offset, ascending>();
    ce_tail<indices_offset, group_2_offset, inc_dst_addr>();
}

// Record the full body (recording IS the first executed iteration) into
// replay slots [0, 16) for the runtime direction. Fold iterations replay
// [0, 15) and emit their own tail.
template <int indices_offset, int group_2_offset, int inc_dst_addr>
inline void record_ce_full(const bool dir)
{
    detail::with_direction(
        dir,
        [](auto direction)
        { load_replay_buf<Exec>(0 /*start*/, 16 /*len*/, [] { ce_full<indices_offset, group_2_offset, inc_dst_addr, decltype(direction)::value>(); }); });
}

} // namespace topk_xl_unfused_macro

#endif // TOPK_XL_UNFUSED_MACRO

// -----------------------------------------------------------------------------
//  Fused bodies
// -----------------------------------------------------------------------------

#if TOPK_XL_FUSED_MACRO

// Macro i loads LREG4+i, swaps it with LREG i i instructions later and stores it back once the swap has written it;
// the thread loads and stores LREG0..3 in the slots the scheduled stores leave free.
//
// That describes `step_group` and the merge/rebuild bodies. `column_pair_len8_loads` / `column_pair_len16_loads`
// deliberately break it: they reprogram the four templates so a macro can host an arbitrary pair, and
// they use the `VDHi = 0` encoding so a macro can load LREG0..3 as well as LREG4..7.
namespace topk_xl_fused_macro
{

// Macro m loads row `far` + 4m into LREG4+m; its scheduled swap and store follow (see the Sequence words).
template <int far>
inline void far_strip_macros()
{
    TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::INT32, ADDR_MOD_7, (far + 0) | 1);
    TTI_SFPLOADMACRO((1 << 2) | 1, InstrModLoadStore::INT32, ADDR_MOD_7, (far + 4) | 1);
    TTI_SFPLOADMACRO((2 << 2) | 2, InstrModLoadStore::INT32, ADDR_MOD_7, (far + 8) | 1);
    TTI_SFPLOADMACRO((3 << 2) | 3, InstrModLoadStore::INT32, ADDR_MOD_7, (far + 12) | 1);
}

// load16_rows_x2, bitonic_sort_len_k and store16_rows_x2 in 15 instructions. The macro swaps have a fixed operand
// order, so an ascending group loads the second strip into LREG0..3 instead.
template <int group_2_offset, int inc_dst_addr, bool ascending>
inline void step_group()
{
    constexpr int near = ascending ? group_2_offset : 0;
    constexpr int far  = ascending ? 0 : group_2_offset;
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, near + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, near + 4);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, near + 8);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, near + 12);
    far_strip_macros<far>();
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, near + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, near + 4);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, near + 8);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, detail::dst_advance_addr_mod<inc_dst_addr>(), near + 12);
    TTI_SFPNOP;
}

// Software-pipelined `step_group`. This group's four plain SFPLOADs were issued by the previous
// iteration (or by `prologue_loads`), and this body prefetches the next group's, so the steady-state
// body costs 12 issue slots.
//
// The three SFPNOPs in `step_group` are not swap padding: they are the slots the macro's own scheduled
// stores land on. With the macros at slots 0..3 those stores fire at `trigger + delay + 1` = 3, 6, 8
// and 10 (store delays {2,4,5,6} from `sequence_word`). A regular instruction that collides with a
// scheduled one on the same sub-unit is silently discarded (SFPLOADMACRO.md:149), so the plain stores
// have to dodge those slots -- but the prefetch loads do not, because they occupy the load sub-unit
// rather than the store sub-unit (SFPLOADMACRO.md:13):
//
//   slot:        0   1   2   3   4   5   6   7   8   9  10  11
//   explicit:   M0  M1  M2  M3  S0  S1  P0  S2  P1  S3  P2  P3
//   macro store:            F0          F1      F2      F3
//
// `LREG_i` is consumable at 2i+3 = 3, 5, 7, 9, so the stores at 4, 5, 7 and 9 are all legal: a
// macro-scheduled SFPSWAP reads on its 1st cycle and writes on its 2nd and has no automatic stalling
// (SFPSWAP.md:115), so no extra margin is owed.
//
// S3 folds the Dst advance, so the prefetches issued ahead of it carry `inc_dst_addr` in their
// immediates and those after it do not. The pending macro stores are indifferent to that advance: a
// macro-scheduled SFPSTORE reuses the address computed at its SFPLOADMACRO, "regardless of whether
// SFPLOADMACRO (or any other intermediate instruction) advanced any RWCs" (SFPLOADMACRO.md:140).
template <int group_2_offset, int inc_dst_addr, bool ascending>
inline void step_group_pipelined()
{
    static_assert(inc_dst_addr != 0, "a group that does not advance Dst has no next group to prefetch");
    constexpr int near = ascending ? group_2_offset : 0;
    constexpr int far  = ascending ? 0 : group_2_offset;
    far_strip_macros<far>();
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, near + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, near + 4);
    // Prefetches ahead of the advancing store address the next group explicitly.
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, near + inc_dst_addr + 0);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, near + 8);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, near + inc_dst_addr + 4);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, detail::dst_advance_addr_mod<inc_dst_addr>(), near + 12);
    // Past the advance, so these address the next group implicitly.
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, near + 8);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, near + 12);
}

constexpr int pipelined_group_len = 12;

// ── Hosting the column-pair sort's first swap layer on the loads ────────────────────────────────────
//
// `_topk_xl_split_column_pairs_` loads eight rows and then runs a multi-layer in-register sort. Only the
// FIRST layer can ride the loads -- every later layer has no load left to attach a macro to -- but
// that layer is four explicit `SFPSWAP`s, and an explicit swap costs two cycles because the Vector
// Unit accepts only `SFPNOP` on the cycle after it (`SFPSWAP.md:112`). Hosting it removes those four
// stalls.
//
// Two encoding facts make this work:
//   * A template is "an arbitrary Vector Unit instruction" (`SFPLOADMACRO.md:59`) written by issuing
//     that instruction with `VD >= 12` (`SFPCONFIG.md:118`), so a template's `VC` is not confined to
//     LREG0..3 the way the step-group configuration leaves it.
//   * `VD = (VDHi << 2) | VDLo` with `Imm10 = (Imm9 << 1) | VDHi`, and the address's low bit "goes
//     unused" (`SFPLOAD.md:83`). So `VDHi` rides a bit that does not affect addressing: dropping the
//     `| 1` gives `VDHi = 0`, which lets a macro load LREG0..3 without changing the address.
//
// Step 3 of `window_sort<EIGHT, FOUR, ONE>` is `(VC,VD)` = (L1,L0) (L2,L3) (L5,L4) (L6,L7) ascending, and
// (L0,L1) (L3,L2) (L4,L5) (L7,L6) descending, so all four pairs are hostable once both facts hold.
constexpr int column_pair_len8_loads_len = 13;

// `load16_rows_x2<2>` in macro form: the four VC rows load plainly first, then the four VD rows load
// through macros 0..3, whose scheduled swaps fire at slots 5, 7, 9 and 11 (simple delay m, trigger
// 4+m, fire = trigger + delay + 1) -- each after both of its operands have landed. The trailing NOPs
// carry the last swap's second cycle, so the caller's leading SFPTRANSP reads settled registers.
// The macros' scheduled stores write a mid-sort value back to the row each one loaded; the group's
// own `store16_rows_x2` overwrites all eight afterwards, exactly as `len32_loads` relies on.
template <bool ascending>
inline void column_pair_len8_loads()
{
    if constexpr (ascending)
    {
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
        TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 6);
        TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, 10);
        TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);      // VD = LREG0
        TTI_SFPLOADMACRO((1 << 2) | 3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);     // VD = LREG3
        TTI_SFPLOADMACRO((2 << 2) | 0, InstrModLoadStore::INT32, ADDR_MOD_7, 2 | 1);  // VD = LREG4
        TTI_SFPLOADMACRO((3 << 2) | 3, InstrModLoadStore::INT32, ADDR_MOD_7, 14 | 1); // VD = LREG7
    }
    else
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
        TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 2);
        TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, 14);
        TTI_SFPLOADMACRO((0 << 2) | 1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);      // VD = LREG1
        TTI_SFPLOADMACRO((1 << 2) | 2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);      // VD = LREG2
        TTI_SFPLOADMACRO((2 << 2) | 1, InstrModLoadStore::INT32, ADDR_MOD_7, 6 | 1);  // VD = LREG5
        TTI_SFPLOADMACRO((3 << 2) | 2, InstrModLoadStore::INT32, ADDR_MOD_7, 10 | 1); // VD = LREG6
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

inline NOINLINE void record_column_pair_len8_loads(const bool ascending)
{
    detail::with_direction(
        ascending,
        [](auto direction) { load_replay_buf<Exec>(0 /*start*/, column_pair_len8_loads_len, [] { column_pair_len8_loads<decltype(direction)::value>(); }); });
}

// The same trick for `sort_len == 16`, whose first layer is step 4 of `window_sort<THIRTY_TWO, EIGHT, FOUR>`:
// (VC,VD) = (L2,L0) (L3,L1) (L6,L4) (L7,L5) ascending and (L0,L2) (L1,L3) (L4,L6) (L5,L7) descending.
// `sort_len == 4` is absent: `_topk_xl_split_column_pair_sort_<4>` issues an SFPTRANSP before
// `window_sort<FOUR, TWO, ONE>`, so the first thing after the load is a transpose. Scheduling that transpose
// on a macro costs all four Sequence words (three suppressed, one re-delayed) to save one
// instruction.
constexpr int column_pair_len16_loads_len = 13;

// Slot layout as `column_pair_len8_loads`, with one difference worth stating: this tail opens with step
// 3's SWAPS rather than a transpose, so the margin is larger (step 3 reads registers that settled at
// 6/8/10/12 while it issues from 13). The five NOPs are nonetheless MINIMAL, not merely sufficient:
// with four, the tail's first explicit SFPSWAP would issue on the cycle right after macro 3's
// scheduled swap, which the SFPLOADMACRO footnote forbids (Simple and Round must be idle there). A
// future "the tail only needs L0/L1, drop a NOP" cleanup would break this silently.
template <bool ascending>
inline void column_pair_len16_loads()
{
    if constexpr (ascending)
    {
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
        TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
        TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, 10);
        TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, 14);
        TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);     // VD = LREG0
        TTI_SFPLOADMACRO((1 << 2) | 1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);     // VD = LREG1
        TTI_SFPLOADMACRO((2 << 2) | 0, InstrModLoadStore::INT32, ADDR_MOD_7, 2 | 1); // VD = LREG4
        TTI_SFPLOADMACRO((3 << 2) | 1, InstrModLoadStore::INT32, ADDR_MOD_7, 6 | 1); // VD = LREG5
    }
    else
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
        TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 2);
        TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 6);
        TTI_SFPLOADMACRO((0 << 2) | 2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);      // VD = LREG2
        TTI_SFPLOADMACRO((1 << 2) | 3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);     // VD = LREG3
        TTI_SFPLOADMACRO((2 << 2) | 2, InstrModLoadStore::INT32, ADDR_MOD_7, 10 | 1); // VD = LREG6
        TTI_SFPLOADMACRO((3 << 2) | 3, InstrModLoadStore::INT32, ADDR_MOD_7, 14 | 1); // VD = LREG7
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

inline NOINLINE void record_column_pair_len16_loads(const bool ascending)
{
    detail::with_direction(
        ascending,
        [](auto direction) { load_replay_buf<Exec>(0 /*start*/, column_pair_len16_loads_len, [] { column_pair_len16_loads<decltype(direction)::value>(); }); });
}

// The four plain loads `step_group_pipelined` expects to already have been issued, for the first group.
template <int group_2_offset, bool ascending>
inline void prologue_loads()
{
    constexpr int near = ascending ? group_2_offset : 0;
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, near + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, near + 4);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, near + 8);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, near + 12);
}

template <int group_2_offset>
inline void prologue_loads_rt(const bool ascending)
{
    detail::with_direction(ascending, [](auto direction) { prologue_loads<group_2_offset, decltype(direction)::value>(); });
}

template <int group_2_offset, int inc_dst_addr>
inline void step_group_pipelined_rt(const bool ascending)
{
    detail::with_direction(ascending, [](auto direction) { step_group_pipelined<group_2_offset, inc_dst_addr, decltype(direction)::value>(); });
}

template <int group_2_offset, int inc_dst_addr>
inline void record_step_group_pipelined(const bool ascending)
{
    detail::with_direction(
        ascending,
        [](auto direction)
        { load_replay_buf<Exec>(0 /*start*/, pipelined_group_len, [] { step_group_pipelined<group_2_offset, inc_dst_addr, decltype(direction)::value>(); }); });
}

} // namespace topk_xl_fused_macro

#endif // TOPK_XL_FUSED_MACRO

#if TOPK_XL_FUSED_MACRO
namespace topk_xl_fused_macro
{

// Loads of a stride-16 sort_len_32 group, ordered so step 5's macro swaps run at the 4th, 6th, 8th and 10th
// instruction, each after its LREG i load; len32_stores overwrites their scheduled store-backs.
template <bool ascending>
inline void len32_loads()
{
    constexpr int group_2_offset = 16;
    constexpr int near           = ascending ? group_2_offset : 0;
    constexpr int far            = ascending ? 0 : group_2_offset;
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, near + 0);
    TTI_SFPLOADMACRO((3 << 2) | 3, InstrModLoadStore::INT32, ADDR_MOD_7, (far + 12) | 1);
    TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::INT32, ADDR_MOD_7, (far + 0) | 1);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, near + 12);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, near + 4);
    TTI_SFPLOADMACRO((1 << 2) | 1, InstrModLoadStore::INT32, ADDR_MOD_7, (far + 4) | 1);
    TTI_SFPLOADMACRO((2 << 2) | 2, InstrModLoadStore::INT32, ADDR_MOD_7, (far + 8) | 1);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, near + 8);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool ascending>
inline void len32_stores()
{
    constexpr int group_2_offset = 16;
    constexpr int near           = ascending ? group_2_offset : 0;
    constexpr int far            = ascending ? 0 : group_2_offset;
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, near + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, near + 4);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, near + 8);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, near + 12);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, far + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, far + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, far + 8);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_6, far + 12);
}

constexpr int len32_loads_len  = 11;
constexpr int len32_stores_len = 8;

// `iterations` × (len32_loads · W32 sort from stride 8 · len32_stores) with one `dir`, from the replay buffer. Out of
// line: the K = 2048 math image has little room.
inline NOINLINE void len32_groups(const int iterations, const bool dir)
{
    detail::window_sort_replay_loop<detail::WindowSize::THIRTY_TWO, detail::Stride::EIGHT, detail::Stride::ONE, len32_loads_len, len32_stores_len>(
        iterations,
        dir,
        [dir] { detail::with_direction(dir, [](auto direction) { len32_loads<decltype(direction)::value>(); }); },
        [dir] { detail::with_direction(dir, [](auto direction) { len32_stores<decltype(direction)::value>(); }); });
}

} // namespace topk_xl_fused_macro
#endif // TOPK_XL_FUSED_MACRO

// Smallest-stride merge step (Step 6 — stride 32). Pairs the lower 4 LREGs
// with the upper 4 LREGs across the full lane block.
//
// In fused mode all 8 LREGs participate (the packed value+index moves
// together). In unfused mode only the value LREGs (0..3) are swapped here;
// the matching index swaps are issued by the surrounding code through the
// parallel DST region at `indices_offset`.
template <bool fused = true>
inline void bitonic_sort_len_k(bool ascending)
{
    // Both modes compare rows 32 apart; the partner group sits in LREG4-7 when fused but in
    // LREG2-3 when unfused, so the register-model stride differs.
    constexpr detail::Stride stride  = fused ? detail::Stride::SIXTEEN : detail::Stride::EIGHT;
    constexpr detail::Fused is_fused = static_cast<detail::Fused>(fused);
    window_sort<detail::WindowSize::THIRTY_TWO, stride, stride, detail::Transposes::BOTH, is_fused>(ascending);
}

/**
 * @brief Run `groups` × GROUP_OFFSET / 8 unfused bodies of load8_rows_x2_unfused · bitonic_sort_len_k<false>(dir) ·
 * store8_rows_x2_unfused, each advancing Dst by 8 rows.
 *
 * The last body of a group advances by GROUP_OFFSET + 8 instead, past the rows the group's partner LREGs covered. The
 * first body is recorded; the others replay it, and the last body of a group replays all but its LREG7 store and
 * issues that store with the larger advance. With SFPLOADMACRO the body is the 16-instruction macro body
 * (topk_xl_unfused_macro::ce_full).
 *
 * @tparam INDICES: Dest distance from the value region to the index region, values = <ONE_TILE/TWO_TILES/FOUR_TILES>
 * @tparam GROUP_OFFSET: Dest distance from the LREG0-1 rows to the LREG2-3 rows, values = <ONE_FACE/TWO_FACES>
 * @param groups: Groups to run, at least 1
 * @param dir: Direction of every sort
 * @note Overwrites replay slots [0, 18). With SFPLOADMACRO, call loadmacro_configure<EIGHT, THIRTY_TWO, No>(dir) first.
 */
template <detail::IndicesOffset INDICES, detail::GroupOffset GROUP_OFFSET>
inline void unfused_step_groups(const int groups, const bool dir)
{
    static_assert(GROUP_OFFSET == detail::GroupOffset::ONE_FACE || GROUP_OFFSET == detail::GroupOffset::TWO_FACES, "no ADDR_MOD advances Dst past the group");
    constexpr int indices_offset   = static_cast<int>(INDICES);
    constexpr int group_offset     = static_cast<int>(GROUP_OFFSET);
    constexpr int bodies_per_group = group_offset / HALF_FACE_INC;
#if TOPK_XL_UNFUSED_MACRO
    constexpr std::uint32_t body_len = 16; // topk_xl_unfused_macro::ce_full
    auto record_first_body           = [dir] { topk_xl_unfused_macro::record_ce_full<indices_offset, group_offset, HALF_FACE_INC>(dir); };
#else
    constexpr std::uint32_t rows_x2_len = 8; // load8_rows_x2_unfused / store8_rows_x2_unfused: one SFPLOAD / SFPSTORE per LREG
    constexpr std::uint32_t body_len =
        rows_x2_len +
        window_sort_len<detail::WindowSize::THIRTY_TWO, detail::Stride::EIGHT, detail::Stride::EIGHT, detail::Transposes::BOTH, detail::Fused::No>() +
        rows_x2_len;
    auto record_first_body = [dir]
    {
        load_replay_buf<Exec>(
            0 /*start*/,
            body_len,
            [dir]
            {
                load8_rows_x2_unfused<indices_offset, group_offset>();
                bitonic_sort_len_k<false>(dir);
                store8_rows_x2_unfused<indices_offset, group_offset, HALF_FACE_INC>();
            });
    };
#endif
    static_assert(body_len <= 32, "the body must fit the 32-slot replay buffer");
    for (int g = 0; g < groups; g++)
    {
        for (int b = 0; b < bodies_per_group; b++)
        {
            if (g == 0 && b == 0)
            {
                record_first_body();
            }
            else if (b < bodies_per_group - 1)
            {
                lltt::replay(0 /*start*/, body_len);
            }
            else
            {
                lltt::replay(0 /*start*/, body_len - 1);
                TTI_SFPSTORE(
                    p_sfpu::LREG7, InstrModLoadStore::INT32, detail::dst_advance_addr_mod<group_offset + HALF_FACE_INC>(), indices_offset + group_offset + 4);
            }
        }
    }
}

template <int group_2_offset, int inc_dst_addr>
inline void fused_step_group(const bool ascending)
{
#if TOPK_XL_FUSED_MACRO
    detail::with_direction(ascending, [](auto direction) { topk_xl_fused_macro::step_group<group_2_offset, inc_dst_addr, decltype(direction)::value>(); });
#else
    load16_rows_x2<group_2_offset>();
    bitonic_sort_len_k(ascending);
    store16_rows_x2<group_2_offset, static_cast<detail::DstAdvance>(inc_dst_addr)>();
#endif
}

// =============================================================================
//  Dst transpose helpers
// =============================================================================
//
// The bitonic build steps for lengths > 256 need to view the same 2048-word
// region with rows and columns swapped (otherwise the compare-exchanges
// across columns would have to address Dst words non-sequentially).
// These helpers transpose a 16×16 face of 32-bit Dst data by shuttling
// half-words through SrcB and back, using TRNSPSRCB to rotate the SrcB
// view between passes.
//
// The 32-bit transpose is done in two passes through SrcB[16:31]:
//   Pass 1 (lo16) : copy the low 16 bits of each Dst word into SrcA via
//                   SrcB+TRNSPSRCB+MOVB2A, then write them back to Dst.
//   Pass 2 (hi16) : same shuffle for the high 16 bits, then write back to
//                   Dst with Fp32_enabled=0 so the writeback preserves the
//                   already-stored hi16 half.
//
// Entry/exit invariant: Fp32_enabled = 1 (needed for 32-bit Dst reads).

// Tensix instruction count of transpose_dest_face_32b. Nothing checks this: the count is hand maintained,
// and if the two ever disagree the recording captures the wrong number of instructions and every replayed
// face is wrong. Re-count when editing transpose_dest_face_32b.
// 5 x RMWCIB (both CFG fields are single-byte masks, so one instruction each) + 8 x MOVD2B + 2 x TRNSPSRCB
// + 4 x MOVB2A + 4 x MOVB2D + 2 x MOVA2D.
constexpr std::uint32_t transpose_face_body_len = 25;

// Transpose one 16x16 face of 32-bit Dst data at `dst_offset` from RWC.Dst. The final MOVA2D uses
// `last_addr_mod`: ADDR_MOD_7 (the default) leaves RWC.Dst alone, the absolute-offset form for the
// unfused index faces and transpose_N_faces, where the face count varies and a recording cannot be
// replayed across it; ADDR_MOD_5 (+16) steps RWC.Dst to the next face, see transpose_faces_from_dst.
template <int dst_offset, std::uint32_t last_addr_mod>
inline void transpose_dest_face_32b()
{
    // Transpose low 16 bits and backup it in SrcA
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float16_b));
    TTI_MOVD2B(1, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVD2B(1, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVD2B(1, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVD2B(1, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 12);

    TTI_TRNSPSRCB;

    TTI_MOVB2A(0, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 16);
    TTI_MOVB2A(4, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 20);
    TTI_MOVB2A(8, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 24);
    TTI_MOVB2A(12, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 28);

    // --- Pass 1: hi16 ---
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));

    TTI_MOVD2B(0, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVD2B(0, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVD2B(0, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVD2B(0, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 12);

    TTI_TRNSPSRCB;

    TTI_MOVB2D(0, 16, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVB2D(0, 20, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVB2D(0, 24, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVB2D(0, 28, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 12);

    // lo16 writeback: Fp32_enabled=0 + SrcA=Float32 -> writes lo16, preserves hi16
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float32));

    TTI_MOVA2D(1, 0, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, dst_offset + 0);
    TTI_MOVA2D(1, 8, last_addr_mod, p_mova2d::MOV_8_ROWS, dst_offset + 8);

    // Restore Fp32_enabled=1 for subsequent 32-bit Dst access
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
}

// Transpose `N` consecutive faces starting at the current RWC.Dst, recording the body once and replaying it.
// The face is addressed through RWC.Dst and the last MOVA2D steps it on by one face (ADDR_MOD_5 = +16), so the
// immediates are identical for every face and one recording serves them all. Dst addressing sums the
// instruction immediate, RWC.Dst and DEST_TARGET_REG_CFG_MATH_Offset, so this also leaves the MATH offset,
// which the sort owns, untouched.
template <std::uint32_t N>
inline void transpose_faces_from_dst()
{
    lltt::record<lltt::Exec>(0, transpose_face_body_len);
    transpose_dest_face_32b<0, ADDR_MOD_5>();
    for (std::uint32_t face = 1; face < N; face++)
    {
        lltt::replay(0, transpose_face_body_len);
    }
}

// `enter_transpose_cfg_block` / `leave_transpose_cfg_block` wrap a group of
// `transpose_8_faces` calls in a single CFG mode switch. Both `transpose_*`
// helpers need:
//   * `DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32 = 1` so the MOVD2B-style ops
//     read the explicit SrcA format rather than the implied one.
//   * `ALU_ACC_CTRL_Zero_Flag_disabled_src = 1` so the SrcB → SrcA copy
//     preserves the upper half-word during the two-pass transpose.
//
// They do NOT need `ALU_ACC_CTRL_Zero_Flag_disabled_dst`. `alu.h` scopes that
// bit broadly -- "disables zero flag reads in DEST" -- but the only Dst reads
// in this block are the transposes' MOVD2B, and every row they read has just
// been written, so none is flagged. Leaving it clear is also the safer
// reading: a flagged row yields zeros rather than whatever it last held.
//
// The bitonic work that runs between transposes does not touch either bit,
// so wrapping the whole batch saves 4 CFG ops per transpose call.
inline void enter_transpose_cfg_block()
{
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 1);
    math::_configure_src_zero_flag_(true);
}

inline void leave_transpose_cfg_block()
{
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 0);
    math::_configure_src_zero_flag_(false);
}

// enter_transpose_cfg_block, then hold the math class until SrcA and SrcB are math-owned and their in-flight consumers
// have drained: the face moves write SrcA/SrcB, which hardware arbitration does not gate.
inline void enter_transpose_cfg_block_after_src_drain()
{
    enter_transpose_cfg_block();
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
}

// Transpose all 8 value faces (and, for the unfused path, all 8 index
// faces). When the caller has already opened a transpose CFG block via
// `enter_transpose_cfg_block`, pass `manage_outer_cfg=false` to skip the
// (redundant) CFG ops at the entry and exit of this helper.
template <bool fused = true, int indices_offset = 256, bool manage_outer_cfg = true>
inline void transpose_8_faces()
{
    if constexpr (manage_outer_cfg)
    {
        enter_transpose_cfg_block();
    }

    // 8 value faces at +0, +16, ..., +112, walked through RWC.Dst from one recording. The callers reach here
    // with RWC.Dst == 0; make that explicit rather than implicit, since a stale Dst corrupts silently.
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    if constexpr (!fused && indices_offset == 8 * 16)
    {
        // Unfused mode keeps the indices in a parallel DST region that begins exactly where the eight value
        // faces end, so the sixteen faces are contiguous and one walk covers both.
        transpose_faces_from_dst<16>();
    }
    else if constexpr (!fused)
    {
        transpose_faces_from_dst<8>();
        // The index region does not abut the value faces here, so these keep the absolute-offset form.
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        detail::transpose_32b_dest_faces<indices_offset, 8>();
    }
    else
    {
        transpose_faces_from_dst<8>();
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    if constexpr (manage_outer_cfg)
    {
        leave_transpose_cfg_block();
    }
}

// Generalised face-transpose for the K=512 / K=1024 / K=2048 paths. `N`
// is the number of value faces to transpose (2 / 4 / 8 — corresponds to
// `row_scale_factor * 2` in the local_sort / rebuild bodies). With
// `manage_outer_cfg=true` (the default) this helper opens and closes its
// own transpose CFG block; pass `false` when the caller is wrapping
// several `transpose_N_faces` calls in one shared
// `enter_transpose_cfg_block` / `leave_transpose_cfg_block` pair.
//
// It emits each face's transpose unrolled; `transpose_8_faces` walks the
// faces from one replay recording instead.
template <int N, bool fused = true, int indices_offset = 256, bool manage_outer_cfg = true>
inline void transpose_N_faces()
{
    if constexpr (manage_outer_cfg)
    {
        enter_transpose_cfg_block();
    }

    detail::transpose_32b_dest_faces<0, N>();
    if constexpr (!fused)
    {
        detail::transpose_32b_dest_faces<indices_offset, N>();
    }

    if constexpr (manage_outer_cfg)
    {
        leave_transpose_cfg_block();
    }
}

/**
 * @brief Transpose N value faces (and, unfused, N index faces) inside an open transpose CFG block: the replay walk of
 * @ref transpose_8_faces for N = 8, the unrolled @ref transpose_N_faces otherwise.
 *
 * @tparam N: Value faces to transpose, values = <2/4/8>
 * @tparam fused: Fused (indices packed in the values) or unfused layout
 * @tparam indices_offset: Dest distance from the value region to the index region, unfused only
 * @note Call between enter_transpose_cfg_block and leave_transpose_cfg_block.
 */
template <int N, bool fused, int indices_offset>
inline void transpose_faces()
{
    if constexpr (N == 8)
    {
        transpose_8_faces<fused, indices_offset, false /* manage_outer_cfg */>();
    }
    else
    {
        transpose_N_faces<N, fused, indices_offset, false /* manage_outer_cfg */>();
    }
}

// =============================================================================
//  MOP programs
// =============================================================================

// Program the MOP Expander for the length-2048 column-pair build phase of
// `_topk_xl_rebuild_<fused=true>`. The loop body is
//
//     load16_rows_x2<2> + window_sort<detail::WindowSize::THIRTY_TWO, detail::Stride::EIGHT, detail::Stride::ONE>(dir) + store16_rows_x2<2, 16>
//     = 8 + 18 + 8 = 34 instructions
//
// which is too large to fit in the 32-slot replay buffer as one contiguous
// window. The two `SFPTRANSP` instructions inside the sort are natural
// structural break points, so we record only the 32 "data" instructions
// and emit the two SFPTRANSPs from the MOP template itself (slots A1 / A3).
//
//   Replay buffer layout (recorded once at the start of the loop):
//     slots [0..15] : load16_rows_x2<2>           (8)
//                   + first 8 SFPSWAPs of sort    (8)
//     slots [16..23]: second 8 SFPSWAPs of sort   (8)
//     slots [24..31]: store16_rows_x2<2, 16>      (8)
//
//   MOP template (5 slots, unpackB + unpack_halo path):
//     A0 = REPLAY(0, 16)   ← load + first half of sort
//     A1 = SFPTRANSP       ← first transpose of the sort
//     A2 = REPLAY(16, 8)   ← second half of sort
//     A3 = SFPTRANSP       ← second transpose of the sort
//     B  = REPLAY(24, 8)   ← store
//
// One `TTI_MOP` issue per iter replaces 2 REPLAY + 18 inline sort = 20
// math-thread issues. Clobbers the merge MOP setup, so callers must run
// `topk_mop_config<true>()` again before the next `_topk_xl_merge_`.
inline void topk_rebuild_build2048_mop_config()
{
    constexpr std::uint32_t replay_load_and_first_swaps = lltt::replay_insn(0, 16);
    constexpr std::uint32_t replay_second_swaps         = lltt::replay_insn(16, 8);
    constexpr std::uint32_t replay_store                = lltt::replay_insn(24, 8);
    constexpr std::uint32_t sfptransp_op                = TT_OP_SFPTRANSP(0, 0, 0, 0);

    // Build the 5-slot template directly (unpackB + unpack_halo both enabled),
    // bypassing the `lA` / `lBhA` factories whose default A0..A3 / B values
    // are UNPACR instructions — we need arbitrary instruction words here.
    ckernel_unpack_template tmpl(
        true /* unpackB */,
        true /* unpack_halo */,
        replay_load_and_first_swaps /* A0 */,
        sfptransp_op /* A1 */,
        replay_second_swaps /* A2 */,
        sfptransp_op /* A3 */,
        TT_OP_NOP /* skipA */,
        replay_store /* B */,
        TT_OP_NOP /* skipB */);
    tmpl.program();
}

/**
 * @brief Sort each lane's even and odd column in `dir`, `iterations` times: load16_rows_x2<ODD_COLUMNS> ·
 * window_sort<THIRTY_TWO, EIGHT, ONE> · store16_rows_x2<ODD_COLUMNS, ONE_FACE>.
 *
 * The body is 8 + 18 + 8 = 34 instructions, more than the 32 replay slots, so it runs through the MOP template of
 * @ref topk_rebuild_build2048_mop_config: the 32 loads, swaps and stores sit in three replay ranges and the template
 * issues the two SFPTRANSPs (8 4 · T · 2 1 · T) from its A1 / A3 slots. The first iteration records the ranges while
 * executing; the others are one TTI_MOP issue each.
 *
 * @param iterations: Bodies to run, at least 1
 * @param dir: Direction of every sort; it must not change between iterations
 * @note Reprograms the MOP template; run `topk_mop_config<true>()` again before the next `_topk_xl_merge_`.
 */
inline void sort_column_pairs_with_mop(const int iterations, const bool dir)
{
    topk_rebuild_build2048_mop_config();
    load_replay_buf<Exec>(
        0,
        16,
        [dir]
        {
            load16_rows_x2<2>();
            window_sort<detail::WindowSize::THIRTY_TWO, detail::Stride::EIGHT, detail::Stride::FOUR>(dir);
        });
    TTI_SFPTRANSP(0, 0, 0, 0);
    load_replay_buf<Exec>(
        16, 8, [dir] { window_sort<detail::WindowSize::THIRTY_TWO, detail::Stride::TWO, detail::Stride::ONE, detail::Transposes::NONE>(dir); });
    TTI_SFPTRANSP(0, 0, 0, 0);
    load_replay_buf<Exec>(24, 8, [] { store16_rows_x2<2, detail::DstAdvance::ONE_FACE>(); });
    ckernel_unpack_template::run(iterations - 1);
}

// =============================================================================
//  Canonical "big block" (A + B + C)
// =============================================================================
//
// Every bitonic build phase whose per-column "tail" runs through the
// stride-64 / stride-32 / stride-16 cascade ends up calling this helper.
// It is shared between the K=2048 fast path, the K=1024 generic path, and
// the K=512 generic path; which sub-blocks are emitted is controlled by
// `row_scale_factor`:
//
//   row_scale_factor == 4 (K=2048): sub-blocks A + B + C
//   row_scale_factor == 2 (K=1024): sub-blocks B + C
//   row_scale_factor == 1 (K=512):  sub-block C only (single iter)
//
// Sub-block A only kicks in when the per-column run is deep enough to need
// a stride-64 compare-exchange — K=2048 is the only such case. For K=1024
// the column run is half as deep, so stride-32 is the first scale; for
// K=512 only the stride-16 tail runs.
//
// Sub-blocks A and C record their first iteration and replay the rest (`window_sort_rows`, or `len32_groups` for a
// K=2048 sub-block C with macros); sub-block B is left inline since recording it doesn't pay back at this size.
// With TOPK_XL_FUSED_MACRO, sub-blocks A and B are software-pipelined: A records one `step_group_pipelined<64, 16>` in
// slots [0..11] behind a `prologue_loads`, and B runs `step_group_pipelined` inline behind its own prologue.
template <int row_scale_factor>
inline void canonical_big_block_with_replay_body(bool dir)
{
    constexpr int consecutive_32_offset = 16;

    if constexpr (row_scale_factor >= 4)
    {
        // ── Sub-block A: `row_scale_factor` × (load<64> + sort_k + store<64, 16>) ──
#if TOPK_XL_FUSED_MACRO
        // Pipelined; see `step_group_pipelined`. The last iteration prefetches one group past the end, which only writes LREGs that are dead by then.
        topk_xl_fused_macro::prologue_loads_rt<64>(dir);
        topk_xl_fused_macro::record_step_group_pipelined<64, 16>(dir);
        for (int i = 1; i < row_scale_factor; i++)
        {
            lltt::replay(0, topk_xl_fused_macro::pipelined_group_len);
        }
#else
        window_sort_rows<
            detail::WindowSize::THIRTY_TWO,
            detail::Stride::SIXTEEN,
            detail::Stride::SIXTEEN,
            detail::GroupOffset::FOUR_FACES,
            detail::DstAdvance::ONE_FACE,
            detail::Fused::Yes>(row_scale_factor, dir);
#endif
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }

    if constexpr (row_scale_factor >= 2)
    {
        // ── Sub-block B: `row_scale_factor / 2` × paired (load<32> + sort_k + store<32, 16/48>) ──
        // The +48 store on the second of each pair uses ADDR_MOD_1 to fold
        // the trailing +16 advance into the last SFPSTORE — saves one
        // math-thread issue per inner iter compared to the original
        // K!=2048 generic body which emitted `store<32, 32> + 2×INCRWC(+8)`.
#if TOPK_XL_FUSED_MACRO
        // Pipelined like sub-block A. Its trailing SETRWC reset Dst, so A's final prefetch is stale and this prologue reloads; within B the groups chain.
        topk_xl_fused_macro::prologue_loads_rt<32>(dir);
        for (int i = 0; i < (row_scale_factor >> 1); i++)
        {
            topk_xl_fused_macro::step_group_pipelined_rt<32, 16>(dir);
            topk_xl_fused_macro::step_group_pipelined_rt<32, 48>(dir);
        }
#else
        for (int i = 0; i < (row_scale_factor >> 1); i++)
        {
            fused_step_group<32, 16>(dir);
            fused_step_group<32, 48>(dir);
        }
#endif
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        // ── Sub-block C: `row_scale_factor` × (load<16> + W32 sort + store<16, 32>) ──
#if TOPK_XL_FUSED_MACRO
        // At K = 1024 each recording serves two groups only, too few for the macro loads to pay off.
        if constexpr (row_scale_factor >= 4)
        {
            topk_xl_fused_macro::len32_groups(row_scale_factor, dir);
        }
        else
#endif
        {
            window_sort_rows<
                detail::WindowSize::THIRTY_TWO,
                detail::Stride::SIXTEEN,
                detail::Stride::ONE,
                detail::GroupOffset::ONE_FACE,
                detail::DstAdvance::TWO_FACES,
                detail::Fused::Yes>(row_scale_factor, dir);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
    else
    {
        // K=512 (row_scale_factor=1): single stride-16 W32 sort inline.
        // No replay buffer here because there's nothing to amortise the
        // recording across.
        load16_rows_x2<consecutive_32_offset>();
        window_sort<detail::WindowSize::THIRTY_TWO, detail::Stride::SIXTEEN, detail::Stride::ONE>(dir);
        store16_rows_x2<consecutive_32_offset, detail::DstAdvance::TWO_FACES>();
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
}

// Out of line: its call sites are many, and the K = 2048 math image has little room.
inline NOINLINE void canonical_big_block_2048(bool dir)
{
    canonical_big_block_with_replay_body<4>(dir);
}

template <int row_scale_factor>
inline void canonical_big_block_with_replay(bool dir)
{
    if constexpr (row_scale_factor >= 4)
    {
        canonical_big_block_2048(dir);
    }
    else
    {
        canonical_big_block_with_replay_body<row_scale_factor>(dir);
    }
}

/**
 * @brief Flip the swap direction of the lanes FLIP selects, until @ref clear_lane_flip.
 *
 * @tparam FLIP: Lanes to flip, values = <EVERY_LANE/EVERY_2_LANES/EVERY_4_LANES>
 */
template <detail::LaneFlip FLIP>
inline void set_lane_flip()
{
    static_assert(FLIP != detail::LaneFlip::NONE, "nothing to flip");
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0100);
    TTI_SFPCONFIG(static_cast<std::uint32_t>(FLIP), 0xF, 8);
}

/** @brief Restore the default SFPU config after @ref set_lane_flip. */
inline void clear_lane_flip()
{
    TTI_SFPCONFIG(0x0000, 0xF, 1);
}

/**
 * @brief Run canonical_big_block_with_replay on the even, then the odd Dest columns of the tiles at `tile_offset`, and
 * leave the Dst write offset on the even columns.
 *
 * @tparam ROW_SCALE_FACTOR: K / 512, values = <1/2/4>
 * @param tile_offset: Dst offset of the first tile
 * @param dir: Direction of the sort
 */
template <int ROW_SCALE_FACTOR>
inline void canonical_big_block_columns(const std::uint32_t tile_offset, const bool dir)
{
    for (int col = 0; col < 2; col++)
    {
        canonical_big_block_with_replay<ROW_SCALE_FACTOR>(dir);
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
}

/**
 * @brief Transpose the faces, sort each lane's even and odd column together with window_sort<WINDOW, FROM, ONE>, and
 * transpose back.
 *
 * @tparam ROW_SCALE_FACTOR: K / 512, values = <1/2/4>; 2 · ROW_SCALE_FACTOR faces and column-pair groups
 * @param dir: Direction of every sort
 * @note Call inside an open transpose CFG block (see @ref enter_transpose_cfg_block_after_src_drain).
 */
template <int ROW_SCALE_FACTOR, detail::WindowSize WINDOW, detail::Stride FROM>
inline void sort_transposed_column_pairs(const bool dir)
{
    constexpr int faces = 2 * ROW_SCALE_FACTOR;
    transpose_faces<faces, true /* fused */, 256 /* indices_offset */>();
    window_sort_rows<WINDOW, FROM, detail::Stride::ONE, detail::GroupOffset::ODD_COLUMNS, detail::DstAdvance::ONE_FACE, detail::Fused::Yes>(faces, dir);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    transpose_faces<faces, true /* fused */, 256 /* indices_offset */>();
}

namespace detail
{

/**
 * @brief Local sort of one chunk up to its first face transpose: runs of 32, 64 and 128 down each Dest column, then
 * runs of 256 across each lane's even and odd column.
 *
 * Per column: row_scale_factor = K / 512 sort_32_rows blocks; for K >= 1024, row_scale_factor / 2 rounds of the
 * stride-32 step groups and of the stride-16 W32 sort; for K = 2048 the canonical big block. Then, unless EARLY_EXIT,
 * 2 · row_scale_factor column-pair groups under LaneFlip::EVERY_LANE and the canonical big block on both columns.
 *
 * @tparam K: Elements per chunk, values = <512/1024/2048>
 * @tparam USE_LOADMACRO: Run the length-256 column pairs as fused SFPLOADMACRO bodies, values = <No/Yes>
 * @tparam EARLY_EXIT: Stop once each 64-row column is sorted on its own, values = <No/Yes>
 * @param tile_offset: Dst offset of the chunk's first tile
 * @param ascending: Direction of the first run
 * @return The direction the build phases after the head continue with
 * @note With TOPK_XL_FUSED_MACRO the step groups always run as macros: program the fused step-group macros first for
 *       K >= 1024, and for USE_LOADMACRO.
 */
template <std::uint32_t K, UseLoadMacro USE_LOADMACRO, EarlyExit EARLY_EXIT>
inline bool local_sort_head(const std::uint32_t tile_offset, const bool ascending)
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024 or 2048");
    static_assert(TOPK_XL_FUSED_MACRO || USE_LOADMACRO == UseLoadMacro::No, "SFPLOADMACRO is disabled in this build");
    constexpr int row_scale_factor = K / 512;
    constexpr bool early_exit      = EARLY_EXIT == EarlyExit::Yes;
    // Blaze, and an early exit, keep `dir` across the stride-32 and stride-16 rounds when there is only one of each
    // (K = 1024).
    constexpr bool flip_per_round = (!topk_xl_blaze_compat && !early_exit) || (row_scale_factor >> 1) > 1;
    bool dir                      = ascending;

    for (int col = 0; col < 2; col++)
    {
        // The first block records the replay bodies, the others replay them (see `sort_32_rows`).
        sort_32_rows<ReplayMode::RECORD>(ascending, dir);
        if constexpr (row_scale_factor > 1)
        {
            dir = !dir;
        }
        for (int i = 1; i < row_scale_factor; i++)
        {
            sort_32_rows<ReplayMode::REPLAY>(ascending, dir);
            dir = !dir;
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        if constexpr (row_scale_factor >= 2)
        {
            // Length 64: stride-32 step groups in pairs; the second store of a pair folds the trailing +16 into
            // ADDR_MOD_1 (+48).
            for (int i = 0; i < (row_scale_factor >> 1); i++)
            {
#if TOPK_XL_FUSED_MACRO
                // Pipelined within the pair, which share `dir`; a prefetch's `near` offset depends on the direction,
                // so each pair re-prologues (see `step_group_pipelined`).
                topk_xl_fused_macro::prologue_loads_rt<32>(dir);
                topk_xl_fused_macro::step_group_pipelined_rt<32, 16>(dir);
                topk_xl_fused_macro::step_group_pipelined_rt<32, 48>(dir);
#else
                fused_step_group<32, 16>(dir);
                fused_step_group<32, 48>(dir);
#endif
                if constexpr (flip_per_round)
                {
                    dir = !dir;
                }
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

            // Stride-16 W32 sort, 2 groups per round.
            for (int i = 0; i < (row_scale_factor >> 1); i++)
            {
#if TOPK_XL_FUSED_MACRO
                if constexpr (row_scale_factor >= 4 && !early_exit)
                {
                    topk_xl_fused_macro::len32_groups(2 /*iterations*/, dir);
                }
                else
#endif
                {
                    window_sort_rows<WindowSize::THIRTY_TWO, Stride::SIXTEEN, Stride::ONE, GroupOffset::ONE_FACE, DstAdvance::TWO_FACES, Fused::Yes>(
                        2 /*iterations*/, dir);
                }
                if constexpr (flip_per_round)
                {
                    dir = !dir;
                }
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        }

        // Length 128.
        if constexpr (row_scale_factor >= 4 && !early_exit)
        {
            canonical_big_block_with_replay<row_scale_factor>(dir);
        }
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
        if constexpr (!early_exit)
        {
            // The next column runs the other way, so the two columns form a bitonic input for length 256.
            dir = !dir;
        }
    }

    if constexpr (!early_exit)
    {
        // Length 256: the runs span a lane's even and odd column; flipping every other lane lets one body cover both
        // halves of the merge.
        set_lane_flip<LaneFlip::EVERY_LANE>();
#if TOPK_XL_FUSED_MACRO
        if constexpr (USE_LOADMACRO == UseLoadMacro::Yes)
        {
            // Pipelined; see `step_group_pipelined`. The last iteration prefetches one group past the end, which only
            // writes LREGs that are dead by then.
            topk_xl_fused_macro::prologue_loads_rt<2>(dir);
            topk_xl_fused_macro::record_step_group_pipelined<2, 16>(dir);
            for (int i = 1; i < 2 * row_scale_factor; i++)
            {
                lltt::replay(0 /*start*/, topk_xl_fused_macro::pipelined_group_len);
            }
        }
        else
#endif
        {
            window_sort_rows<WindowSize::THIRTY_TWO, Stride::SIXTEEN, Stride::SIXTEEN, GroupOffset::ODD_COLUMNS, DstAdvance::ONE_FACE, Fused::Yes>(
                2 * row_scale_factor, dir);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        canonical_big_block_columns<row_scale_factor>(tile_offset, dir);
        clear_lane_flip();
    }
    return dir;
}

} // namespace detail

// #############################################################################
//  Public API
//
//  Entry points used by the llk_api wrappers, ckernel_sfpu_topk_xl_split.h and the tt-llk tests.
// #############################################################################

// =============================================================================
//  MOP Expander programs
// =============================================================================
//
// The math-thread MOP Expander packs up to 5 instruction slots (A0..A3 + B)
// behind a single `TTI_MOP` issue, with an optional iteration count. We
// program it once at init so the merge's hot loop collapses to one MOP
// issue per "column".

// Program the MOP Expander to issue a single `REPLAY(0, body_len)` per slot.
// `_topk_xl_merge_` uses this to fire its inner loop body — recorded into
// replay slots [0, body_len) — with one MOP issue per column.
//
// Recording window sizes (must match the body recorded by the merge):
//   fused:   8 (load16) + 4 (sort_k) + 4 (store4_top_only) = 16 instructions
//            (15 as `topk_xl_fused_macro::step_group`, plus one SFPNOP)
//   unfused: 8 (load8)  + 2 (sort_k) + 8 (store8)          = 18 instructions
//
// Without the fused macros the fused body drops the 4 SFPSTOREs for LREG4..7 (the per-pair min
// half), which is dead by construction (see `store4_rows_top_only`); the
// macros store that half too.
//
// Nothing else on the math thread touches the MOP Expander, so this
// programming survives across every merge call.
template <bool fused>
inline void topk_mop_config()
{
    // Unfused: the SFPLOADMACRO body is 16 instructions (the two SFPSWAPs and
    // two of the eight stores ride the macros); the plain body is 18. The fused
    // body is 16 either way. Must match what `_topk_xl_merge_` records.
#if TOPK_XL_UNFUSED_MACRO
    constexpr int body_len = 16;
#else
    constexpr int body_len = fused ? 16 : 18;
#endif
    constexpr std::uint32_t replay_body = lltt::replay_insn(0, body_len);
    ckernel_unpack_template tmpl        = ckernel_unpack_template::lA(replay_body, TT_OP_NOP);
    tmpl.program();
}

// _llk_math_topk_xl_copy_init_ rewrites ADDR_MOD_0 and ADDR_MOD_3 for datacopy, so ADDR_MOD_3 must be restored here.
inline void topk_reinit_unfused_rebuild_after_copy()
{
    detail::set_dst_addr_mode<ADDR_MOD_3, TWO_FACE_AND_HALF_INC>();
    detail::set_dst_addr_mode<ADDR_MOD_2, FACE_AND_HALF_INC>();

    topk_mop_config<false>();
}

// =============================================================================
//  Init
// =============================================================================
//
// Programs the ADDR_MODs (see "Address-mod recipe" at the top of the file)
// and the merge MOP Expander. Must be called once per query before the sort
// phases (local sort, merge, rebuild). Every supported K uses the same
// ADDR_MODs and MOP programming.
//
// In `fused` mode values and indices share a single 32-bit DST word, so
// no separate index-tracking path needs to be programmed. The two
// stride-folding ADDR_MODs (`ADDR_MOD_2`, `ADDR_MOD_3`) that only the
// unfused path consumes — plus `ADDR_MOD_4`'s +8 step also unused fused
// — are skipped in that case, saving ~9 CFG writes per init.
template <std::uint32_t K, bool fused>
inline void _topk_xl_init_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");

    // ADDR_MOD_6 — +32 (two faces).
    detail::set_dst_addr_mode<ADDR_MOD_6, TWO_FACE_INC>();

    // ADDR_MOD_5 — +16 (one face).
    detail::set_dst_addr_mode<ADDR_MOD_5, FACE_INC>();

    // ADDR_MOD_4/3/2 are only used by the unfused path.
    if constexpr (!fused)
    {
        // ADDR_MOD_4 — +8 (half a face), used by the unfused store helper.
        detail::set_dst_addr_mode<ADDR_MOD_4, HALF_FACE_INC>();

        // ADDR_MOD_3 — +40 (two and a half faces).
        // Folds the trailing +8 advance into the stride-32 loop's last store.
        detail::set_dst_addr_mode<ADDR_MOD_3, TWO_FACE_AND_HALF_INC>();

        // ADDR_MOD_2 — +24 (one and a half faces).
        // Folds the trailing +8 advance into the stride-16 loop's last store.
        detail::set_dst_addr_mode<ADDR_MOD_2, FACE_AND_HALF_INC>();

        // Enable index tracking; unfused indices occupy a parallel DST region.
        constexpr std::uint32_t index_tracking_cfg = 0x4;
        _sfpu_load_config32_(sfpi::SFPCONFIG_DEST_SFPU_CTRL, 0x0 /* upper 16bit */, index_tracking_cfg /* lower 16bit */);
    }

    // ADDR_MOD_1 — +48 (three faces).
    // Folds the trailing +16 advance into sub-block B's last store.
    detail::set_dst_addr_mode<ADDR_MOD_1, THREE_FACE_INC>();

    // Configure the merge MOP; also configures the unfused macro sequences.
    topk_mop_config<fused>();
}

// =============================================================================
//  Public entry: local sort
// =============================================================================
//
// Sort K elements in place (two consecutive 1024-element DST tiles starting at `dst_index` for K = 2048). Each
// element is the packed u32 `[ bf16 value | u16 index ]`. The sort proceeds bottom-up, doubling the run length at
// each phase:
//
//   len =  32 /  64 / 128 : down each Dest column              (detail::local_sort_head)
//   len = 256             : across each lane's even/odd column (detail::local_sort_head)
//   len = K/4, K/2, K     : need transposes                    (sort_transposed_column_pairs)
//
// Each transposed phase sorts the column pairs between two face transposes, then runs the canonical big block on both
// columns with alternating groups of lanes flipped. One transpose CFG block wraps all six transposes; the bitonic
// work between them doesn't touch the affected CFG bits.
//
// early_exit_K64 sorts each 64-row column in isolation and returns before the cross-column merge phases (sparse-K
// reader: sink all-zero packed mask words to the bottom of each column).
template <std::uint32_t K, bool early_exit_K64 = false>
inline void _topk_xl_local_sort_generic_(const std::uint32_t dst_index, const bool ascending)
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    // The length-64 build lives in the K >= 1024 block, so a K=512 instantiation would return before it runs.
    static_assert(!early_exit_K64 || K >= 1024, "early_exit_K64 requires K >= 1024: the length-64 build phase lives in the K >= 1024 block");
#if TOPK_XL_FUSED_MACRO
    // K = 512 runs two step groups only, too few to repay the macro programming.
    if constexpr (K >= 1024)
    {
        loadmacro_configure<detail::Stride::SIXTEEN, detail::WindowSize::THIRTY_TWO, detail::Order::DESCENDING, detail::Fused::Yes>();
    }
    constexpr detail::UseLoadMacro use_loadmacro = K >= 1024 ? detail::UseLoadMacro::Yes : detail::UseLoadMacro::No;
#else
    constexpr detail::UseLoadMacro use_loadmacro = detail::UseLoadMacro::No;
#endif
    constexpr detail::EarlyExit early_exit = early_exit_K64 ? detail::EarlyExit::Yes : detail::EarlyExit::No;
    constexpr int row_scale_factor         = K / 512;
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    [[maybe_unused]] const bool dir = detail::local_sort_head<K, use_loadmacro, early_exit>(tile_offset, ascending);
    if constexpr (early_exit_K64)
    {
        return;
    }
    else
    {
        enter_transpose_cfg_block_after_src_drain();

        // len = K/4
        sort_transposed_column_pairs<row_scale_factor, detail::WindowSize::FOUR, detail::Stride::TWO>(dir);
        set_lane_flip<detail::LaneFlip::EVERY_2_LANES>();
        canonical_big_block_columns<row_scale_factor>(tile_offset, dir);
        clear_lane_flip();

        // len = K/2
        sort_transposed_column_pairs<row_scale_factor, detail::WindowSize::EIGHT, detail::Stride::FOUR>(dir);
        set_lane_flip<detail::LaneFlip::EVERY_4_LANES>();
        canonical_big_block_columns<row_scale_factor>(tile_offset, dir);
        clear_lane_flip();

        // len = K
        sort_transposed_column_pairs<row_scale_factor, detail::WindowSize::THIRTY_TWO, detail::Stride::EIGHT>(dir);
        leave_transpose_cfg_block();
        canonical_big_block_columns<row_scale_factor>(tile_offset, dir);

        // Clear SrcA and SrcB valids — the next caller might issue a DST transpose, which requires them invalid.
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
    }
}

template <std::uint32_t K>
inline void _topk_xl_local_sort_(const std::uint32_t dst_index, const bool ascending)
{
    _topk_xl_local_sort_generic_<K>(dst_index, ascending);
}

// =============================================================================
//  Public entry: merge
// =============================================================================
//
// Bitonic-merge two adjacent sorted runs in DST into a single top-K run.
// For K=2048 each run spans two DST tiles; for K=1024 / K=512 a run spans
// one DST tile (or half a tile for K=512). On entry, the first run sits
// at `dst_index` and the second run sits one sorted-sequence further
// along; on exit the merged top-K replaces the first run and the second
// run's slots are left dead (the next merge stage overwrites them via
// `topk_xl_copy_tile(recv, ...)`).
//
// The same body shape is used for every K — only the load/store distance,
// the parallel-index offset (unfused only), and the per-column iteration
// count change with K:
//
//   distance       = (fused ? 64 : 128) * num_tiles_per_sequence
//   indices_offset = 64 * num_tiles_per_sequence
//   n_iters        = fused ? row_scale_factor * 2 : row_scale_factor * 4
//
// where `num_tiles_per_sequence = (K == 2048) ? 2 : 1` and
// `row_scale_factor = K / 512`.
//
// Body lengths fit in a single replay window:
//
//   fused (16 slots):    8 (load16) + 4 (sort_k) + 4 (store4_top_only)
//                        (`topk_xl_fused_macro::step_group` and one SFPNOP)
//   unfused (18 slots):  8 (load8)  + 2 (sort_k) + 8 (store8)
//                        (16 as `topk_xl_unfused_macro::ce_full`)
//
// Fused path: only the "top half" (LREG0..3, the per-pair max) is stored
// back to DST (the fused macros at K >= 1024 store the min half too).
// LREG4..7 hold the per-pair min, which is dead by
// construction — the next-stage `topk_xl_copy_tile(recv, ...)` overwrites
// the slot it lives in before any reader sees it (in K=2048 that's the
// +128-tile region; in K=1024 it's the +64 region within the same tile).
// See `store4_rows_top_only` for the full justification.
//
// We record the body once (recording IS iter 0 of col=0), then fire it
// through the MOP Expander programmed at init: each column collapses to
// a single TTI_MOP issue that re-runs the recorded body N_ITERS times.
template <std::uint32_t K, bool fused>
inline void _topk_xl_merge_(const std::uint32_t dst_index)
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");

    [[maybe_unused]] constexpr bool descending    = false; // direction the merge writes
    constexpr int num_tiles_per_sequence          = (K == 2048) ? 2 : 1;
    constexpr int row_scale_factor                = (K == 512) ? 1 : (K == 1024) ? 2 : 4;
    constexpr int distance                        = fused ? (64 * num_tiles_per_sequence) : (128 * num_tiles_per_sequence);
    [[maybe_unused]] constexpr int indices_offset = 64 * num_tiles_per_sequence;
    const std::uint32_t tile_offset               = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

#if TOPK_XL_UNFUSED_MACRO
    constexpr int body_len = 16; // must match topk_mop_config's REPLAY length
#else
    constexpr int body_len = fused ? 16 : 18;
#endif
    constexpr int n_iters = fused ? (row_scale_factor * 2) : (row_scale_factor * 4);
    static_assert(n_iters >= 2, "n_iters < 2 would skip the MOP firing of col=0");
    static_assert(n_iters <= 255, "ckernel_unpack_template::run takes a uint8_t count");

    // Record the loop body once into replay slots [0, body_len). The Exec
    // mode means this recording also counts as iter 0 of col=0's work, so
    // col=0 only needs (n_iters - 1) replays below.
    if constexpr (fused)
    {
#if TOPK_XL_FUSED_MACRO
        // The macros also store the minimum half, which is dead after the merge (see `store4_rows_top_only`).
        // K = 512 runs four groups only, too few to repay the macro programming.
        if constexpr (K >= 1024)
        {
            loadmacro_configure<detail::Stride::SIXTEEN, detail::WindowSize::THIRTY_TWO, detail::Order::DESCENDING, detail::Fused::Yes>();
            load_replay_buf<Exec>(
                0,
                body_len,
                []
                {
                    topk_xl_fused_macro::step_group<distance, 16, descending>();
                    TTI_SFPNOP;
                });
        }
        else
#endif
        {
            load_replay_buf<Exec>(
                0,
                body_len,
                []
                {
                    load16_rows_x2<distance>();
                    bitonic_sort_len_k(descending);
                    store4_rows_top_only<16>();
                });
        }
    }
    else
    {
#if TOPK_XL_UNFUSED_MACRO
        // SFPLOADMACRO body: the two SFPSWAPs ride the loads' Simple slots
        // and the macroVD stores ride their Store slots. The merge always
        // sorts with `descending` (= the ascending=false operand order), so
        // macroVD = LREG2/3 and the templates' VC = LREG0/1. Every scheduled
        // store retires inside the body — no trailing SFPNOPs needed.
        loadmacro_configure<detail::Stride::EIGHT, detail::WindowSize::THIRTY_TWO, detail::Order::DESCENDING, detail::Fused::No>();
        load_replay_buf<Exec>(0, body_len, [] { topk_xl_unfused_macro::ce_full<indices_offset, distance, 8 /* inc_dst_addr */, false>(); });
#else
        load_replay_buf<Exec>(
            0,
            body_len,
            []
            {
                load8_rows_x2_unfused<indices_offset, distance>();
                bitonic_sort_len_k<fused>(descending);
                store8_rows_x2_unfused<indices_offset, distance, 8 /* inc_dst_addr */>();
            });
#endif
    }

    // col=0: fire the remaining (n_iters - 1) iters via one MOP issue.
    ckernel_unpack_template::run(n_iters - 1);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Switch the Dst write pointer to the odd column group.
    set_dst_write_addr_offset(tile_offset + 2);

    // col=1: full n_iters worth of work, again as one MOP issue.
    ckernel_unpack_template::run(n_iters);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    set_dst_write_addr_offset(tile_offset + 0);
}

// =============================================================================
//  Public entry: rebuild
// =============================================================================
//
// After `_topk_xl_merge_` the top-K result lives in DST tiles dst_index / dst_index+1 but its bitonic structure is
// broken (the merge only guarantees ordering, not the bitonic property the next merge would need). Rebuild restores
// it by repeating the length-K build phase of the local sort: transpose, sort the column pairs, transpose back, then
// the compare steps that cross faces within each column.
//
//   Fused: the column-pair body is 34 instructions, too large for one replay window, so it runs from the 5-slot
//          MOP template (sort_column_pairs_with_mop); canonical_big_block_with_replay does the per-column steps,
//          and topk_mop_config<true>() restores the merge MOP config the template clobbered.
//   Unfused: every body fits the replay buffer; the per-column steps are stride 64 (K = 2048 only), 32 (K >= 1024),
//            16 and the W32 sort from stride 8.
//
// One `enter_transpose_cfg_block` / `leave_transpose_cfg_block` pair wraps both transposes.
template <std::uint32_t K, bool fused>
inline void _topk_xl_rebuild_(const std::uint32_t dst_index, const bool ascending)
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    bool dir                             = ascending;
    constexpr int num_tiles_per_sequence = K == 512 ? 1 : K == 1024 ? 1 : 2;
    const std::uint32_t tile_offset      = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];
    constexpr int row_scale_factor       = K == 512 ? 1 : K == 1024 ? 2 : 4;
    constexpr int n_iters_column_pairs   = row_scale_factor * 2;
#if TOPK_XL_FUSED_MACRO
    // The K = 512 rebuild has no step groups.
    if constexpr (fused && K >= 1024)
    {
        loadmacro_configure<detail::Stride::SIXTEEN, detail::WindowSize::THIRTY_TWO, detail::Order::DESCENDING, detail::Fused::Yes>();
    }
#endif

    if constexpr (fused)
    {
        enter_transpose_cfg_block_after_src_drain();
        transpose_faces<n_iters_column_pairs, true /* fused */, 256 /* indices_offset */>();

        // In the transposed faces, each body loads a lane's even column into LREG0-3 and its odd column into
        // LREG4-7 (GroupOffset::ODD_COLUMNS) and sorts both in `dir`: window 32, strides 8 down to 1.
        sort_column_pairs_with_mop(n_iters_column_pairs, dir);

        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        transpose_faces<n_iters_column_pairs, true /* fused */, 256 /* indices_offset */>();
        leave_transpose_cfg_block();

        canonical_big_block_columns<row_scale_factor>(tile_offset, dir);

        topk_mop_config<true>();
    }
    else
    {
        constexpr detail::IndicesOffset indices = num_tiles_per_sequence == 2 ? detail::IndicesOffset::TWO_TILES : detail::IndicesOffset::ONE_TILE;
        constexpr int indices_offset            = static_cast<int>(indices);
        enter_transpose_cfg_block_after_src_drain();
        transpose_faces<n_iters_column_pairs, fused, indices_offset>();

        // ── stride-8 + W32 sort from stride 8 build phase (both columns) ──────────
        // The unfused body is 26 instructions and fits the replay buffer whole; the second col replays the first
        // col's recording.
        window_sort_rows<
            detail::WindowSize::THIRTY_TWO,
            detail::Stride::EIGHT,
            detail::Stride::ONE,
            detail::GroupOffset::HALF_FACE,
            detail::DstAdvance::ONE_FACE,
            detail::Fused::No,
            indices>(n_iters_column_pairs, dir);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        set_dst_write_addr_offset(tile_offset + 2);
        window_sort_rows<
            detail::WindowSize::THIRTY_TWO,
            detail::Stride::EIGHT,
            detail::Stride::ONE,
            detail::GroupOffset::HALF_FACE,
            detail::DstAdvance::ONE_FACE,
            detail::Fused::No,
            indices,
            detail::ReplayMode::REPLAY>(n_iters_column_pairs, dir);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        set_dst_write_addr_offset(tile_offset + 0);

        transpose_faces<n_iters_column_pairs, fused, indices_offset>();
        leave_transpose_cfg_block();

#if TOPK_XL_UNFUSED_MACRO
        // Program the direction's macros once for the stride phases below.
        loadmacro_configure<detail::Stride::EIGHT, detail::WindowSize::THIRTY_TWO, detail::Fused::No>(dir);
#endif
        for (int col = 0; col < 2; col++)
        {
            if constexpr (row_scale_factor >= 4)
            {
                // ── stride-64 ─────────────────────────────────────────────
#if TOPK_XL_UNFUSED_MACRO
                topk_xl_unfused_macro::record_ce_full<indices_offset, 64, 8>(dir);
                for (int i = 1; i < 8; i++)
                {
                    lltt::replay(0 /*start*/, 16 /*len*/);
                }
#else
                window_sort_rows<
                    detail::WindowSize::THIRTY_TWO,
                    detail::Stride::EIGHT,
                    detail::Stride::EIGHT,
                    detail::GroupOffset::FOUR_FACES,
                    detail::DstAdvance::HALF_FACE,
                    detail::Fused::No,
                    indices>(8 /*iterations*/, dir);
#endif
                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
            }
            if constexpr (row_scale_factor >= 2)
            {
                // ── stride-32 — the last body of each group folds the +8 INCRWC into its LREG7 store
                unfused_step_groups<indices, detail::GroupOffset::TWO_FACES>(row_scale_factor >> 1, dir);
                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
            }

            // ── stride-16 — the last body of each group folds the +8 INCRWC into its LREG7 store
            unfused_step_groups<indices, detail::GroupOffset::ONE_FACE>(row_scale_factor, dir);
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

            // ── stride-8 + W32 sort from stride 8 ─────────────────────────
            window_sort_rows<
                detail::WindowSize::THIRTY_TWO,
                detail::Stride::EIGHT,
                detail::Stride::ONE,
                detail::GroupOffset::HALF_FACE,
                detail::DstAdvance::ONE_FACE,
                detail::Fused::No,
                indices>(n_iters_column_pairs, dir);
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

            set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
        }

        // The unfused path does not reprogram the MOP Expander, so the merge MOP config from `_topk_xl_init_` is
        // still live.
    }

    // Clear SrcA / SrcB valids — the next caller might issue a DST transpose, which requires both to be invalid.
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

/**
 * Every 64 row column must already be bitonic. Needs _topk_xl_init_<1024, true> and the fused macros of the
 * _topk_xl_merge_<1024, true> run just before it, with no unfused phase between: it does not program them. It
 * overwrites replay slots [0, 16) and issues no CLR_AB, so it must not consume the SrcB dummy valid of _topk_xl_rebuild_.
 */
template <std::uint32_t K>
inline void _topk_xl_rebuild_columns_(const std::uint32_t dst_index, const bool ascending)
{
    static_assert(K == 1024, "K must be 1024: the column rebuild sorts the 64 row columns of one tile");
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];
    for (int col = 0; col < 2; col++)
    {
        canonical_big_block_with_replay<2>(ascending);
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
}

// =============================================================================
//  Index injection / extraction
// =============================================================================
//
// These two pairs are what make the "fused" representation work:
//   * `add_lsb_indices`   : write each element's global index into the low
//                           16 bits of its DST word, just after the input
//                           tiles have been copied in.
//   * `remove_msb_values` : strip the high 16 bits (the value bits) at the
//                           end, leaving only the index in each DST word so
//                           the caller can pack the indices to output.

// Programs the ADDR_MODs used by `_topk_xl_add_lsb_indices_`. Reuses
// ADDR_MOD_4 for a +16 advance — the other ADDR_MOD_4 user is the unfused
// topk path, but `distributed_topk` runs fused, so the reuse is safe.
inline void _topk_xl_add_lsb_indices_init_()
{
    // ADDR_MOD_6 — +4 (a quarter face).
    detail::set_dst_addr_mode<ADDR_MOD_6, QUARTER_FACE_INC>();

    // ADDR_MOD_4 — +16 (one face).
    // Completes the advance to the next face pair after the inner loop.
    // Replaces two +8 no-op SFPLOADs with one +16 issue.
    detail::set_dst_addr_mode<ADDR_MOD_4, FACE_INC>();
}

// Builds the per-element global index in LREG0..LREG3 and ORs it into the
// low 16 bits of each fused payload in the first two DST tiles.
//
// The 16-bit index layout is:
//
//   bits [15:11] — core_id (5 bits, up to 32 cores)
//   bits [10: 5] — row    (6 bits, derived from the tile id)
//   bits [ 4: 0] — within-row column (5 bits, lane id)
//
// After this routine each DST word reads as `[ bf16 value | u16 index ]`.
// `row_major` (default false) selects the within-chunk index ordering. The
// default numbers the two [32, 32] tiles as a 64-row x 32-col grid in
// COLUMN-MAJOR order: index(tile, row, col) = col*64 + tile*32 + row. Set
namespace detail
{

/**
 * @brief OR LREG0-3 into the low half of the four Dest rows at +0, +2, +16 and +18, then add STEP to LREG0-3.
 *
 * 16 instructions: 4 loads, 4 ORs, 4 IADDs, 4 stores. Callers record it once into replay slots [0, 16) and replay it.
 *
 * @tparam STEP: Index advance per block, values = <4 column-major / 64 row-major>
 * @tparam LAST_ADDR_MOD: ADDR_MOD of the last store, which advances Dst to the next block
 */
template <int STEP, std::uint32_t LAST_ADDR_MOD>
inline void or_lsb_index_block()
{
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, 16 + 0);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, 16 + 2);

    TTI_SFPOR(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
    TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG5, 0);
    TTI_SFPOR(0, p_sfpu::LREG2, p_sfpu::LREG6, 0);
    TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG7, 0);

    TTI_SFPIADD(STEP, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(STEP, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(STEP, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(STEP, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 2);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, 16 + 0);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, LAST_ADDR_MOD, 16 + 2);
}

/**
 * @brief Column-major index stamp shared by @ref _topk_xl_add_lsb_indices_ and @ref _topk_xl_add_lsb_indices_rt_.
 *
 * @tparam LAST_ADDR_MOD: ADDR_MOD of each block's last store
 * @tparam SKIP_LREG: Destination of the load that only advances Dst by a face-pair between outer iterations
 * @param load_id: Puts the 5-bit chunk id in LREG1 (immediate or runtime)
 */
template <std::uint32_t K, std::uint32_t LAST_ADDR_MOD, std::uint32_t SKIP_LREG, typename LoadId>
inline void add_lsb_indices_column_major(LoadId load_id)
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Tile ID into LREG0 (across the 32 lanes: 0, 2, 4, ..., 60, 62).
    TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG0, 0);

    // LREG1 = LREG0 + 1, LREG2 = LREG0 + 16, LREG3 = LREG0 + 17 — give the
    // four index variants we need across the four-row LREG block.
    TTI_SFPIADD(1, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(16, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(17, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Multiply by 64 (one row per 64 lanes), so per-column start values are
    // (0, 64, 128, 192, ..., 1920, 1984).
    // NOTE: SFPSHFT_MOD1_SHIFT_IMM in sfpi is buggy — passing 1 as the imm
    // arg is the documented workaround.
    TTI_SFPSHFT(6, 0, p_sfpu::LREG0, 1);

    // Load the id and place it in bits [15:11] of every lane.
    load_id();
    TTI_SFPSHFT(11, 0, p_sfpu::LREG1, 1);

    // Merge core_id into LREG0, then propagate to LREG1..3 with the same
    // +1 / +16 / +17 offsets so all four LREGs hold their final 16-bit
    // index values.
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(1, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(2, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(3, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // ── OR the precomputed indices into the low 16 bits of every DST word.
    // The body (12 instructions: 4 loads + 4 ORs + 4 stores + 4 IADDs) is
    // recorded into replay slots [0..15] once and replayed for the
    // remaining iters.
    lltt::record<lltt::Exec>(0, 16);
    or_lsb_index_block<4, LAST_ADDR_MOD>();

    for (int i = 1; i < 4; i++)
    {
        lltt::replay(0, 16);
    }

    // Outer loop over the remaining face-pairs. We hoist the
    // "skip-a-face-pair" SFPLOAD to the prologue of each subsequent outer
    // iter so the trailing iter doesn't pay for it: the next caller
    // (`_topk_xl_local_sort_`) starts with TTI_SETRWC(SET_D), which would
    // clobber RWC anyway.
    //
    // Using ADDR_MOD_4 (+16) collapses what would otherwise be two +8
    // SFPLOADs into a single issue.
    //
    // Iteration count is K-derived:
    //   K=512  → 1 face-pair total (just the initial recording above); no
    //            extra outer iters needed.
    //   K=1024 → 2 face-pairs total → 1 extra outer iter here.
    //   K=2048 → 4 face-pairs total → 3 extra outer iters here.
    constexpr int row_scale_factor = K == 512 ? 1 : K == 1024 ? 2 : 4;
    for (int j = 1; j < row_scale_factor; j++)
    {
        TTI_SFPLOAD(SKIP_LREG, InstrModLoadStore::INT32, ADDR_MOD_4, 0);

        for (int i = 0; i < 4; i++)
        {
            lltt::replay(0, 16);
        }
    }
}

} // namespace detail

// `row_major = true` for a plain ROW-MAJOR order: index = tile*1024 + row*32 + col.
template <std::uint32_t K, std::uint32_t core_id, bool row_major = false>
inline void _topk_xl_add_lsb_indices_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");

    if constexpr (row_major)
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        // Tile ID into LREG0 (across the 32 lanes: 0, 2, 4, ..., 60, 62).
        TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG0, 0);

        // Load core_id and place it in bits [15:11] of every lane.
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, core_id);
        TTI_SFPSHFT(11, 0, p_sfpu::LREG1, 1);
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);

        // LREG1 = LREG0 + 1, LREG2 = LREG0 + 256, LREG3 = LREG0 + 257 — give the
        // four index variants we need across the four-row LREG block.
        TTI_SFPIADD(1, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(257, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);

        // ── OR the precomputed indices into the low 16 bits of every DST word.
        // Same 16-slot record/replay shape as the column-major body below.
        lltt::record<lltt::Exec>(0, 16);
        detail::or_lsb_index_block<64, ADDR_MOD_6>();

        for (int i = 1; i < 4; i++)
        {
            lltt::replay(0, 16);
        }

        // Outer loop over the remaining face-pairs, same shape as the
        // column-major body below; the row-major index advance between
        // face-pairs is +256 (tile*1024 spans two 512-element face-pairs).
        constexpr int row_scale_factor = K == 512 ? 1 : K == 1024 ? 2 : 4;
        for (int j = 1; j < row_scale_factor; j++)
        {
            TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPIADD(256, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPIADD(256, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPIADD(256, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPLOAD(topk_xl_blaze_compat ? p_sfpu::LREG4 : p_sfpu::LCONST_0, InstrModLoadStore::INT32, ADDR_MOD_4, 0);

            for (int i = 0; i < 4; i++)
            {
                lltt::replay(0, 16);
            }
        }
        return;
    }

    detail::add_lsb_indices_column_major<K, ADDR_MOD_6, topk_xl_blaze_compat ? p_sfpu::LREG4 : p_sfpu::LCONST_0>(
        [] { TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, core_id); });
}

// Runtime-chunk-id variant of `_topk_xl_add_lsb_indices_`: the 5-bit id in bits [15:11] comes from a runtime argument
// (composed via TT_SFPLOADI) instead of the template parameter, so one stamp instantiation serves every chunk of a fused
// end-to-end row. The rest is the shared `add_lsb_indices_column_major`.
template <std::uint32_t K, std::uint8_t advance_mod = ADDR_MOD_6>
inline void _topk_xl_add_lsb_indices_rt_(const std::uint32_t chunk_id)
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    // The id is masked to the 5-bit field at composition: an out-of-contract id can only wrap within the index field,
    // never spill into the BF16 value half (RISC-side arithmetic on the composed immediate, zero SFPU cost).
    detail::add_lsb_indices_column_major<K, advance_mod, p_sfpu::LCONST_0>([chunk_id]
                                                                           { TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, chunk_id & 0x1Fu); });
}

// =============================================================================
//  Post-reduction PACK-thread phases (`remove_msb_values`, `separate_indices`)
// =============================================================================
//
// These two helpers run AFTER all merge/rebuild work has finished, so they
// are free to reprogram ADDR_MODs without disturbing the bitonic hot path.
// To keep that contract explicit they touch ADDR_MOD_0 (which the bitonic
// init never programs) rather than reusing a slot that the +24 / +40 / +48
// folds depend on.
//
// `remove_msb_values` runs on the PACK thread (TRISC2): the >64K (extended
// 256K) flow needs to pack values out first, then overwrite the value half
// of each DST word with zero, then pack out indices.

// Program ADDR_MOD_0 with a +2 increment: the two Dest rows of an SFPU
// load/store. Used by both `remove_msb_values` and `separate_indices`,
// which both walk the K-element region 32 datums at a time. Also sets
// ADDR_MOD_7 (zero advance) for the load part of the sequence.
inline void _topk_xl_remove_msb_values_init_()
{
    // ADDR_MOD_0 — +2 (one 32-lane group, an eighth of a face).
    // Not programmed by _topk_xl_init_; independent of ADDR_MOD_2's +24 fold.
    detail::set_dst_addr_mode<ADDR_MOD_0, EIGHTH_FACE_INC>();

    if constexpr (!topk_xl_blaze_compat)
    {
        // ADDR_MOD_7 — no advance, used by SFPLOAD.
        detail::set_dst_addr_mode<ADDR_MOD_7, 0 /* Increment */>();
    }
}

// Store zero into the hi16 half of every DST word in the K-sized region.
// Runs on PACK (TRISC2): on the extended 256K path the value half has
// already been packed out by the time we get here, so blanking the hi16
// half leaves clean indices in DST ready for the second pack.
//
// K=2048 → 64 iters; K=1024 → 32 iters; K=512 → 16 iters, 32 lanes each.
template <std::uint32_t K, DstSync Dst>
inline void _topk_xl_remove_msb_values_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");

    constexpr int row_scale_factor = K == 512 ? 1 : K == 1024 ? 2 : 4;
    if constexpr (topk_xl_blaze_compat)
    {
        // Caller must quiesce MATH and retain ADDR_MOD_7 = zero advance while
        // PACK owns LREG0 and replay slots [0, 3). SyncHalf alone is insufficient.
        lltt::record<lltt::Exec>(0, 3);
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, 0);
        for (int i = 1; i < row_scale_factor * 16; i++)
        {
            lltt::replay(0, 3);
        }
    }
    else
    {
        // The loop below holds a value in LREG0 across instructions while running on
        // PACK, and MATH does SFPU work itself, so this is only safe under SyncFull.
        static_assert(Dst == DstSync::SyncFull, "_topk_xl_remove_msb_values_ needs MATH quiesced: SyncFull only");

        // Drain any SFPU work still in flight before touching LREG0.
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::WAIT_SFPU);

        for (int i = 0; i < row_scale_factor * 16; i++)
        {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);  // fused [value|index]
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0);            // clear hi16, keep the index
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, 0); // [0|index], advance one group
        }

        // Hold the packer until those stores drain.
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
    }
}

// In addition to programming ADDR_MOD_0, this also stashes the runtime
// `group_id_bit_shift` into LREG12 so that `_topk_xl_separate_indices_`
// can SFPSHFT the (template-static) `group_id` by it at body issue time.
inline void _topk_xl_separate_indices_init_(std::uint32_t group_id_bit_shift)
{
    detail::set_dst_addr_mode<ADDR_MOD_0, EIGHTH_FACE_INC>();

    // Save the runtime group_id bit shift into a programmable SFPU constant
    // register (LREG12), so _topk_xl_separate_indices_ can SFPSHFT by it.
    _sfpu_load_config32_(sfpi::SFPCONFIG_DEST_LREG12, 0 /* upper16 */, detail::lower16(group_id_bit_shift));
}

// ============================================================================
// TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split support
// ============================================================================
//
// The base `_topk_xl_separate_indices_` helper keeps the low-16 index field as a
// tile/SFPU coordinate. This TTNN op needs public row-major UINT32 indices, so
// the helpers in this block decode that coordinate to a within-row position,
// track the input chunk base, and write the result into the existing unfused
// INT32 index region consumed by merge/rebuild.
inline void _topk_xl_separate_indices_row_major_init_(std::uint32_t chunk_base)
{
    detail::set_dst_addr_mode<ADDR_MOD_0, EIGHTH_FACE_INC>();

    _sfpu_load_config32_(sfpi::SFPCONFIG_DEST_LREG12, detail::upper16(chunk_base), detail::lower16(chunk_base));
    _sfpu_load_config32_(sfpi::SFPCONFIG_DEST_LREG13, 0 /* upper16 */, 0x000F /* lower16 */);
    _sfpu_load_config32_(sfpi::SFPCONFIG_DEST_LREG14, 0 /* upper16 */, 0x0001 /* lower16 */);
}

template <std::uint32_t chunk_base_upper16>
inline void _topk_xl_separate_indices_row_major_init_upper_(std::uint32_t chunk_base_low16)
{
    detail::set_dst_addr_mode<ADDR_MOD_0, EIGHTH_FACE_INC>();

    _sfpu_load_config32_(p_sfpu::LREG12, chunk_base_upper16 /* upper16 */, chunk_base_low16 & 0xFFFF /* lower16 */);
    _sfpu_load_config32_(p_sfpu::LREG13, 0 /* upper16 */, 0x000F /* lower16 */);
    _sfpu_load_config32_(p_sfpu::LREG14, 0 /* upper16 */, 0x0001 /* lower16 */);
}

template <std::uint32_t chunk_base_upper16, std::uint32_t chunk_base_lower16>
inline void _topk_xl_separate_indices_row_major_init_static_()
{
    detail::set_dst_addr_mode<ADDR_MOD_0, EIGHTH_FACE_INC>();

    _sfpu_load_config32_(p_sfpu::LREG12, chunk_base_upper16 /* upper16 */, chunk_base_lower16 /* lower16 */);
    _sfpu_load_config32_(p_sfpu::LREG13, 0 /* upper16 */, 0x000F /* lower16 */);
    _sfpu_load_config32_(p_sfpu::LREG14, 0 /* upper16 */, 0x0001 /* lower16 */);
}

inline void _topk_xl_separate_indices_row_major_reinit_()
{
    detail::set_dst_addr_mode<ADDR_MOD_0, EIGHTH_FACE_INC>();
}

template <std::uint32_t K>
inline void _topk_xl_separate_indices_row_major_advance_chunk_base_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    TTI_SFPMOV(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, K);
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, 4);
    TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);
}

// In:
//   LREG0 raw low bits: [col bits at 10:6 | tile bit at 5 | row bits at 4:0]
// Out:
//   LREG0: row-major within-chunk index.
//
// K=1024 ignores raw bit 5. K=2048 maps raw bit 5 to row-major bit 10.
template <std::uint32_t K>
inline void _topk_xl_decode_row_major_index_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");

    // part0 = col_low4: ((raw >> 6) & 0xf)
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
    TTI_SFPSHFT((-6) & 0xFFF, p_sfpu::LREG2, p_sfpu::LREG2, 1);
    TTI_SFPAND(0, p_sfpu::LREG13, p_sfpu::LREG2, 0);

    // part1 = row_low4 << 4
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
    TTI_SFPAND(0, p_sfpu::LREG13, p_sfpu::LREG3, 0);
    TTI_SFPSHFT(4, p_sfpu::LREG3, p_sfpu::LREG3, 1);
    TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);

    // part2 = col_hi << 8
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
    TTI_SFPSHFT((-10) & 0xFFF, p_sfpu::LREG3, p_sfpu::LREG3, 1);
    TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG3, 0);
    TTI_SFPSHFT(8, p_sfpu::LREG3, p_sfpu::LREG3, 1);
    TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);

    // part3 = row_hi << 9
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
    TTI_SFPSHFT((-4) & 0xFFF, p_sfpu::LREG3, p_sfpu::LREG3, 1);
    TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG3, 0);
    TTI_SFPSHFT(9, p_sfpu::LREG3, p_sfpu::LREG3, 1);
    TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);

    if constexpr (K == 2048)
    {
        // part4 = tile_in_sequence << 10
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
        TTI_SFPSHFT((-5) & 0xFFF, p_sfpu::LREG3, p_sfpu::LREG3, 1);
        TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG3, 0);
        TTI_SFPSHFT(10, p_sfpu::LREG3, p_sfpu::LREG3, 1);
        TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);
    }

    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
}

template <std::uint32_t K>
inline void _topk_xl_separate_indices_row_major_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr int row_scale_factor       = K == 512 ? 1 : K == 1024 ? 2 : 4;
    constexpr int num_tiles_per_sequence = K == 512 ? 1 : K == 1024 ? 1 : 2;
    constexpr int indices_offset         = num_tiles_per_sequence * 64;

    for (int i = 0; i < row_scale_factor * 16; i++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);

        // Value region: keep BF16 value high half and clear low half.
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, 0);

        // Index region: clear high half, decode low tile coordinate, OR chunk base.
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0);
        _topk_xl_decode_row_major_index_<K>();
        TTI_SFPOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);
    }
}

// Fused end-to-end variant: like `_topk_xl_separate_indices_row_major_init_`,
// but LREG12 carries the chunk-id FIELD MASK (bits [15:11]) instead of a
// chunk base — the global index is recovered from the stamp itself, so no
// per-chunk base bookkeeping exists at all.
inline void _topk_xl_separate_indices_row_major_global_init_()
{
    detail::set_dst_addr_mode<ADDR_MOD_0, EIGHTH_FACE_INC>();

    _sfpu_load_config32_(p_sfpu::LREG12, 0 /* upper16 */, 0xF800 /* lower16 */);
    _sfpu_load_config32_(p_sfpu::LREG13, 0 /* upper16 */, 0x000F /* lower16 */);
    _sfpu_load_config32_(p_sfpu::LREG14, 0 /* upper16 */, 0x0001 /* lower16 */);
}

namespace detail
{

/**
 * @brief Loop of the global index split: per pair of Dest rows, keep the BF16 value half in place and write the
 * decoded global index (chunk id rescaled to chunk_id * K, plus the within-chunk coordinate) to the index region.
 *
 * @tparam OR_SEGMENT_BASE: Also OR the segment base held in LREG5 into each index
 */
template <std::uint32_t K, bool OR_SEGMENT_BASE>
inline void separate_indices_global_loop()
{
    constexpr int row_scale_factor       = K == 512 ? 1 : K == 1024 ? 2 : 4;
    constexpr int num_tiles_per_sequence = K == 512 ? 1 : K == 1024 ? 1 : 2;
    constexpr int indices_offset         = num_tiles_per_sequence * 64;
    // chunk_id sits at [15:11] = chunk_id * 2048; rescale to chunk_id * K.
    constexpr int chunk_field_shift = K == 2048 ? 0 : K == 1024 ? -1 : -2;

    for (int i = 0; i < row_scale_factor * 16; i++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);

        // Value region: keep BF16 value high half and clear low half.
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, 0);

        // Index region: clear high half, save the raw u16 (incl. chunk id),
        // decode the within-chunk coordinate, then OR the rescaled chunk
        // field back in.
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0);
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
        _topk_xl_decode_row_major_index_<K>();
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG4, 0);
        if constexpr (chunk_field_shift != 0)
        {
            TTI_SFPSHFT(chunk_field_shift & 0xFFF, p_sfpu::LREG4, p_sfpu::LREG4, 1);
        }
        TTI_SFPOR(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        if constexpr (OR_SEGMENT_BASE)
        {
            TTI_SFPOR(0, p_sfpu::LREG5, p_sfpu::LREG0, 0);
        }

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);
    }
}

} // namespace detail

// Fused end-to-end split: runs ONCE per row on the final fused survivor
// (instead of once per chunk pre-merge). Each u16 payload carries
// [chunk_id 15:11 | within-chunk coordinate 10:0]; the global index is
// chunk_id * K + row-major(within-chunk). For K=2048 the stamp field IS
// chunk_id * 2048; narrower windows shift it down to the right weight.
// Requires `_topk_xl_separate_indices_row_major_global_init_` (LREG12 =
// 0xF800). Only sound for rows of <= 32 chunks (5-bit id) — the factory
// gates FUSED_E2E on the padded chunk count.
template <std::uint32_t K>
inline void _topk_xl_separate_indices_row_major_global_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");

    detail::separate_indices_global_loop<K, false>();
}

// Segment-base variant of the global split (segmented fusion for rows wider
// than 32 chunks): identical decode, then ORs a runtime segment base into the
// index word. seg_base = seg * 32 * K is a multiple of 2^(5+log2(K)) while the
// decoded local index occupies exactly those low bits, so OR == ADD with zero
// bit overlap. LREG5 is free across the loop (body uses LREG0-4, constants
// LREG12-14) and is loaded once per call.
template <std::uint32_t K>
inline void _topk_xl_separate_indices_row_major_global_base_(std::uint32_t seg_base)
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");

    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, (seg_base >> 16) & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, seg_base & 0xFFFF);

    detail::separate_indices_global_loop<K, true>();
}

// ============================================================================
// END TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split support
// ============================================================================

// Split the packed `[ bf16 value | u16 index ]` word into a separate
// values-only region and an index-only region with the "extended"
// index `[group_id << group_id_bit_shift | u16 index]`. Used at stage 5
// of the >64K reduction tree to switch the remaining stages into unfused
// mode. The bit position where `group_id` is placed is set at init time
// via `_topk_xl_separate_indices_init_(group_id_bit_shift)`, which stores
// that shift into LREG12 so this body can SFPSHFT by it.
//
// Body per face-pair (9 instructions):
//   SFPLOAD  LREG0 ← DST[+0]                       (value+index word)
//   SFPMOV   LREG1 ← LREG0                         (copy)
//   SFPLOADI LREG1 imm-low  0                      (zero lo16 of LREG1 → keeps hi16 = value)
//   SFPLOADI LREG0 imm-high 0                      (clear LREG0 hi16; keeps lo16 = u16 index)
//   SFPLOADI LREG2 imm-ushort group_id             (LREG2 = zero-extended group_id, lo16 = group_id)
//   SFPSHFT  LREG2 ← LREG2 << LREG12               (shift group_id by runtime bit position)
//   SFPOR    LREG0 |= LREG2                        (LREG0 = [group_id << shift | u16 index])
//   SFPSTORE LREG1 → DST[+0]                       (values)
//   SFPSTORE LREG0 → DST[+indices_offset]          (extended indices, advances Dst by +2)
//
// `indices_offset` is K-derived: the index region sits one sorted-sequence
// worth of tiles away from the values region. For K=2048 (2 tiles per
// sequence) that's +128; for K=512/1024 (1 tile per sequence) it's +64.
//
// Recording-then-replaying this 9-instruction body saves 8 × (n-1) TRISC1
// issues vs an inline n-iter loop, at zero correctness risk because the
// body is fully template-static (`group_id` is a template parameter) and
// the per-call shift amount has already been latched into LREG12.
template <std::uint32_t K, std::uint32_t group_id>
inline void _topk_xl_separate_indices_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr int row_scale_factor       = K == 512 ? 1 : K == 1024 ? 2 : 4;
    constexpr int num_tiles_per_sequence = K == 512 ? 1 : K == 1024 ? 1 : 2;
    constexpr int indices_offset         = num_tiles_per_sequence * 64;

    lltt::record<lltt::Exec>(0, 9);
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, 0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, group_id);
    TTI_SFPSHFT(0, p_sfpu::LREG12, p_sfpu::LREG2, 0);
    TTI_SFPOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);

    for (int i = 1; i < row_scale_factor * 16; i++)
    {
        lltt::replay(0, 9);
    }
}

// Split fused K = 512, 1024 and 2048 chunks: each SFPU segment on PACK is the work between two face transposes of the
// local sort, merge or rebuild, in the same instruction order; MATH runs the transposes and the SrcA/SrcB releases.

// Column passes per sequence, and the sorted-run merge iterations (two per pass).
template <std::uint32_t K>
constexpr int split_row_scale = K == 512    ? 1
                                : K == 1024 ? 2
                                            : 4;
template <std::uint32_t K>
constexpr int split_n_iters = 2 * split_row_scale<K>;

// PACK, once per row: ADDR_MOD_3 takes the stamp's +4 so ADDR_MOD_6 keeps the sort's +32; ADDR_MOD_4 is the K >= 1024
// stamp's face skip. With `macros` (default K >= 1024) the row's segments share one fused macro configuration,
// programmed here. The column-pair len 8 and len 16 passes program their own swap layer and put the step-group one
// back; MATH must issue no SFPLOADMACRO while theirs is installed.
template <std::uint32_t K, bool macros = (K >= 1024)>
inline void _topk_xl_split_sfpu_init_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024 or 2048");
    _init_sfpu_config_reg();

    detail::set_dst_addr_mode<ADDR_MOD_7, 0>();
    detail::set_dst_addr_mode<ADDR_MOD_6, TWO_FACE_INC>();
    detail::set_dst_addr_mode<ADDR_MOD_5, FACE_INC>();
    detail::set_dst_addr_mode<ADDR_MOD_1, THREE_FACE_INC>();
    detail::set_dst_addr_mode<ADDR_MOD_3, QUARTER_FACE_INC>();

    if constexpr (K >= 1024)
    {
        detail::set_dst_addr_mode<ADDR_MOD_4, FACE_INC>();
    }
#if TOPK_XL_FUSED_MACRO
    if constexpr (macros)
    {
        loadmacro_configure<detail::Stride::SIXTEEN, detail::WindowSize::THIRTY_TWO, detail::Order::DESCENDING, detail::Fused::Yes>();
    }
#endif
}

inline void _topk_xl_split_begin_(const std::uint32_t tile_offset)
{
    set_dst_write_addr_offset(tile_offset);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

// _topk_xl_add_lsb_indices_rt_<K> with its +4 store on ADDR_MOD_3.
template <std::uint32_t K>
inline void _topk_xl_split_stamp_(const std::uint32_t tile_offset, const std::uint32_t chunk_id)
{
    set_dst_write_addr_offset(tile_offset);
    _topk_xl_add_lsb_indices_rt_<K, ADDR_MOD_3>(chunk_id);
}

/**
 * @brief Local sort of one chunk up to its first face transpose: runs of 32, 64 and 128 down each column, then runs of
 * 256 across each lane's even and odd column.
 *
 * See @ref detail::local_sort_head.
 *
 * @tparam K: Elements per chunk, values = <512/1024/2048>
 * @tparam macros: Use the fused SFPLOADMACRO bodies, which @ref _topk_xl_split_sfpu_init_ must have programmed; only
 *         K = 512 runs without them
 * @param tile_offset: Dst offset of the chunk's first tile
 * @param ascending: Direction of the first run
 * @note PACK half of the split local sort; MATH runs the face transpose that follows.
 */
template <std::uint32_t K, bool macros = true>
inline void _topk_xl_split_sort_head_(const std::uint32_t tile_offset, const bool ascending)
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024 or 2048");
    static_assert(K == 512 || macros, "K = 1024 and 2048 always use the fused macros");
#if TOPK_XL_FUSED_MACRO
    constexpr detail::UseLoadMacro use_loadmacro = macros ? detail::UseLoadMacro::Yes : detail::UseLoadMacro::No;
#else
    constexpr detail::UseLoadMacro use_loadmacro = detail::UseLoadMacro::No;
#endif
    _topk_xl_split_begin_(tile_offset);
    detail::local_sort_head<K, use_loadmacro, detail::EarlyExit::No>(tile_offset, ascending);
}

// The column-pair pass between two transposes.
//
// `macros` must match what `_topk_xl_split_sfpu_init_` was given: the sort_len 8 and 16 paths below
// host their first swap layer on `SFPLOADMACRO` and finish by putting the step-group configuration
// back, which only a caller initialised with macros expects. The default is OFF and such a caller
// opts in, as `_topk_xl_split_sort_head_<512, macros>` already does.
template <std::uint32_t K, int sort_len, bool macros = false>
inline void _topk_xl_split_column_pairs_(const std::uint32_t tile_offset, const bool dir)
{
    static_assert(sort_len == 4 || sort_len == 8 || sort_len == 16, "sort_len must be 4, 8 or 16");
    constexpr int n_iters = split_n_iters<K>;
    _topk_xl_split_begin_(tile_offset);
#if TOPK_XL_FUSED_MACRO
    if constexpr (macros && sort_len == 16)
    {
        // Step 4 of `window_sort<THIRTY_TWO, EIGHT, FOUR>` rides the loads, same as the sort_len == 8 case.
        // sort_len == 4 is deliberately absent: its dispatch issues an SFPTRANSP before the sort, so
        // the first thing after the load is a transpose and there is no swap for a macro to host.
        constexpr int stores_at = topk_xl_fused_macro::column_pair_len16_loads_len;
        loadmacro_configure<detail::Stride::EIGHT, detail::WindowSize::THIRTY_TWO, detail::Fused::Yes>(dir);
        topk_xl_fused_macro::record_column_pair_len16_loads(dir);
        window_sort<detail::WindowSize::THIRTY_TWO, detail::Stride::FOUR, detail::Stride::ONE>(dir);
        load_replay_buf<Exec>(stores_at, 8, [] { store16_rows_x2<2, detail::DstAdvance::ONE_FACE>(); });
        for (int i = 1; i < n_iters; i++)
        {
            lltt::replay(0, topk_xl_fused_macro::column_pair_len16_loads_len);
            window_sort<detail::WindowSize::THIRTY_TWO, detail::Stride::FOUR, detail::Stride::ONE>(dir);
            lltt::replay(stores_at, 8);
        }
        // Put the step-group configuration back for the later step groups.
        loadmacro_configure<detail::Stride::SIXTEEN, detail::WindowSize::THIRTY_TWO, detail::Order::DESCENDING, detail::Fused::Yes>();
    }
    else if constexpr (macros && sort_len == 8)
    {
        // Step 3 rides the loads instead of costing four explicit swaps, each of which would stall a
        // cycle (`SFPSWAP.md:112`). Only this first layer can be hosted -- the later layers have no
        // load left to attach a macro to. See `column_pair_len8_loads` for the two encoding facts it
        // needs, and note the templates must be put back before the next `step_group` runs.
        constexpr int stores_at = topk_xl_fused_macro::column_pair_len8_loads_len;
        loadmacro_configure<detail::Stride::FOUR, detail::WindowSize::EIGHT, detail::Fused::Yes>(dir);
        topk_xl_fused_macro::record_column_pair_len8_loads(dir);
        window_sort<detail::WindowSize::EIGHT, detail::Stride::TWO, detail::Stride::ONE>(dir);
        load_replay_buf<Exec>(stores_at, 8, [] { store16_rows_x2<2, detail::DstAdvance::ONE_FACE>(); });
        for (int i = 1; i < n_iters; i++)
        {
            lltt::replay(0, topk_xl_fused_macro::column_pair_len8_loads_len);
            window_sort<detail::WindowSize::EIGHT, detail::Stride::TWO, detail::Stride::ONE>(dir);
            lltt::replay(stores_at, 8);
        }
        // Put the step-group configuration back for the later step groups.
        loadmacro_configure<detail::Stride::SIXTEEN, detail::WindowSize::THIRTY_TWO, detail::Order::DESCENDING, detail::Fused::Yes>();
    }
    else
#endif
    {
        // The even and odd column's 16-runs sort the same way, so 16 uses window 32 rather than 16.
        constexpr auto window = sort_len == 16 ? detail::WindowSize::THIRTY_TWO : static_cast<detail::WindowSize>(sort_len);
        window_sort_rows<
            window,
            static_cast<detail::Stride>(sort_len / 2),
            detail::Stride::ONE,
            detail::GroupOffset::ODD_COLUMNS,
            detail::DstAdvance::ONE_FACE,
            detail::Fused::Yes>(n_iters, dir);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

// The per-column passes between two transposes, with the lanes FLIP selects running the other way.
template <std::uint32_t K, detail::LaneFlip FLIP>
inline void _topk_xl_split_columns_(const std::uint32_t tile_offset, const bool dir)
{
    _topk_xl_split_begin_(tile_offset);
    if constexpr (FLIP != detail::LaneFlip::NONE)
    {
        set_lane_flip<FLIP>();
    }
    canonical_big_block_columns<split_row_scale<K>>(tile_offset, dir);
    if constexpr (FLIP != detail::LaneFlip::NONE)
    {
        clear_lane_flip();
    }
}

// _topk_xl_merge_<K, true> of the runs at tile_offset and `distance` rows past it (mod 1024 rows), result at tile_offset.
template <std::uint32_t K, int distance, bool macros = (K >= 1024)>
inline void _topk_xl_split_merge_(const std::uint32_t tile_offset)
{
    constexpr int tiles_per_sequence = K == 2048 ? 2 : 1;
    static_assert(
        distance > 0 && distance % (64 * tiles_per_sequence) == 0 && distance < 1024, "the other run sits whole sequences past tile_offset, mod 1024 rows");
    constexpr int n_iters = split_n_iters<K>;
    topk_mop_config<true>();
    _topk_xl_split_begin_(tile_offset);
#if TOPK_XL_FUSED_MACRO
    if constexpr (macros)
    {
        load_replay_buf<Exec>(
            0,
            16,
            []
            {
                topk_xl_fused_macro::step_group<distance, 16, false>();
                TTI_SFPNOP;
            });
    }
    else
#endif
    {
        load_replay_buf<Exec>(
            0,
            16,
            []
            {
                load16_rows_x2<distance>();
                bitonic_sort_len_k(false);
                store4_rows_top_only<16>();
            });
    }
    ckernel_unpack_template::run(n_iters - 1);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    set_dst_write_addr_offset(tile_offset + 2);
    ckernel_unpack_template::run(n_iters);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    set_dst_write_addr_offset(tile_offset);
}

// The rebuild's even/odd-column sort between its two face transposes (see `sort_column_pairs_with_mop`).
template <std::uint32_t K>
inline void _topk_xl_split_rebuild_build_(const std::uint32_t tile_offset, const bool dir)
{
    _topk_xl_split_begin_(tile_offset);
    sort_column_pairs_with_mop(split_n_iters<K>, dir);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

// MATH: one transpose of the K-word run at tile_offset; the caller holds the transpose CFG block open.
template <std::uint32_t K>
inline void _topk_xl_split_transpose_(const std::uint32_t tile_offset)
{
    // A SrcA/SrcB release just before must land before the transpose tests the banks.
    TTI_STALLWAIT(p_stall::STALL_CFG | p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    set_dst_write_addr_offset(tile_offset);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    transpose_N_faces<2 * split_row_scale<K>, true, 256, false>();
}

// The first fused chunk in each row runs the full TopK init. Later chunks can
// use this hot-path reinit because copy init only clobbers ADDR_MOD_0/3 and the
// MOP, while index stamping additionally changes ADDR_MOD_6 from +32 to +4.
// Fused sort/merge never consumes ADDR_MOD_0/3/4; ADDR_MOD_1/5 and the other
// TopK state remain live. Restoring +32 and the fused MOP therefore avoids the
// two redundant ADDR_MOD writes in every subsequent chunk's full init.
inline void _topk_large_indices_reinit_fused_after_stamp_()
{
    detail::set_dst_addr_mode<ADDR_MOD_6, TWO_FACE_INC>();
    topk_mop_config<true>();
}

// The final TopK XL survivor stores FP32 value words followed by UINT32 index
// words. Mark indices paired with exact -inf values as invalid without
// disturbing the engine-specific survivor layout.
inline void _topk_large_indices_mark_neginf_indices_init_()
{
    detail::set_dst_addr_mode<ADDR_MOD_0, EIGHTH_FACE_INC>();
}

template <std::uint32_t K>
inline void _topk_large_indices_mark_neginf_indices_()
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr std::uint32_t tiles_per_sequence = K == 2048 ? 2 : 1;
    constexpr std::uint32_t indices_offset     = tiles_per_sequence * 64;
    constexpr std::uint32_t iterations         = (K == 512 ? 1 : K == 1024 ? 2 : 4) * 16;

    // LREG2 = fp32 -inf, LREG3 = the invalid-index marker.
    constexpr std::uint32_t neg_inf_lo         = 0x0000;
    constexpr std::uint32_t neg_inf_hi         = 0xFF80;
    constexpr std::uint32_t invalid_index_half = 0xFFFF;
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, neg_inf_lo);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, neg_inf_hi);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, invalid_index_half);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, invalid_index_half);

    for (std::uint32_t i = 0; i < iterations; ++i)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

// A K = 2048 segment's unfused result from tiles 0 to 3 to tiles 4 to 7, raw words; ADDR_MOD_0 steps two rows.
inline void _topk_large_indices_move_segment_result_()
{
    constexpr std::uint32_t rows_per_tile      = 64;
    constexpr std::uint32_t result_tiles       = 4;
    constexpr std::uint32_t rows_per_access    = 2;
    constexpr std::uint32_t iterations         = result_tiles * rows_per_tile / rows_per_access;
    constexpr std::uint32_t destination_offset = result_tiles * rows_per_tile;
    for (std::uint32_t i = 0; i < iterations; ++i)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, destination_offset);
    }
}

} // namespace sfpu
} // namespace ckernel
