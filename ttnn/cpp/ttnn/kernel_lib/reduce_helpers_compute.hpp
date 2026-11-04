// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_types.hpp"
/**
 * @file reduce_helpers_compute.hpp
 * @brief reduce(): REDUCE_ROW, REDUCE_COL and REDUCE_SCALAR over a block of tiles, managing DST, CB
 * synchronization, init and packing.
 *
 * Call compute_kernel_hw_startup() exactly once at kernel start. With input reconfiguration enabled (the
 * default), reduce() configures every operand it reads; their tile/face geometry must match the startup
 * operands. The auxiliary CB must hold the tiles the selected algorithm reads before reduce() is called.
 *
 * Host-planned usage: let the host planner (kernel_lib/host/reduce_host.hpp) choose every parameter and issue
 * reduce<Call>(), where Call is one ttnn::kernel_lib::ReduceCallArgs decoded from the suffix the host appends after
 * the kernel's own compile-time arguments, [call_count][call_0]...[call_(call_count - 1)]. Include
 * reduce_plan_args.hpp for the device views:
 *
 *   constexpr uint32_t call_count = get_compile_time_arg_val(KERNEL_OWNED_CTA_COUNT);
 *   template <uint32_t I>
 *   using CallAt = ttnn::kernel_lib::ReduceCallAtT<KERNEL_OWNED_CTA_COUNT, I>;
 *
 *   template <uint32_t I = 0>
 *   ALWI void issue_calls() {
 *       if constexpr (I < call_count) {
 *           compute_kernel_lib::reduce<CallAt<I>>();  // caller work may run between calls
 *           issue_calls<I + 1>();
 *       }
 *   }
 *
 * call_count only bounds the walk: Call carries accumulation, final-call and partial-tile behavior explicitly, and
 * calls may repeat an input CB. A tail-capable Call selects full ([0, 0, 0]) or planned tail
 * ([height, width, batches]) work at runtime, including its auxiliary slice and AVG normalization; output lanes past
 * the valid non-reduced extent are unspecified. The writer materializes the unit's auxiliary recipe with
 * dataflow_kernel_lib::prepare_reduce_auxiliary_tiles(); compute never decodes it.
 */

namespace compute_kernel_lib {

// =============================================================================
// Configuration Types
// =============================================================================

/** Tile strides of a resident input; rows may be padded (row_stride > cols). */
struct ReduceInputMemoryLayout {
    std::uint32_t row_stride = 0;    // 0 = auto-detect from Wt (contiguous row-major)
    std::uint32_t batch_stride = 0;  // 0 = rows * row_stride; resident inputs only

    explicit constexpr ReduceInputMemoryLayout() = default;
    explicit constexpr ReduceInputMemoryLayout(std::uint32_t row, std::uint32_t batch = 0) :
        row_stride(row), batch_stride(batch) {}

    static constexpr ReduceInputMemoryLayout contiguous() { return ReduceInputMemoryLayout(); }
    static constexpr ReduceInputMemoryLayout with_row_stride(std::uint32_t s) { return ReduceInputMemoryLayout(s); }
    static constexpr ReduceInputMemoryLayout with_strides(std::uint32_t row, std::uint32_t batch) {
        return ReduceInputMemoryLayout(row, batch);
    }
};

/** Input tile block. Outputs: rows x batches (ROW), cols x batches (COL), batches (SCALAR). */
struct ReduceInputBlockShape {
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t batches;

    static constexpr ReduceInputBlockShape of(std::uint32_t r, std::uint32_t c, std::uint32_t b = 1) {
        return {r, c, b};
    }
    static constexpr ReduceInputBlockShape single() { return {1, 1, 1}; }
    static constexpr ReduceInputBlockShape row(std::uint32_t c, std::uint32_t b = 1) { return {1, c, b}; }
    static constexpr ReduceInputBlockShape col(std::uint32_t r, std::uint32_t b = 1) { return {r, 1, b}; }
};

/** Accumulator CB and DST index for cross-call accumulation. */
struct AccumulationConfig {
    std::uint32_t cb_accumulator = 0;
    std::uint32_t dst_index = 0;

    static constexpr AccumulationConfig with_cb(std::uint32_t cb, std::uint32_t dst = 0) { return {cb, dst}; }
};

/**
 * Accumulates across reduce() calls through cb_accumulator, one tile per output in output order.
 * Iteration 0 starts fresh; later iterations reload the accumulator first. Only the call marked
 * Accumulate::at_last runs post_reduce_op. A partial mode applies per call, so pass it only on the call
 * holding the short tile. MAX + REDUCE_SCALAR, and MAX + REDUCE_ROW on Quasar, are rejected.
 *
 *   for (uint32_t i = 0; i < num_blocks; ++i) {
 *       const auto acc = i + 1 == num_blocks ? Accumulate::at_last(cb_accum, i) : Accumulate::at(cb_accum, i);
 *       reduce<SUM, REDUCE_ROW>(..., acc);
 *   }
 */
struct Accumulate {
    AccumulationConfig config;
    // ReduceAlgorithm::TileSum only. Streamed and grouped REDUCE_COL input always use CopySeedPairs.
    AccumulateReloadMode reload = AccumulateReloadMode::CopySeedPairs;
    std::uint32_t iteration = 0;
    // Only the last call runs post_reduce_op. ReduceAlgorithm::TileSum also collapses within the tile only on the last
    // call, because its accumulator holds the raw sum.
    bool last = false;

    explicit constexpr Accumulate(AccumulationConfig cfg, std::uint32_t iter = 0, bool lst = false) :
        config(cfg), iteration(iter), last(lst) {}
    explicit constexpr Accumulate(std::uint32_t cb, std::uint32_t iter = 0, bool lst = false) :
        config{cb, 0}, iteration(iter), last(lst) {}

    static constexpr Accumulate at(std::uint32_t cb, std::uint32_t iter, std::uint32_t dst = 0) {
        return Accumulate(AccumulationConfig{cb, dst}, iter);
    }
    static constexpr Accumulate at_last(std::uint32_t cb, std::uint32_t iter, std::uint32_t dst = 0) {
        return Accumulate(AccumulationConfig{cb, dst}, iter, /*last=*/true);
    }

    constexpr Accumulate with_reload(AccumulateReloadMode m) const {
        Accumulate a = *this;
        a.reload = m;
        return a;
    }

    constexpr bool is_first() const { return iteration == 0; }
    constexpr bool is_last() const { return last; }
};

// =============================================================================
// Type Traits
// =============================================================================

template <typename T>
struct is_accumulate : std::false_type {};

template <>
struct is_accumulate<Accumulate> : std::true_type {};

template <typename T>
inline constexpr bool is_accumulate_v = is_accumulate<T>::value;

template <typename T>
struct is_accumulation_type : std::false_type {};

template <>
struct is_accumulation_type<NoAccumulation> : std::true_type {};

template <>
struct is_accumulation_type<Accumulate> : std::true_type {};

template <typename T>
inline constexpr bool is_accumulation_type_v = is_accumulation_type<T>::value;

template <typename T, typename = void>
struct is_post_reduce_op : std::false_type {};

template <typename T>
struct is_post_reduce_op<T, std::void_t<decltype(std::declval<T>()(std::declval<std::uint32_t>()))>> : std::true_type {
};

template <typename T>
inline constexpr bool is_post_reduce_op_v = is_post_reduce_op<T>::value;

// =============================================================================
// Main Reduce Function
// =============================================================================

/**
 * @brief Reduce a block of tiles along reduce_dim.
 *
 * @tparam reduce_type SUM, AVG, MAX or MIN
 * @tparam reduce_dim REDUCE_ROW (Ht outputs per batch), REDUCE_COL (Wt) or REDUCE_SCALAR (1)
 * @tparam input_dfb_id, scaler_dfb_id, output_dfb_id CB ids. Formats are deduced from them: Int32, bf16 MIN and
 *         Accurate fp32 use the SFPU path (ROW/COL only), where scaler_dfb_id may be REDUCE_NO_AUXILIARY_CB.
 * @tparam input_policy, reconfig_mode, fp32_mode, algorithm, within_tile See their enums in reduce_types.hpp.
 * @tparam scale_bits float32 bits of the factor each finished output is multiplied by before post_reduce_op, on the
 *         TileSum and SFPU paths (1/N for an AVG). Native requires 1.0f: its scaler tile carries the scale.
 * @param input_memory_layout Strides of a resident input (NoWaitNoPop / WaitUpfrontNoPop).
 * @param accumulate NoAccumulation or Accumulate.
 * @param post_reduce_op Called with each output's DST index before packing; with Accumulate, only on at_last.
 * @param partial_mode Non-tile-aligned reduce edge; not for REDUCE_SCALAR or the SFPU paths.
 * @param output_group REDUCE_COL only: output columns held in DEST together; 0 selects the DEST limit.
 * @param auxiliary_tile_offset First auxiliary tile of this call.
 *
 * @example
 *   // Softmax denominator: tiles stay resident, recip applied to each row sum.
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW, dfb_exps, dfb_scaler, dfb_out, ReduceInputPolicy::NoWaitNoPop>(
 *       ReduceInputBlockShape::row(Wt), ReduceInputMemoryLayout::contiguous(), NoAccumulation{},
 *       [](uint32_t dst) { recip_tile_init(); recip_tile(dst); });
 */
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    std::uint32_t input_dfb_id,
    std::uint32_t scaler_dfb_id,
    std::uint32_t output_dfb_id,
    ReduceInputPolicy input_policy = ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode reconfig_mode = ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
    ReduceFp32Mode fp32_mode = ReduceFp32Mode::Fast,
    ReduceAlgorithm algorithm = ReduceAlgorithm::Native,
    // Precedes the deduced parameters so callers passing a lambda can still set it.
    ReduceWithinTile within_tile = ReduceWithinTile::Collapse,
    std::uint32_t scale_bits = REDUCE_UNIT_SCALE_BITS,
    typename AccumulateT = NoAccumulation,
    typename PostReduceOp = NoOp>
ALWI void reduce(
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout = ReduceInputMemoryLayout::contiguous(),
    AccumulateT accumulate = AccumulateT{},
    PostReduceOp post_reduce_op = PostReduceOp{},
    ReducePartialMode partial_mode = ReducePartialMode::None,
    std::uint32_t output_group = 0,
    std::uint32_t auxiliary_tile_offset = 0);

/**
 * @brief Issue one host-planned tiled reduce call.
 *
 * Lowers every planner-selected field of `Call` into the explicit reduce() above, including
 * accumulation, scaling, partial handling, the output group and the shared auxiliary-CB slice.
 *
 * It does not perform compute-kernel startup or any surrounding CB preparation. The kernel owns when
 * this call runs and may perform arbitrary work, including refilling a reused input CB, between
 * planned calls.
 *
 * An optional caller post operation runs on each completed output tile after the plan's
 * scaling. Intermediate accumulation calls do not run it.
 *
 * @tparam Call A constexpr call descriptor such as ttnn::kernel_lib::ReduceCallArgs<CTA_OFFSET>.
 */
template <typename Call, typename PostReduceOp = NoOp>
ALWI void reduce(PostReduceOp post_reduce_op = PostReduceOp{});

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl"
