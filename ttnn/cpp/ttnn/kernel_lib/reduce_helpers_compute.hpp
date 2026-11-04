// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/id_only_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_types.hpp"
/**
 * @file reduce_helpers_compute.hpp
 * @brief reduce(): REDUCE_ROW, REDUCE_COL and REDUCE_SCALAR over a block of tiles, managing DST, CB
 * synchronization, init and packing.
 *
 * Call compute_kernel_hw_startup() exactly once at kernel start. With input reconfiguration enabled (the
 * default), reduce() configures every operand it reads; their tile/face geometry must match the startup
 * operands. The auxiliary CB must hold the tiles the selected algorithm reads before reduce() is called.
 * Construct each DataflowBuffer once at kernel scope and pass it by reference: on Quasar a DataflowBuffer
 * drains when it goes out of scope, so reduce() never constructs its own.
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

/**
 * Accumulator CB and DST index for cross-call accumulation. The kernel's accumulator DataflowBuffer is bound
 * type-erased, because on Quasar each access-pattern pair is its own DataflowBuffer type; it must outlive every
 * reduce() call using this config. A config built from a CB id alone is accepted only by the id-only reduce().
 */
struct AccumulationConfig {
    std::uint32_t cb_accumulator = 0;
    std::uint32_t dst_index = 0;
    void* accumulator = nullptr;
    void (*accumulator_wait_front)(void*, std::uint32_t) = nullptr;
    void (*accumulator_pop_front)(void*, std::uint32_t) = nullptr;

    static constexpr AccumulationConfig with_cb(std::uint32_t cb, std::uint32_t dst = 0) { return {cb, dst}; }

    template <typename DFB>
    void bind(DFB& dfb) {
        accumulator = &dfb;
        accumulator_wait_front = [](void* p, std::uint32_t n) { static_cast<DFB*>(p)->wait_front(n); };
        accumulator_pop_front = [](void* p, std::uint32_t n) { static_cast<DFB*>(p)->pop_front(n); };
    }
    void wait_front(std::uint32_t n) const { accumulator_wait_front(accumulator, n); }
    void pop_front(std::uint32_t n) const { accumulator_pop_front(accumulator, n); }
};

/**
 * Accumulates across reduce() calls through cb_accumulator, one tile per output in output order.
 * Iteration 0 starts fresh; later iterations reload the accumulator first. Only the call marked
 * Accumulate::at_last runs post_reduce_op. A partial mode applies per call, so pass it only on the call
 * holding the short tile. MAX + REDUCE_SCALAR, and MAX + REDUCE_ROW on Quasar, are rejected.
 *
 *   DataflowBuffer accum(cb_accum);  // once, at kernel scope
 *   for (uint32_t i = 0; i < num_blocks; ++i) {
 *       const auto acc = i + 1 == num_blocks ? Accumulate::at_last(accum, i) : Accumulate::at(accum, i);
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
    template <typename DFB, std::enable_if_t<is_dataflow_buffer_v<std::remove_cv_t<DFB>>, int> = 0>
    explicit Accumulate(DFB& dfb, std::uint32_t iter = 0, std::uint32_t dst = 0, bool lst = false) :
        config{dfb.get_id(), dst}, iteration(iter), last(lst) {
        config.bind(dfb);
    }

    template <typename DFB, std::enable_if_t<is_dataflow_buffer_v<std::remove_cv_t<DFB>>, int> = 0>
    static Accumulate at(DFB& dfb, std::uint32_t iter, std::uint32_t dst = 0) {
        return Accumulate(dfb, iter, dst);
    }
    template <typename DFB, std::enable_if_t<is_dataflow_buffer_v<std::remove_cv_t<DFB>>, int> = 0>
    static Accumulate at_last(DFB& dfb, std::uint32_t iter, std::uint32_t dst = 0) {
        return Accumulate(dfb, iter, dst, /*lst=*/true);
    }

    KERNEL_LIB_ID_ONLY_HELPER static constexpr Accumulate at(
        std::uint32_t cb, std::uint32_t iter, std::uint32_t dst = 0) {
        return Accumulate(AccumulationConfig{cb, dst}, iter);
    }
    KERNEL_LIB_ID_ONLY_HELPER static constexpr Accumulate at_last(
        std::uint32_t cb, std::uint32_t iter, std::uint32_t dst = 0) {
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
 * @param input_dfb, scaler_dfb, output_dfb The kernel's DataflowBuffers for the three ids.
 *        No auxiliary CB (scaler_dfb_id == REDUCE_NO_AUXILIARY_CB): pass input_dfb again as scaler_dfb. The slot
 *        is a placeholder; reduce() never waits on, reads or pops it. Passing any other buffer fails the debug
 *        ASSERT on scaler_dfb.get_id(). Native and TileSum reject REDUCE_NO_AUXILIARY_CB at compile time.
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
 *       exps, scaler, out, ReduceInputBlockShape::row(Wt), ReduceInputMemoryLayout::contiguous(), NoAccumulation{},
 *       [](uint32_t dst) { recip_tile_init(); recip_tile(dst); });
 *
 *   // Int32 row sum without an auxiliary CB: input_dfb fills the scaler slot.
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW, dfb_in, REDUCE_NO_AUXILIARY_CB, dfb_out>(
 *       in, in, out, ReduceInputBlockShape::row(Wt));
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
    typename PostReduceOp = NoOp,
    typename InputDFB,
    typename ScalerDFB,
    typename OutputDFB>
ALWI void reduce(
    InputDFB& input_dfb,
    ScalerDFB& scaler_dfb,
    OutputDFB& output_dfb,
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout = ReduceInputMemoryLayout::contiguous(),
    AccumulateT accumulate = AccumulateT{},
    PostReduceOp post_reduce_op = PostReduceOp{},
    ReducePartialMode partial_mode = ReducePartialMode::None,
    std::uint32_t output_group = 0,
    std::uint32_t auxiliary_tile_offset = 0);

// The id-only form: builds the DataflowBuffers from the ids and runs the form above. See id_only_helpers.hpp.
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
    ReduceWithinTile within_tile = ReduceWithinTile::Collapse,
    std::uint32_t scale_bits = REDUCE_UNIT_SCALE_BITS,
    typename AccumulateT = NoAccumulation,
    typename PostReduceOp = NoOp>
KERNEL_LIB_ID_ONLY_HELPER ALWI void reduce(
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout = ReduceInputMemoryLayout::contiguous(),
    AccumulateT accumulate = AccumulateT{},
    PostReduceOp post_reduce_op = PostReduceOp{},
    ReducePartialMode partial_mode = ReducePartialMode::None,
    std::uint32_t output_group = 0,
    std::uint32_t auxiliary_tile_offset = 0) {
    DataflowBuffer input_dfb(input_dfb_id);
    DataflowBuffer output_dfb(output_dfb_id);
    auto run = [&](auto& scaler_dfb) {
        reduce<
            reduce_type,
            reduce_dim,
            input_dfb_id,
            scaler_dfb_id,
            output_dfb_id,
            input_policy,
            reconfig_mode,
            fp32_mode,
            algorithm,
            within_tile,
            scale_bits>(
            input_dfb,
            scaler_dfb,
            output_dfb,
            input_block_shape,
            input_memory_layout,
            accumulate,
            post_reduce_op,
            partial_mode,
            output_group,
            auxiliary_tile_offset);
    };
    auto run_with_accumulator = [&](auto& scaler_dfb) {
        if constexpr (is_accumulate_v<AccumulateT>) {
            if (accumulate.config.accumulator == nullptr) {
                DataflowBuffer accum_dfb(accumulate.config.cb_accumulator);
                accumulate.config.bind(accum_dfb);
                run(scaler_dfb);
                return;
            }
        }
        run(scaler_dfb);
    };
    if constexpr (scaler_dfb_id == REDUCE_NO_AUXILIARY_CB) {
        // Reuse input_dfb: a second DataflowBuffer on the same id would drain it again on Quasar scope exit.
        run_with_accumulator(input_dfb);
    } else {
        DataflowBuffer scaler_dfb(scaler_dfb_id);
        run_with_accumulator(scaler_dfb);
    }
}

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl"
