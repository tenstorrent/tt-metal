// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/debug/assert.h"
#include "llk_defs.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_types.hpp"

// Supplied by the compute/dataflow runtime argument API, as for TensorAccessor.
template <typename T>
T get_arg_val(int arg_idx);

/**
 * @file reduce_plan_args.hpp
 * @brief Constexpr device views over host-planned reduction arguments.
 *
 * A compute planning unit is a flat compile-time-argument suffix, not an object
 * passed to the kernel:
 *
 * @code
 * [kernel-owned prefix][call_count][call_0]...[call_(call_count - 1)]
 * @endcode
 *
 * Read call_count at the known unit offset, then address each call
 * with ReduceCallAtT. The count is only a bound for walking the calls. A kernel
 * must not derive accumulation, finalization, partial-tile handling, or any
 * other behavior from the count or call index; each ReduceCallArgs carries
 * those decisions itself.
 *
 * The matching dataflow suffix is independent and contains exactly one
 * variable-width ReduceAuxiliaryArgs per planning unit. It carries the shared
 * auxiliary CB ID and aggregate physical tile recipe. When multiple units are
 * appended, the next compute unit begins at
 * `ReduceCallAtT<first_call_offset, call_count - 1>::next_compile_time_args_offset()`. Use
 * ReduceAuxiliaryArgs::next_compile_time_args_offset() for the next dataflow
 * unit.
 */

namespace ttnn::kernel_lib {

struct ReduceRuntimeShape {
    std::uint32_t height = 0;
    std::uint32_t width = 0;
    std::uint32_t batches = 0;

    bool has_override() const { return height != 0; }
    bool matches(std::uint32_t h, std::uint32_t w, std::uint32_t b) const {
        return height == h && width == w && batches == b;
    }
};

// An empty recipe has no auxiliary binding. Preserve that absence rather than
// substituting another buffer for scaler metadata.
template <typename BindingToken>
constexpr std::uint32_t optional_auxiliary_cb(const BindingToken* binding) {
    return binding ? static_cast<std::uint32_t>(*binding) : reduce_plan_args::no_cb_id;
}

/** Constexpr view of one physical auxiliary-tile specification. */
template <std::uint32_t CTA_OFFSET, std::uint32_t TILE_INDEX, std::uint32_t TILE_COUNT>
struct ReduceAuxiliaryTileArgs {
private:
    template <reduce_plan_args::AuxiliaryTileWord FIELD>
    static constexpr std::uint32_t word() {
        return get_compile_time_arg_val(reduce_plan_args::auxiliary_tile_word_offset(CTA_OFFSET, TILE_INDEX, FIELD));
    }

    static constexpr std::uint32_t configuration = word<reduce_plan_args::AuxiliaryTileWord::Configuration>();

public:
    static_assert(TILE_INDEX < TILE_COUNT, "Reduction auxiliary tile index is outside the serialized recipe");

    static constexpr ReduceAuxiliaryTileType type = static_cast<ReduceAuxiliaryTileType>(reduce_plan_args::extract(
        configuration,
        reduce_plan_args::auxiliary_configuration::tile_type_shift,
        reduce_plan_args::auxiliary_configuration::tile_type_mask));
    static constexpr std::uint32_t num_valid_elements = reduce_plan_args::extract(
        configuration,
        reduce_plan_args::auxiliary_configuration::valid_elements_shift,
        reduce_plan_args::auxiliary_configuration::valid_elements_mask);
    static constexpr std::uint32_t value_bits = word<reduce_plan_args::AuxiliaryTileWord::ValueBits>();

    static_assert(
        type == ReduceAuxiliaryTileType::Zero || num_valid_elements > 0,
        "A non-zero auxiliary tile must contain at least one valid element");
    static_assert(
        type != ReduceAuxiliaryTileType::Zero || (num_valid_elements == 0 && value_bits == 0),
        "A zero auxiliary tile must use zero value bits and zero valid elements");
};

/**
 * @brief Constexpr view over one planning unit's aggregate auxiliary recipe.
 *
 * The descriptor includes the auxiliary CB ID. Dataflow code passes this type
 * once to prepare_reduce_auxiliary_tiles(); it does not inspect compute calls
 * or create one recipe per call.
 */
template <std::uint32_t CTA_OFFSET>
struct ReduceAuxiliaryArgs {
private:
    static constexpr std::uint32_t header = get_compile_time_arg_val(CTA_OFFSET);

public:
    static constexpr std::uint32_t cb_id = reduce_plan_args::extract(
        header, reduce_plan_args::auxiliary_header::cb_id_shift, reduce_plan_args::auxiliary_header::cb_id_mask);
    static constexpr std::uint32_t num_tiles = reduce_plan_args::extract(
        header,
        reduce_plan_args::auxiliary_header::tile_count_shift,
        reduce_plan_args::auxiliary_header::tile_count_mask);
    static constexpr std::uint32_t tiles_offset = reduce_plan_args::auxiliary_tiles_offset(CTA_OFFSET);

    template <std::uint32_t TILE_INDEX>
    using Tile = ReduceAuxiliaryTileArgs<tiles_offset, TILE_INDEX, num_tiles>;

    static constexpr std::uint32_t num_compile_time_args() {
        return reduce_plan_args::auxiliary_compile_time_arg_count(num_tiles);
    }

    static constexpr std::uint32_t next_compile_time_args_offset() { return CTA_OFFSET + num_compile_time_args(); }

    static_assert(
        num_tiles == 0 || cb_id != reduce_plan_args::no_cb_id, "A non-empty reduction auxiliary recipe requires a CB");
};

/**
 * @brief Constexpr device view of one host-planned reduction call.
 *
 * All fields are decoded from the kernel compile-time argument array, in the
 * same style as TensorAccessorArgs<CTA_OFFSET>. RTA_OFFSET locates the planner's
 * runtime argument section; serialized per-call offsets are relative to it.
 * No descriptor instance is needed: reduce<Call>() reads the runtime arguments
 * through the type when the planned call supports a tail. The descriptor is
 * self-contained: accumulation mode/index, partial mode, auxiliary slice,
 * algorithm, CB IDs, shape, input policy, and reconfiguration choices are all
 * call properties. In particular, partial_mode is authoritative; kernels must
 * not infer partial handling by examining the auxiliary tiles.
 */
template <std::uint32_t CTA_OFFSET, std::uint32_t RTA_OFFSET = 0>
struct ReduceCallArgs {
private:
    template <reduce_plan_args::CallWord FIELD>
    static constexpr std::uint32_t word() {
        return get_compile_time_arg_val(reduce_plan_args::call_word_offset(CTA_OFFSET, FIELD));
    }

    static constexpr std::uint32_t configuration = word<reduce_plan_args::CallWord::Configuration>();
    static constexpr std::uint32_t circular_buffer_ids = word<reduce_plan_args::CallWord::CircularBuffers>();
    static constexpr std::uint32_t chunk_and_auxiliary = word<reduce_plan_args::CallWord::ChunkAndAuxiliary>();

public:
    static constexpr ReducePath path = static_cast<ReducePath>(reduce_plan_args::extract(
        configuration, reduce_plan_args::config::path_shift, reduce_plan_args::config::path_mask));
    static constexpr ckernel::PoolType reduce_type = static_cast<ckernel::PoolType>(reduce_plan_args::extract(
        configuration, reduce_plan_args::config::math_shift, reduce_plan_args::config::math_mask));
    static constexpr ckernel::ReduceDim reduce_dim = static_cast<ckernel::ReduceDim>(reduce_plan_args::extract(
        configuration, reduce_plan_args::config::dimension_shift, reduce_plan_args::config::dimension_mask));
    static constexpr ReduceFp32Mode fp32_mode = static_cast<ReduceFp32Mode>(reduce_plan_args::extract(
        configuration, reduce_plan_args::config::fp32_mode_shift, reduce_plan_args::config::fp32_mode_mask));
    static constexpr compute_kernel_lib::ReduceAlgorithm algorithm =
        static_cast<compute_kernel_lib::ReduceAlgorithm>(reduce_plan_args::extract(
            configuration, reduce_plan_args::config::algorithm_shift, reduce_plan_args::config::algorithm_mask));
    static constexpr compute_kernel_lib::ReduceInputPolicy input_policy =
        static_cast<compute_kernel_lib::ReduceInputPolicy>(reduce_plan_args::extract(
            configuration, reduce_plan_args::config::input_policy_shift, reduce_plan_args::config::input_policy_mask));
    static constexpr compute_kernel_lib::AccumulateReloadMode reload_mode =
        static_cast<compute_kernel_lib::AccumulateReloadMode>(reduce_plan_args::extract(
            configuration, reduce_plan_args::config::reload_mode_shift, reduce_plan_args::config::reload_mode_mask));
    static constexpr compute_kernel_lib::ReduceDataFormatReconfigMode reconfig_mode =
        static_cast<compute_kernel_lib::ReduceDataFormatReconfigMode>(reduce_plan_args::extract(
            configuration,
            reduce_plan_args::config::reconfig_mode_shift,
            reduce_plan_args::config::reconfig_mode_mask));
    static constexpr compute_kernel_lib::ReduceWithinTile within_tile =
        static_cast<compute_kernel_lib::ReduceWithinTile>(reduce_plan_args::extract(
            configuration, reduce_plan_args::config::within_tile_shift, reduce_plan_args::config::within_tile_mask));
    static constexpr ReduceAccumulationMode accumulation_mode =
        static_cast<ReduceAccumulationMode>(reduce_plan_args::extract(
            configuration,
            reduce_plan_args::config::accumulation_mode_shift,
            reduce_plan_args::config::accumulation_mode_mask));
    static constexpr compute_kernel_lib::ReducePartialMode partial_mode =
        static_cast<compute_kernel_lib::ReducePartialMode>(reduce_plan_args::extract(
            configuration, reduce_plan_args::config::partial_mode_shift, reduce_plan_args::config::partial_mode_mask));
    static constexpr std::uint32_t input_cb_id = reduce_plan_args::extract(
        circular_buffer_ids,
        reduce_plan_args::circular_buffers::input_shift,
        reduce_plan_args::circular_buffers::id_mask);
    static constexpr std::uint32_t auxiliary_cb_id = reduce_plan_args::extract(
        circular_buffer_ids,
        reduce_plan_args::circular_buffers::auxiliary_shift,
        reduce_plan_args::circular_buffers::id_mask);
    static constexpr std::uint32_t output_cb_id = reduce_plan_args::extract(
        circular_buffer_ids,
        reduce_plan_args::circular_buffers::output_shift,
        reduce_plan_args::circular_buffers::id_mask);
    static constexpr std::uint32_t accumulator_cb_id = reduce_plan_args::extract(
        circular_buffer_ids,
        reduce_plan_args::circular_buffers::accumulator_shift,
        reduce_plan_args::circular_buffers::id_mask);
    static constexpr bool has_accumulator = accumulator_cb_id != reduce_plan_args::no_cb_id;

    static constexpr std::uint32_t accumulation_index = word<reduce_plan_args::CallWord::AccumulationIndex>();
    static constexpr std::uint32_t rows = word<reduce_plan_args::CallWord::Rows>();
    static constexpr std::uint32_t columns = word<reduce_plan_args::CallWord::Columns>();
    static constexpr std::uint32_t batches = word<reduce_plan_args::CallWord::Batches>();
    static constexpr std::uint32_t relative_tail_runtime_arg_offset =
        word<reduce_plan_args::CallWord::TailRuntimeArgOffset>();
    static_assert(
        relative_tail_runtime_arg_offset == reduce_plan_args::no_runtime_arg ||
            (RTA_OFFSET <= reduce_plan_args::no_runtime_arg - 3U &&
             relative_tail_runtime_arg_offset <= reduce_plan_args::no_runtime_arg - 3U - RTA_OFFSET),
        "Reduction runtime argument offset overflows the shape record");
    static constexpr std::uint32_t tail_runtime_arg_offset =
        relative_tail_runtime_arg_offset == reduce_plan_args::no_runtime_arg
            ? reduce_plan_args::no_runtime_arg
            : RTA_OFFSET + relative_tail_runtime_arg_offset;
    static constexpr bool has_tail_variant = reduce_plan_args::extract(
        configuration,
        reduce_plan_args::config::has_tail_variant_shift,
        reduce_plan_args::config::has_tail_variant_mask);
    static constexpr bool is_tail = reduce_plan_args::extract(
        configuration, reduce_plan_args::config::uses_tail_shape_shift, reduce_plan_args::config::uses_tail_shape_mask);
    using Tail = ReduceCallArgs<CTA_OFFSET + reduce_plan_args::call_compile_time_arg_count(), RTA_OFFSET>;
    // The descriptor carries both compile-time metadata and the location of
    // its per-core override. reduce<Call>() consumes this view internally.
    static ReduceRuntimeShape runtime_shape() {
        if constexpr (tail_runtime_arg_offset != reduce_plan_args::no_runtime_arg) {
            const auto height = get_arg_val<std::uint32_t>(tail_runtime_arg_offset);
            // A zero height selects full work; the remaining words need not exist.
            if (height == 0) {
                return {};
            }
            return {
                height,
                get_arg_val<std::uint32_t>(tail_runtime_arg_offset + 1),
                get_arg_val<std::uint32_t>(tail_runtime_arg_offset + 2)};
        } else {
            return {};
        }
    }
    static bool use_tail() {
        if constexpr (has_tail_variant) {
            const auto override = runtime_shape();
            if (!override.has_override()) {
                return false;
            }
            if constexpr (Tail::is_tail) {
                const auto local =
                    Tail::tail_runtime_arg_offset == tail_runtime_arg_offset ? override : Tail::runtime_shape();
                ASSERT(local.matches(Tail::logical_h, Tail::logical_w, Tail::batches));
            }
            return true;
        } else {
            return false;
        }
    }
    static constexpr std::uint32_t logical_h = word<reduce_plan_args::CallWord::LogicalHeight>();
    static constexpr std::uint32_t logical_w = word<reduce_plan_args::CallWord::LogicalWidth>();
    static constexpr bool has_output_mask =
        is_tail && ((reduce_dim == ckernel::ReduceDim::REDUCE_ROW && logical_h % 32 != 0) ||
                    (reduce_dim == ckernel::ReduceDim::REDUCE_COL && logical_w % 32 != 0));
    static constexpr std::uint32_t row_stride = word<reduce_plan_args::CallWord::RowStride>();
    static constexpr std::uint32_t reduce_factor = word<reduce_plan_args::CallWord::ReduceFactor>();
    static constexpr std::uint32_t output_chunk_tiles = reduce_plan_args::extract(
        chunk_and_auxiliary,
        reduce_plan_args::chunk_and_auxiliary::output_tiles_shift,
        reduce_plan_args::chunk_and_auxiliary::output_tiles_mask);
    static constexpr std::uint32_t auxiliary_tile_offset = reduce_plan_args::extract(
        chunk_and_auxiliary,
        reduce_plan_args::chunk_and_auxiliary::auxiliary_tile_offset_shift,
        reduce_plan_args::chunk_and_auxiliary::auxiliary_tile_offset_mask);
    static constexpr std::uint32_t auxiliary_tile_count = reduce_plan_args::extract(
        chunk_and_auxiliary,
        reduce_plan_args::chunk_and_auxiliary::auxiliary_tile_count_shift,
        reduce_plan_args::chunk_and_auxiliary::auxiliary_tile_count_mask);
    static constexpr std::uint32_t post_scale_bits = word<reduce_plan_args::CallWord::PostScaleBits>();

    static constexpr std::uint32_t num_compile_time_args() {
        return reduce_plan_args::call_compile_time_arg_count() * (has_tail_variant ? 2 : 1);
    }

    static constexpr std::uint32_t next_compile_time_args_offset() { return CTA_OFFSET + num_compile_time_args(); }

    static_assert(rows > 0 && columns > 0 && batches > 0, "Reduction block shape must be non-zero");
    static_assert(reduce_factor > 0, "Reduction factor must be non-zero");
    static_assert(output_chunk_tiles > 0, "Reduction output group must be non-zero");
    static_assert(
        auxiliary_tile_count == 0 || auxiliary_cb_id != reduce_plan_args::no_cb_id,
        "A non-empty reduction auxiliary slice requires a CB");
    static_assert(auxiliary_tile_count != 0 || auxiliary_tile_offset == 0, "An empty auxiliary slice has offset zero");
    static_assert(
        auxiliary_tile_count != 0 || (partial_mode == compute_kernel_lib::ReducePartialMode::None && !has_output_mask &&
                                      reload_mode != compute_kernel_lib::AccumulateReloadMode::CopySeedZeroPair),
        "Partial reductions, output masks and zero-pair reloads require auxiliary tiles");
    static_assert(
        partial_mode == compute_kernel_lib::ReducePartialMode::None ||
            partial_mode == compute_kernel_lib::ReducePartialMode::Scaler ||
            partial_mode == compute_kernel_lib::ReducePartialMode::Mask,
        "Unknown reduction partial mode");
    static_assert(
        partial_mode != compute_kernel_lib::ReducePartialMode::Scaler ||
            algorithm == compute_kernel_lib::ReduceAlgorithm::ReduceTile,
        "A partial-scaler call must use ReduceTile");
    static_assert(
        partial_mode != compute_kernel_lib::ReducePartialMode::Mask ||
            algorithm == compute_kernel_lib::ReduceAlgorithm::AccumulateViaAdd,
        "A partial-mask call must use AccumulateViaAdd");
    static_assert(
        accumulation_mode == ReduceAccumulationMode::None ||
            accumulation_mode == ReduceAccumulationMode::Intermediate ||
            accumulation_mode == ReduceAccumulationMode::Final,
        "Unknown reduction accumulation mode");
    static_assert(
        (accumulation_mode == ReduceAccumulationMode::None) != has_accumulator,
        "A call must carry an accumulator CB exactly when its accumulation mode requires one");
    static_assert(
        accumulation_mode != ReduceAccumulationMode::None || accumulation_index == 0,
        "A non-accumulating call must use accumulation index zero");
};

// Skip each call's optional tail record when locating the next call. Static
// calls retain their original fixed-size representation.
template <std::uint32_t FIRST_CALL_CTA_OFFSET, std::uint32_t CALL_INDEX>
constexpr std::uint32_t reduce_call_offset() {
    if constexpr (CALL_INDEX == 0) {
        return FIRST_CALL_CTA_OFFSET;
    } else {
        return ReduceCallArgs<
            reduce_call_offset<FIRST_CALL_CTA_OFFSET, CALL_INDEX - 1>()>::next_compile_time_args_offset();
    }
}

template <std::uint32_t FIRST_CALL_CTA_OFFSET, std::uint32_t CALL_INDEX, std::uint32_t RTA_OFFSET = 0>
using ReduceCallAtT = ReduceCallArgs<reduce_call_offset<FIRST_CALL_CTA_OFFSET, CALL_INDEX>(), RTA_OFFSET>;

// Metal 2 assigns physical buffer IDs when resolving ProgramSpec bindings.
// Such factories serialize dense, kernel-local logical IDs and bind them here
// to dfb::<name>. Only the CB namespace changes; every reduction decision,
// including the output/accumulator choice, still comes from the host record.
template <typename Call, std::uint32_t... CB_IDS>
struct BoundReduceCallArgs : Call {
private:
    static constexpr std::uint32_t cb_ids[] = {CB_IDS...};

public:
    using Tail = BoundReduceCallArgs<typename Call::Tail, CB_IDS...>;
    static_assert(Call::input_cb_id < sizeof...(CB_IDS));
    static_assert(Call::auxiliary_cb_id == reduce_plan_args::no_cb_id || Call::auxiliary_cb_id < sizeof...(CB_IDS));
    static_assert(Call::output_cb_id < sizeof...(CB_IDS));
    static_assert(!Call::has_accumulator || Call::accumulator_cb_id < sizeof...(CB_IDS));
    static_assert(((CB_IDS <= reduce_plan_args::no_cb_id) && ...));

    static constexpr std::uint32_t input_cb_id = cb_ids[Call::input_cb_id];
    static constexpr std::uint32_t auxiliary_cb_id = Call::auxiliary_cb_id == reduce_plan_args::no_cb_id
                                                         ? reduce_plan_args::no_cb_id
                                                         : cb_ids[Call::auxiliary_cb_id];
    static constexpr std::uint32_t output_cb_id = cb_ids[Call::output_cb_id];
    static constexpr std::uint32_t accumulator_cb_id =
        Call::has_accumulator ? cb_ids[Call::accumulator_cb_id] : reduce_plan_args::no_cb_id;
    static_assert(input_cb_id < reduce_plan_args::no_cb_id && output_cb_id < reduce_plan_args::no_cb_id);
    static_assert(!Call::has_accumulator || accumulator_cb_id < reduce_plan_args::no_cb_id);
    static_assert(Call::auxiliary_tile_count == 0 || auxiliary_cb_id < reduce_plan_args::no_cb_id);
    static_assert(Call::accumulation_mode != ReduceAccumulationMode::Intermediate || output_cb_id == accumulator_cb_id);
    static_assert(Call::accumulation_mode != ReduceAccumulationMode::Final || output_cb_id != accumulator_cb_id);
};

// The reader receives the same physical auxiliary recipe, bound to its
// resolved producer endpoint rather than the host's logical ID.
template <typename Auxiliary, std::uint32_t CB_ID = reduce_plan_args::no_cb_id>
struct BoundReduceAuxiliaryArgs : Auxiliary {
    static_assert(CB_ID <= reduce_plan_args::no_cb_id);
    static_assert(Auxiliary::num_tiles == 0 || CB_ID != reduce_plan_args::no_cb_id);
    static constexpr std::uint32_t cb_id = Auxiliary::num_tiles == 0 ? reduce_plan_args::no_cb_id : CB_ID;
};

}  // namespace ttnn::kernel_lib
