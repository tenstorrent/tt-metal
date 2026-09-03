// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compile_time_args.h"
#include "llk_defs.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_types.hpp"

namespace ttnn::kernel_lib {

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

    static constexpr std::uint32_t cb_id = reduce_plan_args::extract(
        configuration,
        reduce_plan_args::auxiliary_configuration::cb_id_shift,
        reduce_plan_args::auxiliary_configuration::cb_id_mask);
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

/** Constexpr view over the auxiliary tiles owned by one reduce call. */
template <std::uint32_t CTA_OFFSET, std::uint32_t TILE_COUNT>
struct ReduceAuxiliaryTilesArgs {
    static constexpr std::uint32_t num_tiles = TILE_COUNT;

    template <std::uint32_t TILE_INDEX>
    using Tile = ReduceAuxiliaryTileArgs<CTA_OFFSET, TILE_INDEX, num_tiles>;

    static constexpr std::uint32_t num_compile_time_args() {
        return num_tiles * reduce_plan_args::auxiliary_tile_word_count;
    }

    static constexpr std::uint32_t next_compile_time_args_offset() { return CTA_OFFSET + num_compile_time_args(); }
};

/**
 * @brief Constexpr device view of one host-planned reduction call.
 *
 * Instances do not own storage. Every member is decoded from the kernel compile-time argument array, in the
 * same style as TensorAccessorArgs<CTA_OFFSET>.
 */
template <std::uint32_t CTA_OFFSET>
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
    static constexpr std::uint32_t row_stride = word<reduce_plan_args::CallWord::RowStride>();
    static constexpr std::uint32_t reduce_factor = word<reduce_plan_args::CallWord::ReduceFactor>();
    static constexpr std::uint32_t reduce_axis_chunk_tiles = word<reduce_plan_args::CallWord::ReduceAxisChunkTiles>();
    static constexpr std::uint32_t output_chunk_tiles = reduce_plan_args::extract(
        chunk_and_auxiliary,
        reduce_plan_args::chunk_and_auxiliary::output_tiles_shift,
        reduce_plan_args::chunk_and_auxiliary::output_tiles_mask);
    static constexpr std::uint32_t auxiliary_tile_count = reduce_plan_args::extract(
        chunk_and_auxiliary,
        reduce_plan_args::chunk_and_auxiliary::auxiliary_tile_count_shift,
        reduce_plan_args::chunk_and_auxiliary::auxiliary_tile_count_mask);
    static constexpr std::uint32_t post_scale_bits = word<reduce_plan_args::CallWord::PostScaleBits>();

    static constexpr std::uint32_t auxiliary_tiles_offset = reduce_plan_args::call_auxiliary_tiles_offset(CTA_OFFSET);
    using AuxiliaryTiles = ReduceAuxiliaryTilesArgs<auxiliary_tiles_offset, auxiliary_tile_count>;

    static constexpr std::uint32_t num_compile_time_args() {
        return reduce_plan_args::call_compile_time_arg_count(auxiliary_tile_count);
    }

    static constexpr std::uint32_t next_compile_time_args_offset() { return CTA_OFFSET + num_compile_time_args(); }

    static_assert(rows > 0 && columns > 0 && batches > 0, "Reduction block shape must be non-zero");
    static_assert(reduce_factor > 0, "Reduction factor must be non-zero");
    static_assert(reduce_axis_chunk_tiles > 0 && output_chunk_tiles > 0, "Reduction chunk must be non-zero");
    static_assert(auxiliary_tile_count > 0, "Reduction auxiliary tile count must be non-zero");
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

// Locate a call by walking the independently sized records which begin at
// FIRST_CALL_CTA_OFFSET. This is only an addressing utility: it neither reads a
// call count nor infers call behavior from CALL_INDEX.
template <std::uint32_t FIRST_CALL_CTA_OFFSET, std::uint32_t CALL_INDEX>
struct ReduceCallAt {
private:
    using Previous = typename ReduceCallAt<FIRST_CALL_CTA_OFFSET, CALL_INDEX - 1>::type;

public:
    using type = ReduceCallArgs<Previous::next_compile_time_args_offset()>;
};

template <std::uint32_t FIRST_CALL_CTA_OFFSET>
struct ReduceCallAt<FIRST_CALL_CTA_OFFSET, 0> {
    using type = ReduceCallArgs<FIRST_CALL_CTA_OFFSET>;
};

template <std::uint32_t FIRST_CALL_CTA_OFFSET, std::uint32_t CALL_INDEX>
using ReduceCallAtT = typename ReduceCallAt<FIRST_CALL_CTA_OFFSET, CALL_INDEX>::type;

}  // namespace ttnn::kernel_lib
