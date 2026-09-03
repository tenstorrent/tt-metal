// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_host.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args_common.hpp"

namespace ttnn::kernel_lib::host {
namespace {

using compute_kernel_lib::ReduceAlgorithm;
using compute_kernel_lib::ReduceInputPolicy;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::ReduceOpDim;
using tt::tt_metal::ReduceOpMath;

std::uint32_t checked_u32(std::size_t value, const char* label) {
    TT_FATAL(value <= std::numeric_limits<std::uint32_t>::max(), "Reduce planner: {} exceeds uint32_t", label);
    return static_cast<std::uint32_t>(value);
}

std::uint32_t auxiliary_tile_count(const ReducePlan& plan) {
    TT_FATAL(!plan.auxiliary_tiles.empty(), "Reduce planner: auxiliary tile recipe must not be empty");
    return checked_u32(plan.auxiliary_tiles.size(), "auxiliary tile count");
}

std::uint32_t checked_mul_u32(std::uint32_t lhs, std::uint32_t rhs, const char* label) {
    return checked_u32(static_cast<std::uint64_t>(lhs) * rhs, label);
}

std::uint32_t div_up_u32(std::uint32_t value, std::uint32_t divisor) {
    TT_FATAL(divisor != 0, "Reduce planner: divisor must be non-zero");
    return value / divisor + (value % divisor != 0);
}

std::uint32_t shape_batch(const tt::tt_metal::Shape& shape) {
    TT_FATAL(shape.rank() >= 2, "Reduce planner: tensor rank must be at least two, got {}", shape.rank());
    std::uint64_t batches = 1;
    for (std::size_t i = 0; i + 2 < shape.rank(); ++i) {
        TT_FATAL(
            shape[i] != 0 && batches <= std::numeric_limits<std::uint64_t>::max() / shape[i],
            "Reduce planner: flattened batch count overflow");
        batches *= shape[i];
    }
    return checked_u32(batches, "flattened batch count");
}

std::uint32_t shard_page_count(const tt::tt_metal::TensorSpec& spec, std::uint32_t tile_hw) {
    TT_FATAL(tile_hw != 0, "Reduce planner: tile volume must be non-zero");
    const auto& memory = spec.memory_config();
    if (memory.shard_spec().has_value()) {
        const auto& shape = memory.shard_spec()->shape;
        const std::uint64_t volume = static_cast<std::uint64_t>(shape[0]) * shape[1];
        return checked_u32(volume / tile_hw + (volume % tile_hw != 0), "shard page count");
    }
    if (memory.nd_shard_spec().has_value()) {
        std::uint64_t volume = 1;
        for (auto dim : memory.nd_shard_spec()->shard_shape) {
            TT_FATAL(
                dim != 0 && volume <= std::numeric_limits<std::uint64_t>::max() / dim,
                "Reduce planner: ND shard volume overflow");
            volume *= dim;
        }
        return checked_u32(volume / tile_hw + (volume % tile_hw != 0), "ND shard page count");
    }
    TT_THROW("Reduce planner: sharded tensor does not carry a shard specification");
}

bool is_supported_add_type(DataType dtype) {
    return dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT8_B ||
           dtype == DataType::BFLOAT4_B;
}

bool supports_direct_input_alias(const tt::tt_metal::TensorSpec& input, ReduceOpDim dim) {
    if (!input.memory_config().is_sharded() || !input.memory_config().is_l1()) {
        return false;
    }
    const auto layout = input.memory_config().memory_layout();
    return (dim == ReduceOpDim::W && layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) ||
           (dim == ReduceOpDim::H && layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED);
}

bool supports_direct_output_alias(
    const tt::tt_metal::TensorSpec& input, const tt::tt_metal::TensorSpec& output, ReduceOpDim dim) {
    if (!output.memory_config().is_sharded() || !output.memory_config().is_l1() ||
        input.memory_config().memory_layout() != output.memory_config().memory_layout() ||
        !supports_direct_input_alias(input, dim) || !input.memory_config().shard_spec().has_value() ||
        !output.memory_config().shard_spec().has_value()) {
        return false;
    }
    const auto& input_shard = *input.memory_config().shard_spec();
    const auto& output_shard = *output.memory_config().shard_spec();
    const bool same_partition = input_shard.grid == output_shard.grid &&
                                input_shard.orientation == output_shard.orientation &&
                                (dim == ReduceOpDim::W ? input_shard.shape[0] == output_shard.shape[0]
                                                       : input_shard.shape[1] == output_shard.shape[1]);
    return same_partition;
}

std::uint32_t destination_tiles(const ReduceHardwareConfig& hardware) {
    if (hardware.dst_full_sync_en) {
        return hardware.fp32_dest_acc_en ? 8U : 16U;
    }
    return hardware.fp32_dest_acc_en ? 4U : 8U;
}

float padding_identity(ReduceOpMath math) {
    if (math == ReduceOpMath::MAX) {
        return -std::numeric_limits<float>::infinity();
    }
    if (math == ReduceOpMath::MIN) {
        return std::numeric_limits<float>::infinity();
    }
    return 0.0F;
}

std::uint32_t padding_identity_bits(tt::DataFormat format, ReduceOpMath math) {
    const float value = padding_identity(math);
    if (format == tt::DataFormat::Float32) {
        return std::bit_cast<std::uint32_t>(value);
    }
    const auto bf16 = std::bit_cast<std::uint16_t>(bfloat16::truncate(value));
    return static_cast<std::uint32_t>(bf16);
}

void add_requirement(ReducePlan& plan, ReduceCbRequirement requirement) {
    if (requirement.owns_l1()) {
        TT_FATAL(
            requirement.total_size_bytes <= std::numeric_limits<std::size_t>::max() - plan.total_owned_l1_bytes,
            "Reduce planner: reduction-owned L1 byte count overflow");
        plan.total_owned_l1_bytes += requirement.total_size_bytes;
    }
    plan.cb_requirements.push_back(requirement);
}

std::size_t available_for_input(
    const ReduceHardwareConfig& hardware,
    std::size_t fixed_bytes,
    const std::optional<std::size_t>& max_input_cb_bytes) {
    TT_FATAL(
        hardware.available_l1_bytes >= fixed_bytes,
        "Reduce planner: fixed reduction CBs require {} bytes, but only {} bytes of L1 are available",
        fixed_bytes,
        hardware.available_l1_bytes);
    const auto l1_remainder = hardware.available_l1_bytes - fixed_bytes;
    if (!max_input_cb_bytes.has_value()) {
        return l1_remainder;
    }
    return std::min(*max_input_cb_bytes, l1_remainder);
}

void choose_tiled_chunk(
    ReducePlan& plan,
    ReduceOpDim dim,
    std::uint32_t reduced_tiles,
    std::uint32_t output_tiles,
    std::uint32_t tile_bytes,
    std::size_t input_budget) {
    const std::uint64_t natural_pages = static_cast<std::uint64_t>(reduced_tiles) * output_tiles;
    const std::uint64_t natural_bytes = natural_pages * tile_bytes;
    if (natural_bytes <= input_budget) {
        plan.input_policy = ReduceInputPolicy::BulkWaitBulkPop;
        plan.chunk = {.reduce_axis_tiles = reduced_tiles, .output_tiles = output_tiles, .buffers = 1};
        return;
    }

    std::uint32_t selected_outputs = output_tiles;
    if (dim == ReduceOpDim::H) {
        selected_outputs = std::min<std::uint32_t>(selected_outputs, input_budget / (2ULL * tile_bytes));
        if (selected_outputs == 0) {
            selected_outputs = std::min<std::uint32_t>(output_tiles, input_budget / tile_bytes);
        }
    }
    TT_FATAL(
        selected_outputs > 0,
        "Reduce planner: input CB cap {} cannot hold the minimum {}-byte tile",
        input_budget,
        tile_bytes);

    const std::size_t bytes_per_axis_tile = static_cast<std::size_t>(selected_outputs) * tile_bytes;
    std::uint32_t axis_tiles = std::min<std::uint32_t>(reduced_tiles, input_budget / (2 * bytes_per_axis_tile));
    std::uint32_t buffers = 2;
    if (axis_tiles == 0) {
        axis_tiles = std::min<std::uint32_t>(reduced_tiles, input_budget / bytes_per_axis_tile);
        buffers = 1;
    }
    TT_FATAL(
        axis_tiles > 0,
        "Reduce planner: input CB cap {} cannot hold a minimum reduction chunk of {} bytes",
        input_budget,
        bytes_per_axis_tile);

    plan.input_policy = ReduceInputPolicy::ChunkedWaitChunkedPop;
    plan.chunk = {.reduce_axis_tiles = axis_tiles, .output_tiles = selected_outputs, .buffers = buffers};
}

bool add_is_legal(
    const tt::tt_metal::TensorSpec& input,
    ReduceOpMath math,
    ReduceOpDim dim,
    ReduceFp32Mode fp32_mode,
    const ReduceHardwareConfig& hardware,
    bool scalar_has_2d_partial) {
    if (hardware.arch == tt::ARCH::QUASAR || (math != ReduceOpMath::SUM && math != ReduceOpMath::AVG) ||
        !is_supported_add_type(input.data_type())) {
        return false;
    }
    if (input.data_type() == DataType::FLOAT32 && fp32_mode == ReduceFp32Mode::Accurate) {
        return false;
    }
    return dim != ReduceOpDim::HW || !scalar_has_2d_partial;
}

std::uint32_t add_threshold(ReduceOpDim dim) { return dim == ReduceOpDim::W ? 4U : 8U; }

void configure_scalar_and_aux(
    ReducePlan& plan,
    ReduceOpMath math,
    ReduceOpDim dim,
    std::uint32_t tile_h,
    std::uint32_t tile_w,
    float scalar,
    std::uint32_t logical_reduce_elements,
    std::uint32_t partial_elements,
    bool has_partial) {
    TT_FATAL(tile_h > 0 && tile_w > 0, "Reduce planner: auxiliary tile shape must be non-zero");
    plan.auxiliary_tiles.clear();
    plan.partial_mode = compute_kernel_lib::ReducePartialMode::None;

    if (plan.algorithm == ReduceAlgorithm::AccumulateViaAdd) {
        plan.reduce_factor = math == ReduceOpMath::AVG ? logical_reduce_elements : 1U;
        plan.post_scale = math == ReduceOpMath::AVG ? scalar * logical_reduce_elements : scalar;
        if (has_partial) {
            plan.auxiliary_tiles.push_back(
                {.value = 1.0F,
                 .type =
                     dim == ReduceOpDim::W ? ReduceAuxiliaryTileType::FirstRow : ReduceAuxiliaryTileType::FirstColumn,
                 .num_valid_elements = partial_elements});
            plan.partial_mode = compute_kernel_lib::ReducePartialMode::Mask;
        } else {
            plan.auxiliary_tiles.push_back(
                {.value = 1.0F, .type = ReduceAuxiliaryTileType::FirstRow, .num_valid_elements = tile_w});
        }
    } else {
        TT_FATAL(
            dim != ReduceOpDim::HW || scalar >= 0.0F,
            "Reduce planner: ReduceTile HW reduction cannot represent negative scalar {}",
            scalar);
        const float reader_scaler = dim == ReduceOpDim::HW ? std::sqrt(scalar) : scalar;
        plan.post_scale = 1.0F;
        plan.reduce_factor = 1;
        plan.auxiliary_tiles.push_back(
            {.value = reader_scaler, .type = ReduceAuxiliaryTileType::FirstRow, .num_valid_elements = tile_w});
        if (has_partial) {
            plan.auxiliary_tiles.push_back(
                {.value = reader_scaler,
                 .type = dim == ReduceOpDim::W ? ReduceAuxiliaryTileType::FirstRow
                                               : ReduceAuxiliaryTileType::FirstRowPerFaceRow,
                 .num_valid_elements = partial_elements});
            plan.partial_mode = compute_kernel_lib::ReducePartialMode::Scaler;
        }
    }
    plan.partial_reduce_axis_elements = has_partial ? partial_elements : 0U;
}

ReducePlan make_tiled_plan(
    const tt::tt_metal::TensorSpec& input,
    const tt::tt_metal::TensorSpec& output,
    ReduceOpMath math,
    ReduceOpDim dim,
    float scalar,
    ReduceFp32Mode fp32_mode,
    const ReduceHardwareConfig& hardware,
    std::optional<std::size_t> max_input_cb_bytes,
    std::optional<ReduceAlgorithm> forced_algorithm) {
    ReducePlan plan;
    plan.path = ReducePath::Tiled;

    const auto& logical = input.logical_shape();
    const auto& padded = input.padded_shape();
    const auto tile = input.tile();
    const std::uint32_t tile_h = tile.get_height();
    const std::uint32_t tile_w = tile.get_width();
    const std::uint32_t logical_h = checked_u32(logical[logical.rank() - 2], "logical height");
    const std::uint32_t logical_w = checked_u32(logical[logical.rank() - 1], "logical width");
    const std::uint32_t padded_h = checked_u32(padded[padded.rank() - 2], "padded height");
    const std::uint32_t padded_w = checked_u32(padded[padded.rank() - 1], "padded width");
    TT_FATAL(
        logical_h != 0 && logical_w != 0 && padded_h != 0 && padded_w != 0,
        "Reduce planner: tensor height and width must be non-zero");
    plan.Ht = div_up_u32(padded_h, tile_h);
    plan.Wt = div_up_u32(padded_w, tile_w);
    plan.batches = shape_batch(padded);

    const std::uint32_t reduced_tiles =
        dim == ReduceOpDim::W ? plan.Wt
                              : (dim == ReduceOpDim::H ? plan.Ht : checked_mul_u32(plan.Ht, plan.Wt, "HW tile count"));
    const std::uint32_t logical_reduce_elements =
        dim == ReduceOpDim::W
            ? logical_w
            : (dim == ReduceOpDim::H ? logical_h
                                     : checked_mul_u32(logical_h, logical_w, "logical HW reduction volume"));
    const std::uint32_t partial_elements =
        dim == ReduceOpDim::W ? logical_w % tile_w : (dim == ReduceOpDim::H ? logical_h % tile_h : 0U);
    const bool has_axis_partial = partial_elements != 0 && (math == ReduceOpMath::SUM || math == ReduceOpMath::AVG);
    const bool scalar_has_2d_partial = dim == ReduceOpDim::HW && ((logical_h % tile_h) || (logical_w % tile_w));

    const auto automatic_algorithm = add_is_legal(input, math, dim, fp32_mode, hardware, scalar_has_2d_partial) &&
                                             reduced_tiles >= add_threshold(dim)
                                         ? ReduceAlgorithm::AccumulateViaAdd
                                         : ReduceAlgorithm::ReduceTile;
    TT_FATAL(
        !forced_algorithm.has_value() || *forced_algorithm != ReduceAlgorithm::AccumulateViaAdd ||
            automatic_algorithm == ReduceAlgorithm::AccumulateViaAdd,
        "Reduce planner: AccumulateViaAdd was forced for an unsupported tiled reduction");
    plan.algorithm = forced_algorithm.value_or(automatic_algorithm);
    configure_scalar_and_aux(
        plan, math, dim, tile_h, tile_w, scalar, logical_reduce_elements, partial_elements, has_axis_partial);

    const auto input_format = tt::tt_metal::datatype_to_dataformat_converter(input.data_type());
    const auto output_format = tt::tt_metal::datatype_to_dataformat_converter(output.data_type());
    const auto aux_format =
        input_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const std::uint32_t input_tile_bytes = tt::tt_metal::tile_size(input.data_type());
    const std::uint32_t output_tile_bytes = tt::tt_metal::tile_size(output.data_type());
    const std::uint32_t aux_tile_bytes = tt::tile_size(aux_format);

    const bool alias_input = max_input_cb_bytes.has_value() && *max_input_cb_bytes == 0;
    if (alias_input) {
        TT_FATAL(
            supports_direct_input_alias(input, dim),
            "Reduce planner: a zero input-CB cap requires a directly consumable L1 shard "
            "(HEIGHT_SHARDED for W or WIDTH_SHARDED for H)");
        plan.input_policy = ReduceInputPolicy::NoWaitNoPop;
        plan.chunk = {
            .reduce_axis_tiles = reduced_tiles,
            .output_tiles = dim == ReduceOpDim::H ? std::min(plan.Wt, destination_tiles(hardware)) : 1U,
            .buffers = 0};
        const auto pages = shard_page_count(input, tile.get_tile_hw());
        add_requirement(
            plan,
            {.role = ReduceCbRole::Input,
             .data_format = input_format,
             .page_size = input_tile_bytes,
             .page_count = pages,
             .total_size_bytes = static_cast<std::size_t>(pages) * input_tile_bytes,
             .alias = ReduceCbAlias::InputTensor});
    }

    const bool alias_output = supports_direct_output_alias(input, output, dim);
    const std::uint32_t output_pages = alias_output ? shard_page_count(output, output.tile().get_tile_hw()) : 2U;
    const std::size_t output_bytes = static_cast<std::size_t>(output_pages) * output_tile_bytes;
    std::size_t aux_bytes = static_cast<std::size_t>(auxiliary_tile_count(plan)) * aux_tile_bytes;
    const std::size_t fixed_owned_bytes = aux_bytes + (alias_output ? 0U : output_bytes);

    if (!alias_input) {
        TT_FATAL(
            !max_input_cb_bytes.has_value() || *max_input_cb_bytes > 0,
            "Reduce planner: internal error resolving input CB cap");
        const auto input_budget = available_for_input(hardware, fixed_owned_bytes, max_input_cb_bytes);
        std::uint32_t output_group = 1;
        if (dim == ReduceOpDim::H) {
            output_group = std::min(plan.Wt, destination_tiles(hardware));
            const bool uses_sfpu_work_tile =
                input.data_type() == DataType::INT32 ||
                (input.data_type() == DataType::FLOAT32 && fp32_mode == ReduceFp32Mode::Accurate);
            if (uses_sfpu_work_tile) {
                TT_FATAL(output_group > 1, "Reduce planner: H reduction has no DEST output slots");
                --output_group;
            }
        }
        choose_tiled_chunk(plan, dim, reduced_tiles, output_group, input_tile_bytes, input_budget);

        // ROW/SCALAR pairwise streams use even chunks. COL carries a row-major output group and retains one
        // DEST accumulator per column, so any axis chunk of at least two rows can make additive progress.
        if (plan.algorithm == ReduceAlgorithm::AccumulateViaAdd &&
            plan.input_policy == ReduceInputPolicy::ChunkedWaitChunkedPop) {
            if (dim != ReduceOpDim::H && (plan.chunk.reduce_axis_tiles & 1U)) {
                --plan.chunk.reduce_axis_tiles;
            }
            const std::uint32_t minimum_axis_chunk = dim == ReduceOpDim::H ? 2U : 1U;
            if (plan.chunk.reduce_axis_tiles < minimum_axis_chunk) {
                plan.algorithm = ReduceAlgorithm::ReduceTile;
                plan.chunk.reduce_axis_tiles = 1;
                configure_scalar_and_aux(
                    plan,
                    math,
                    dim,
                    tile_h,
                    tile_w,
                    scalar,
                    logical_reduce_elements,
                    partial_elements,
                    has_axis_partial);
            }
        }

        const std::uint32_t input_pages = checked_mul_u32(
            checked_mul_u32(plan.chunk.reduce_axis_tiles, plan.chunk.output_tiles, "input chunk tile count"),
            plan.chunk.buffers,
            "input CB page count");
        add_requirement(
            plan,
            {.role = ReduceCbRole::Input,
             .data_format = input_format,
             .page_size = input_tile_bytes,
             .page_count = input_pages,
             .total_size_bytes = static_cast<std::size_t>(input_pages) * input_tile_bytes});

        // A too-small additive chunk falls back to ReduceTile, which can change a partial reduction from a
        // one-tile mask to a two-tile full/partial scaler pair.
        aux_bytes = static_cast<std::size_t>(auxiliary_tile_count(plan)) * aux_tile_bytes;
    }

    add_requirement(
        plan,
        {.role = ReduceCbRole::Auxiliary,
         .data_format = aux_format,
         .page_size = aux_tile_bytes,
         .page_count = auxiliary_tile_count(plan),
         .total_size_bytes = aux_bytes});
    add_requirement(
        plan,
        {.role = ReduceCbRole::Output,
         .data_format = output_format,
         .page_size = output_tile_bytes,
         .page_count = output_pages,
         .total_size_bytes = output_bytes,
         .alias = alias_output ? ReduceCbAlias::OutputTensor : ReduceCbAlias::None});

    TT_FATAL(
        plan.total_owned_l1_bytes <= hardware.available_l1_bytes,
        "Reduce planner: reduction CBs require {} bytes, but only {} bytes of L1 are available",
        plan.total_owned_l1_bytes,
        hardware.available_l1_bytes);
    return plan;
}

std::size_t rm_input_bytes(
    ReduceOpDim dim,
    std::uint32_t axis_chunk,
    std::uint32_t staging_buffers,
    std::uint32_t tile_h,
    std::uint32_t tile_w,
    std::uint32_t src_datum_bytes,
    std::uint32_t input_tile_bytes) {
    const std::uint32_t wt = dim == ReduceOpDim::W ? axis_chunk : 1U;
    const std::uint32_t ht = dim == ReduceOpDim::H ? axis_chunk : 1U;
    const std::size_t staging = static_cast<std::size_t>(staging_buffers) * tile_h * wt * tile_w * src_datum_bytes;
    const std::size_t scratch = static_cast<std::size_t>(std::max(2U, wt * ht)) * input_tile_bytes;
    return staging + scratch;
}

ReducePlan make_row_major_plan(
    const tt::tt_metal::TensorSpec& input,
    const tt::tt_metal::TensorSpec& output,
    ReduceOpMath math,
    ReduceOpDim dim,
    float scalar,
    ReduceFp32Mode fp32_mode,
    const ReduceHardwareConfig& hardware,
    std::optional<std::size_t> max_input_cb_bytes,
    std::optional<ReduceAlgorithm> forced_algorithm) {
    TT_FATAL(dim == ReduceOpDim::W || dim == ReduceOpDim::H, "Reduce planner: dense row-major supports W or H only");
    TT_FATAL(
        math == ReduceOpMath::SUM,
        "Reduce planner: dense row-major requires SUM kernel math; lower mean to SUM plus its scalar first");
    TT_FATAL(
        input.data_type() == DataType::BFLOAT16 || input.data_type() == DataType::FLOAT32,
        "Reduce planner: dense row-major supports BFLOAT16 or FLOAT32 input only");
    TT_FATAL(
        !max_input_cb_bytes.has_value() || *max_input_cb_bytes != 0,
        "Reduce planner: dense row-major input cannot use the zero-cap L1-sharded alias mode");
    TT_FATAL(
        input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "Reduce planner: dense row-major input must be interleaved");
    TT_FATAL(
        output.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "Reduce planner: dense row-major output must be interleaved");
    TT_FATAL(
        dim != ReduceOpDim::W || output.layout() == Layout::ROW_MAJOR,
        "Reduce planner: dense row-major W reduction requires row-major output");

    ReducePlan plan;
    plan.path = ReducePath::DenseRowMajor;
    const auto& logical = input.logical_shape();
    const auto& padded = input.padded_shape();
    const auto tile = input.tile();
    const std::uint32_t tile_h = tile.get_height();
    const std::uint32_t tile_w = tile.get_width();
    const std::uint32_t logical_h = checked_u32(logical[logical.rank() - 2], "logical height");
    const std::uint32_t logical_w = checked_u32(logical[logical.rank() - 1], "logical width");
    const std::uint32_t padded_w = checked_u32(padded[padded.rank() - 1], "padded width");
    TT_FATAL(
        logical_h != 0 && logical_w != 0 && padded_w != 0, "Reduce planner: tensor height and width must be non-zero");
    plan.Ht = div_up_u32(logical_h, tile_h);
    plan.Wt = div_up_u32(padded_w, tile_w);
    plan.batches = shape_batch(logical);

    const std::uint32_t reduced_tiles = dim == ReduceOpDim::W ? plan.Wt : plan.Ht;
    const std::uint32_t logical_reduce_elements = dim == ReduceOpDim::W ? logical_w : logical_h;
    const std::uint32_t partial_elements = dim == ReduceOpDim::W ? logical_w % tile_w : logical_h % tile_h;
    const bool has_partial = partial_elements != 0 && (math == ReduceOpMath::SUM || math == ReduceOpMath::AVG);
    const auto automatic_algorithm =
        add_is_legal(input, math, dim, fp32_mode, hardware, false) && reduced_tiles >= add_threshold(dim)
            ? ReduceAlgorithm::AccumulateViaAdd
            : ReduceAlgorithm::ReduceTile;
    TT_FATAL(
        !forced_algorithm.has_value() || *forced_algorithm != ReduceAlgorithm::AccumulateViaAdd ||
            automatic_algorithm == ReduceAlgorithm::AccumulateViaAdd,
        "Reduce planner: AccumulateViaAdd was forced for an unsupported dense row-major reduction");
    plan.algorithm = forced_algorithm.value_or(automatic_algorithm);
    // Dense input is explicitly identity-padded by the reader before tilization, so it does not
    // need a second scaler or mask tile even when the logical edge is partial.
    configure_scalar_and_aux(plan, math, dim, tile_h, tile_w, scalar, logical_reduce_elements, partial_elements, false);
    plan.partial_reduce_axis_elements = has_partial ? partial_elements : 0U;

    const auto input_format = tt::tt_metal::datatype_to_dataformat_converter(input.data_type());
    const auto output_format = tt::tt_metal::datatype_to_dataformat_converter(output.data_type());
    const auto aux_format =
        input_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const std::uint32_t input_tile_bytes = tt::tt_metal::tile_size(input.data_type());
    const std::uint32_t output_tile_bytes = tt::tt_metal::tile_size(output.data_type());
    const std::uint32_t aux_tile_bytes = tt::tile_size(aux_format);
    const std::uint32_t src_datum_bytes = tt::datum_size(input_format);
    const std::uint32_t dst_datum_bytes = tt::datum_size(output_format);

    // Output, auxiliary, clear-template, and one accumulator tile are fixed. W uses one H tile
    // per staged work unit and H uses one output column, so one accumulator page is sufficient.
    const std::size_t fixed_bytes =
        static_cast<std::size_t>(2) * output_tile_bytes + aux_tile_bytes + input_tile_bytes + output_tile_bytes;
    const auto input_budget = available_for_input(hardware, fixed_bytes, max_input_cb_bytes);

    std::uint32_t axis_chunk = 0;
    std::uint32_t staging_buffers = 2;
    const auto natural_bytes = rm_input_bytes(dim, reduced_tiles, 1, tile_h, tile_w, src_datum_bytes, input_tile_bytes);
    if (natural_bytes <= input_budget) {
        axis_chunk = reduced_tiles;
        staging_buffers = 1;
    } else {
        std::uint64_t low = 1;
        std::uint64_t high = reduced_tiles;
        while (low <= high) {
            const auto candidate = static_cast<std::uint32_t>(low + (high - low) / 2);
            if (rm_input_bytes(dim, candidate, 2, tile_h, tile_w, src_datum_bytes, input_tile_bytes) <= input_budget) {
                axis_chunk = candidate;
                low = static_cast<std::uint64_t>(candidate) + 1;
            } else {
                high = static_cast<std::uint64_t>(candidate) - 1;
            }
        }
        if (axis_chunk == 0 &&
            rm_input_bytes(dim, 1, 1, tile_h, tile_w, src_datum_bytes, input_tile_bytes) <= input_budget) {
            axis_chunk = 1;
            staging_buffers = 1;
        }
    }
    TT_FATAL(
        axis_chunk > 0,
        "Reduce planner: input CB cap {} cannot hold the minimum dense row-major staging and tiled scratch",
        input_budget);

    const std::uint32_t wt_chunk = dim == ReduceOpDim::W ? axis_chunk : 1U;
    const std::uint32_t ht_chunk = dim == ReduceOpDim::H ? axis_chunk : 1U;
    const std::uint32_t staging_page = checked_mul_u32(
        checked_mul_u32(wt_chunk, tile_w, "dense row-major staging width"),
        src_datum_bytes,
        "dense row-major staging page size");
    plan.chunk = {.reduce_axis_tiles = axis_chunk, .output_tiles = 1, .buffers = staging_buffers};
    plan.input_policy =
        reduced_tiles == axis_chunk ? ReduceInputPolicy::BulkWaitBulkPop : ReduceInputPolicy::ChunkedWaitChunkedPop;
    const bool additive_chunks = plan.algorithm == ReduceAlgorithm::AccumulateViaAdd && reduced_tiles != axis_chunk;
    plan.reload_mode = additive_chunks ? compute_kernel_lib::AccumulateReloadMode::CopySeedZeroPair
                                       : compute_kernel_lib::AccumulateReloadMode::CopySeedPairs;
    if (additive_chunks) {
        // Dense input is identity-padded before tilization, so a logical partial does not consume an aux mask.
        // The zero tile is therefore available for odd staged chunks regardless of logical alignment.
        plan.auxiliary_tiles = {{.value = 0.0F, .type = ReduceAuxiliaryTileType::Zero, .num_valid_elements = 0}};
    }

    plan.row_major = DenseRowMajorPlan{
        .H_logical = logical_h,
        .W_logical = logical_w,
        .Ht_rm = plan.Ht,
        .Wt = plan.Wt,
        .rm_rows_per_tile = tile_h,
        .wt_tiles_per_chunk = wt_chunk,
        .ht_tiles_per_chunk = ht_chunk,
        .chunk_row_bytes = staging_page,
        .rm_staging_page_size = staging_page,
        .padding_identity_bits = padding_identity_bits(input_format, math),
        .src_datum_size = src_datum_bytes,
        .dst_datum_size = dst_datum_bytes,
        .staging_buffers = staging_buffers,
    };

    const std::uint32_t staging_pages = checked_mul_u32(staging_buffers, tile_h, "row-major staging page count");
    const std::uint32_t scratch_pages =
        std::max(2U, checked_mul_u32(wt_chunk, ht_chunk, "row-major tiled scratch page count"));
    add_requirement(
        plan,
        {.role = ReduceCbRole::RowMajorStaging,
         .data_format = input_format,
         .page_size = staging_page,
         .page_count = staging_pages,
         .total_size_bytes = static_cast<std::size_t>(staging_pages) * staging_page});
    add_requirement(
        plan,
        {.role = ReduceCbRole::TiledScratch,
         .data_format = input_format,
         .page_size = input_tile_bytes,
         .page_count = scratch_pages,
         .total_size_bytes = static_cast<std::size_t>(scratch_pages) * input_tile_bytes});
    add_requirement(
        plan,
        {.role = ReduceCbRole::PaddingIdentity,
         .data_format = input_format,
         .page_size = input_tile_bytes,
         .page_count = 1,
         .total_size_bytes = input_tile_bytes});
    add_requirement(
        plan,
        {.role = ReduceCbRole::Accumulator,
         .data_format = output_format,
         .page_size = output_tile_bytes,
         .page_count = 1,
         .total_size_bytes = output_tile_bytes});
    add_requirement(
        plan,
        {.role = ReduceCbRole::Auxiliary,
         .data_format = aux_format,
         .page_size = aux_tile_bytes,
         .page_count = auxiliary_tile_count(plan),
         .total_size_bytes = static_cast<std::size_t>(auxiliary_tile_count(plan)) * aux_tile_bytes});
    add_requirement(
        plan,
        {.role = ReduceCbRole::Output,
         .data_format = output_format,
         .page_size = output_tile_bytes,
         .page_count = 2,
         .total_size_bytes = static_cast<std::size_t>(2) * output_tile_bytes});

    TT_FATAL(
        plan.total_owned_l1_bytes <= hardware.available_l1_bytes,
        "Reduce planner: dense row-major CBs require {} bytes, but only {} bytes of L1 are available",
        plan.total_owned_l1_bytes,
        hardware.available_l1_bytes);
    return plan;
}

}  // namespace

const ReduceCbRequirement* ReducePlan::find_cb(ReduceCbRole role) const {
    const auto it = std::find_if(cb_requirements.begin(), cb_requirements.end(), [role](const auto& requirement) {
        return requirement.role == role;
    });
    return it == cb_requirements.end() ? nullptr : &*it;
}

namespace {

ReducePlan make_reduce_plan_impl(
    const tt::tt_metal::TensorSpec& input_spec,
    const tt::tt_metal::TensorSpec& output_spec,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scalar,
    ReduceFp32Mode fp32_mode,
    const ReduceHardwareConfig& hardware,
    std::optional<std::size_t> max_input_cb_bytes,
    std::optional<ReduceAlgorithm> forced_algorithm) {
    TT_FATAL(
        reduce_math != tt::tt_metal::ReduceOpMath::STD && reduce_math != tt::tt_metal::ReduceOpMath::VAR,
        "Reduce planner: Welford STD/VAR reductions are outside this planner");
    TT_FATAL(hardware.arch != tt::ARCH::Invalid, "Reduce planner: hardware architecture must be specified");
    TT_FATAL(hardware.available_l1_bytes > 0, "Reduce planner: available L1 size must be non-zero");
    TT_FATAL(
        input_spec.logical_shape().rank() >= 2 && output_spec.logical_shape().rank() >= 2,
        "Reduce planner: input and output rank must be at least two");
    TT_FATAL(
        input_spec.layout() == Layout::TILE || input_spec.layout() == Layout::ROW_MAJOR,
        "Reduce planner: unsupported input layout {}",
        input_spec.layout());
    TT_FATAL(
        output_spec.layout() == Layout::TILE || output_spec.layout() == Layout::ROW_MAJOR,
        "Reduce planner: unsupported output layout {}",
        output_spec.layout());
    TT_FATAL(
        input_spec.layout() != Layout::TILE || output_spec.layout() == Layout::TILE,
        "Reduce planner: tiled input requires tiled output");
    TT_FATAL(
        input_spec.data_type() != DataType::FLOAT32 || fp32_mode != ReduceFp32Mode::Accurate ||
            (hardware.arch != tt::ARCH::QUASAR && hardware.fp32_dest_acc_en),
        "Reduce planner: accurate FLOAT32 reduction requires fp32 DEST accumulation on a non-Quasar device");

    ReducePlan plan;
    if (input_spec.layout() == Layout::ROW_MAJOR) {
        plan = make_row_major_plan(
            input_spec,
            output_spec,
            reduce_math,
            reduce_dim,
            scalar,
            fp32_mode,
            hardware,
            max_input_cb_bytes,
            forced_algorithm);
    } else {
        plan = make_tiled_plan(
            input_spec,
            output_spec,
            reduce_math,
            reduce_dim,
            scalar,
            fp32_mode,
            hardware,
            max_input_cb_bytes,
            forced_algorithm);
    }
    plan.reduce_math = reduce_math;
    plan.reduce_dim = reduce_dim;
    plan.fp32_mode = fp32_mode;
    return plan;
}

}  // namespace

ReducePlan make_reduce_plan(
    const tt::tt_metal::TensorSpec& input_spec,
    const tt::tt_metal::TensorSpec& output_spec,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scalar,
    ReduceFp32Mode fp32_mode,
    const ReduceHardwareConfig& hardware,
    std::optional<std::size_t> max_input_cb_bytes) {
    return make_reduce_plan_impl(
        input_spec,
        output_spec,
        reduce_math,
        reduce_dim,
        scalar,
        fp32_mode,
        hardware,
        max_input_cb_bytes,
        std::nullopt);
}

namespace {

std::uint32_t reduced_axis_tiles(const ReducePlan& plan, ReduceOpDim dim) {
    if (dim == ReduceOpDim::W) {
        return plan.Wt;
    }
    if (dim == ReduceOpDim::H) {
        return plan.Ht;
    }
    return checked_mul_u32(plan.Ht, plan.Wt, "HW tile count");
}

bool zero_pair_avoids_an_odd_fold(const ReducePlan& plan, ReduceOpDim dim, bool is_first_call) {
    if (plan.algorithm != ReduceAlgorithm::AccumulateViaAdd) {
        return false;
    }

    const std::uint32_t axis_tiles = reduced_axis_tiles(plan, dim);
    const std::uint32_t full_tiles = axis_tiles - (plan.partial_reduce_axis_elements != 0 ? 1U : 0U);
    if (full_tiles == 0) {
        return false;
    }

    const bool grouped_col = dim == ReduceOpDim::H && (plan.input_policy == ReduceInputPolicy::BulkWaitBulkPop ||
                                                       plan.input_policy == ReduceInputPolicy::ChunkedWaitChunkedPop);
    if (!grouped_col) {
        // The first call can copy-seed an odd tile. A later call has already seeded DEST from the accumulator,
        // so an odd full-tile count otherwise needs one DEST-reuse fold.
        return !is_first_call && (full_tiles & 1U) != 0;
    }

    // Grouped COL restarts pairing at every row-chunk boundary while retaining DEST. The first odd group can
    // copy-seed only on the first call; every odd group after DEST is seeded benefits from a zero pair.
    const std::uint32_t row_chunk =
        plan.input_policy == ReduceInputPolicy::ChunkedWaitChunkedPop ? plan.chunk.reduce_axis_tiles : axis_tiles;
    bool dst_seeded = !is_first_call;
    for (std::uint32_t processed = 0; processed < full_tiles;) {
        const std::uint32_t rows = std::min(row_chunk, full_tiles - processed);
        if (dst_seeded && (rows & 1U) != 0) {
            return true;
        }
        dst_seeded = true;
        processed += rows;
    }
    return false;
}

bool try_enable_zero_pair(ReducePlan& plan, const ReduceHardwareConfig& hardware) {
    auto auxiliary = std::find_if(plan.cb_requirements.begin(), plan.cb_requirements.end(), [](const auto& req) {
        return req.role == ReduceCbRole::Auxiliary;
    });
    TT_FATAL(auxiliary != plan.cb_requirements.end(), "Reduce planner: plan is missing its auxiliary CB");

    TT_FATAL(
        plan.algorithm == ReduceAlgorithm::AccumulateViaAdd,
        "Reduce planner: a zero-pair recipe requires AccumulateViaAdd");
    const bool already_has_zero =
        std::any_of(plan.auxiliary_tiles.begin(), plan.auxiliary_tiles.end(), [](const auto& tile) {
            return tile.type == ReduceAuxiliaryTileType::Zero;
        });
    const bool has_partial = plan.partial_mode != compute_kernel_lib::ReducePartialMode::None;
    const std::uint32_t extra_tiles = has_partial && !already_has_zero ? 1U : 0U;

    const std::size_t extra_bytes = static_cast<std::size_t>(extra_tiles) * auxiliary->page_size;
    if (extra_bytes > hardware.available_l1_bytes - plan.total_owned_l1_bytes) {
        return false;  // CopySeedPairs is the correct no-extra-L1 fallback.
    }

    plan.reload_mode = compute_kernel_lib::AccumulateReloadMode::CopySeedZeroPair;
    if (!already_has_zero) {
        const ReduceAuxiliaryTileSpec zero{
            .value = 0.0F, .type = ReduceAuxiliaryTileType::Zero, .num_valid_elements = 0};
        if (has_partial) {
            plan.auxiliary_tiles.push_back(zero);
        } else {
            // The ordinary aligned auxiliary scalar is unused by AccumulateViaAdd,
            // so the zero tile replaces it without increasing L1 usage.
            plan.auxiliary_tiles = {zero};
        }
    }
    auxiliary->page_count = auxiliary_tile_count(plan);
    auxiliary->total_size_bytes += extra_bytes;
    plan.total_owned_l1_bytes += extra_bytes;
    return true;
}

}  // namespace

ReduceSequencePlan make_reduce_sequence_plan(
    const std::vector<ReduceCbConfig>& reductions,
    const ReduceSequenceCbIds& cb_ids,
    const ReduceHardwareConfig& hardware) {
    TT_FATAL(!reductions.empty(), "Reduce sequence planner: at least one input CB is required");
    TT_FATAL(
        reductions.size() <= std::numeric_limits<std::uint32_t>::max(),
        "Reduce sequence planner: call count exceeds uint32_t");

    const bool accumulates = reductions.size() > 1;
    TT_FATAL(
        cb_ids.auxiliary_cb_id != cb_ids.output_cb_id,
        "Reduce sequence planner: auxiliary and final output CB IDs must differ");
    if (accumulates) {
        TT_FATAL(
            cb_ids.accumulator_cb_id != cb_ids.auxiliary_cb_id && cb_ids.accumulator_cb_id != cb_ids.output_cb_id,
            "Reduce sequence planner: accumulator, auxiliary, and final output CB IDs must be distinct");
    }

    const auto& first_config = reductions.front().second;
    std::vector<ReducePlan> plans;
    plans.reserve(reductions.size());
    std::vector<std::uint32_t> input_cb_ids;
    input_cb_ids.reserve(reductions.size());

    for (const auto& [input_cb_id, config] : reductions) {
        TT_FATAL(
            input_cb_id != cb_ids.auxiliary_cb_id && input_cb_id != cb_ids.output_cb_id &&
                (!accumulates || input_cb_id != cb_ids.accumulator_cb_id),
            "Reduce sequence planner: input CB {} collides with a shared reduction CB",
            input_cb_id);
        TT_FATAL(
            std::find(input_cb_ids.begin(), input_cb_ids.end(), input_cb_id) == input_cb_ids.end(),
            "Reduce sequence planner: input CB {} appears more than once",
            input_cb_id);
        input_cb_ids.push_back(input_cb_id);

        if (accumulates) {
            TT_FATAL(
                config.output_spec == first_config.output_spec,
                "Reduce sequence planner: accumulated calls must have identical output tensor specs");
            TT_FATAL(
                config.reduce_math == first_config.reduce_math && config.reduce_dim == first_config.reduce_dim &&
                    config.fp32_mode == first_config.fp32_mode,
                "Reduce sequence planner: accumulated calls must use the same math, dimension, and fp32 mode");
            TT_FATAL(
                config.scalar == first_config.scalar || config.reduce_math == ReduceOpMath::AVG,
                "Reduce sequence planner: non-AVG accumulated calls must use the same scalar");
        }

        plans.push_back(make_reduce_plan_impl(
            config.input_spec,
            config.output_spec,
            config.reduce_math,
            config.reduce_dim,
            config.scalar,
            config.fp32_mode,
            hardware,
            config.max_input_cb_bytes,
            std::nullopt));
    }

    // A raw AccumulateViaAdd partial and a finalized ReduceTile partial are different accumulator formats.
    // If automatic planning chose a mixture, replan every input on the universally composable ReduceTile path.
    const auto sequence_algorithm = plans.front().algorithm;
    const bool mixed_algorithms = std::any_of(
        plans.begin(), plans.end(), [&](const ReducePlan& plan) { return plan.algorithm != sequence_algorithm; });
    if (mixed_algorithms) {
        plans.clear();
        for (const auto& [input_cb_id, config] : reductions) {
            (void)input_cb_id;
            plans.push_back(make_reduce_plan_impl(
                config.input_spec,
                config.output_spec,
                config.reduce_math,
                config.reduce_dim,
                config.scalar,
                config.fp32_mode,
                hardware,
                config.max_input_cb_bytes,
                ReduceAlgorithm::ReduceTile));
        }
    }

    if (accumulates && plans.front().algorithm == ReduceAlgorithm::AccumulateViaAdd) {
        for (std::size_t i = 0; i < plans.size(); ++i) {
            if (zero_pair_avoids_an_odd_fold(plans[i], reductions[i].second.reduce_dim, i == 0)) {
                try_enable_zero_pair(plans[i], hardware);
            }
        }
    }

    const auto output_count = [](const ReducePlan& plan, ReduceOpDim dim) -> std::uint64_t {
        if (dim == ReduceOpDim::W) {
            return static_cast<std::uint64_t>(plan.Ht) * plan.batches;
        }
        if (dim == ReduceOpDim::H) {
            return static_cast<std::uint64_t>(plan.Wt) * plan.batches;
        }
        return plan.batches;
    };
    const auto expected_outputs = output_count(plans.front(), first_config.reduce_dim);
    for (std::size_t i = 1; i < plans.size(); ++i) {
        TT_FATAL(
            output_count(plans[i], reductions[i].second.reduce_dim) == expected_outputs,
            "Reduce sequence planner: every accumulated call must produce the same number of output tiles");
    }

    // AVG is normalized once over the union of all reduced tensors. Config.scalar may include an additional
    // caller multiplier; scalar * local_element_count must agree across calls, then the planner transfers that
    // multiplier to the grand-total normalization.
    if (accumulates && first_config.reduce_math == ReduceOpMath::AVG) {
        std::uint64_t grand_reduce_elements = 0;
        float common_post_multiplier = 0.0F;
        for (std::size_t i = 0; i < reductions.size(); ++i) {
            const auto& config = reductions[i].second;
            const auto& shape = config.input_spec.logical_shape();
            const std::uint64_t height = shape[shape.rank() - 2];
            const std::uint64_t width = shape[shape.rank() - 1];
            const std::uint64_t local_reduce_elements =
                config.reduce_dim == ReduceOpDim::W ? width
                                                    : (config.reduce_dim == ReduceOpDim::H ? height : height * width);
            TT_FATAL(
                local_reduce_elements > 0 && local_reduce_elements <= std::numeric_limits<std::uint32_t>::max() &&
                    grand_reduce_elements <= std::numeric_limits<std::uint32_t>::max() - local_reduce_elements,
                "Reduce sequence planner: grand AVG reduction factor exceeds uint32_t");
            grand_reduce_elements += local_reduce_elements;

            const float post_multiplier = config.scalar * static_cast<float>(local_reduce_elements);
            if (i == 0) {
                common_post_multiplier = post_multiplier;
            } else {
                const float tolerance =
                    1.0e-5F * std::max({1.0F, std::abs(common_post_multiplier), std::abs(post_multiplier)});
                TT_FATAL(
                    std::abs(post_multiplier - common_post_multiplier) <= tolerance,
                    "Reduce sequence planner: AVG calls must describe the same post-reduction multiplier");
            }
        }

        const auto grand_factor = static_cast<std::uint32_t>(grand_reduce_elements);
        const float grand_scalar = common_post_multiplier / static_cast<float>(grand_factor);
        for (auto& plan : plans) {
            if (plan.algorithm == ReduceAlgorithm::AccumulateViaAdd) {
                plan.reduce_factor = grand_factor;
                plan.post_scale = common_post_multiplier;
            } else {
                const float reader_scaler =
                    first_config.reduce_dim == ReduceOpDim::HW ? std::sqrt(grand_scalar) : grand_scalar;
                plan.reduce_factor = 1;
                plan.post_scale = 1.0F;
                for (auto& tile : plan.auxiliary_tiles) {
                    tile.value = reader_scaler;
                }
            }
        }
    }

    const auto* expected_aux = plans.front().find_cb(ReduceCbRole::Auxiliary);
    TT_FATAL(expected_aux != nullptr, "Reduce sequence planner: call plan is missing its auxiliary CB requirement");
    for (std::size_t i = 1; i < plans.size(); ++i) {
        const auto* auxiliary = plans[i].find_cb(ReduceCbRole::Auxiliary);
        TT_FATAL(
            auxiliary != nullptr && auxiliary->data_format == expected_aux->data_format &&
                auxiliary->page_size == expected_aux->page_size,
            "Reduce sequence planner: one shared auxiliary CB cannot represent the planned call formats");
    }

    ReduceSequencePlan sequence;
    sequence.calls.reserve(reductions.size());
    for (std::size_t i = 0; i < reductions.size(); ++i) {
        const bool is_last = i + 1 == reductions.size();
        sequence.calls.push_back(
            {.input_cb_id = reductions[i].first,
             .auxiliary_cb_id = cb_ids.auxiliary_cb_id,
             .output_cb_id = accumulates && !is_last ? cb_ids.accumulator_cb_id : cb_ids.output_cb_id,
             .accumulator_cb_id = accumulates ? std::optional<std::uint32_t>{cb_ids.accumulator_cb_id} : std::nullopt,
             .accumulation_mode = !accumulates ? ReduceAccumulationMode::None
                                  : is_last    ? ReduceAccumulationMode::Final
                                               : ReduceAccumulationMode::Intermediate,
             .accumulation_index = static_cast<std::uint32_t>(i),
             .plan = std::move(plans[i])});
    }
    return sequence;
}

namespace {

std::uint32_t encode_math(tt::tt_metal::ReduceOpMath math) {
    using reduce_plan_args::Math;
    switch (math) {
        case ReduceOpMath::SUM: return static_cast<std::uint32_t>(Math::Sum);
        case ReduceOpMath::AVG: return static_cast<std::uint32_t>(Math::Average);
        case ReduceOpMath::MAX: return static_cast<std::uint32_t>(Math::Maximum);
        case ReduceOpMath::MIN: return static_cast<std::uint32_t>(Math::Minimum);
        case ReduceOpMath::STD:
        case ReduceOpMath::VAR: TT_THROW("Reduce plan args: Welford reductions are not supported");
    }
    TT_THROW("Reduce plan args: invalid reduction math");
}

std::uint32_t encode_dimension(tt::tt_metal::ReduceOpDim dim) {
    using reduce_plan_args::Dimension;
    switch (dim) {
        case ReduceOpDim::W: return static_cast<std::uint32_t>(Dimension::Row);
        case ReduceOpDim::H: return static_cast<std::uint32_t>(Dimension::Column);
        case ReduceOpDim::HW: return static_cast<std::uint32_t>(Dimension::Scalar);
    }
    TT_THROW("Reduce plan args: invalid reduction dimension");
}

void check_fits(std::uint32_t value, std::uint32_t mask, const char* field) {
    TT_FATAL((value & ~mask) == 0, "Reduce plan args: {} value {} does not fit its compile-time record", field, value);
}

std::uint32_t encode_configuration(const ReduceCallPlan& call) {
    using namespace reduce_plan_args;
    const auto& plan = call.plan;
    const auto path = static_cast<std::uint32_t>(plan.path);
    const auto math = encode_math(plan.reduce_math);
    const auto dimension = encode_dimension(plan.reduce_dim);
    const auto fp32_mode = static_cast<std::uint32_t>(plan.fp32_mode);
    const auto algorithm = static_cast<std::uint32_t>(plan.algorithm);
    const auto input_policy = static_cast<std::uint32_t>(plan.input_policy);
    const auto reload_mode = static_cast<std::uint32_t>(plan.reload_mode);
    const auto reconfig_mode = static_cast<std::uint32_t>(plan.reconfig_mode);
    const auto within_tile = static_cast<std::uint32_t>(plan.within_tile);
    const auto accumulation_mode = static_cast<std::uint32_t>(call.accumulation_mode);
    const auto partial_mode = static_cast<std::uint32_t>(plan.partial_mode);

    check_fits(path, config::path_mask, "path");
    check_fits(math, config::math_mask, "math");
    check_fits(dimension, config::dimension_mask, "dimension");
    check_fits(fp32_mode, config::fp32_mode_mask, "fp32 mode");
    check_fits(algorithm, config::algorithm_mask, "algorithm");
    check_fits(input_policy, config::input_policy_mask, "input policy");
    check_fits(reload_mode, config::reload_mode_mask, "reload mode");
    check_fits(reconfig_mode, config::reconfig_mode_mask, "reconfiguration mode");
    check_fits(within_tile, config::within_tile_mask, "within-tile mode");
    check_fits(accumulation_mode, config::accumulation_mode_mask, "accumulation mode");
    check_fits(partial_mode, config::partial_mode_mask, "partial mode");

    return insert(path, config::path_shift, config::path_mask) | insert(math, config::math_shift, config::math_mask) |
           insert(dimension, config::dimension_shift, config::dimension_mask) |
           insert(fp32_mode, config::fp32_mode_shift, config::fp32_mode_mask) |
           insert(algorithm, config::algorithm_shift, config::algorithm_mask) |
           insert(input_policy, config::input_policy_shift, config::input_policy_mask) |
           insert(reload_mode, config::reload_mode_shift, config::reload_mode_mask) |
           insert(reconfig_mode, config::reconfig_mode_shift, config::reconfig_mode_mask) |
           insert(within_tile, config::within_tile_shift, config::within_tile_mask) |
           insert(accumulation_mode, config::accumulation_mode_shift, config::accumulation_mode_mask) |
           insert(partial_mode, config::partial_mode_shift, config::partial_mode_mask);
}

std::uint32_t encode_circular_buffers(const ReduceCallPlan& call) {
    using namespace reduce_plan_args;
    const auto accumulator_cb_id = call.accumulator_cb_id.value_or(no_cb_id);
    TT_FATAL(
        call.input_cb_id < no_cb_id && call.auxiliary_cb_id < no_cb_id && call.output_cb_id < no_cb_id &&
            (!call.accumulator_cb_id.has_value() || *call.accumulator_cb_id < no_cb_id),
        "Reduce plan args: CB IDs must fit in one byte and 255 is reserved for no accumulator");
    return insert(call.input_cb_id, circular_buffers::input_shift, circular_buffers::id_mask) |
           insert(call.auxiliary_cb_id, circular_buffers::auxiliary_shift, circular_buffers::id_mask) |
           insert(call.output_cb_id, circular_buffers::output_shift, circular_buffers::id_mask) |
           insert(accumulator_cb_id, circular_buffers::accumulator_shift, circular_buffers::id_mask);
}

std::uint32_t encode_chunk_and_auxiliary(const ReducePlan& plan) {
    using namespace reduce_plan_args;
    const auto tile_count = auxiliary_tile_count(plan);
    check_fits(
        plan.chunk.output_tiles, chunk_and_auxiliary::output_tiles_mask, "output tiles per synchronization chunk");
    check_fits(tile_count, chunk_and_auxiliary::auxiliary_tile_count_mask, "auxiliary tile count");
    return insert(
               plan.chunk.output_tiles,
               chunk_and_auxiliary::output_tiles_shift,
               chunk_and_auxiliary::output_tiles_mask) |
           insert(
               tile_count,
               chunk_and_auxiliary::auxiliary_tile_count_shift,
               chunk_and_auxiliary::auxiliary_tile_count_mask);
}

std::uint32_t encode_auxiliary_tile_configuration(std::uint32_t cb_id, const ReduceAuxiliaryTileSpec& tile) {
    using namespace reduce_plan_args;
    const auto type = static_cast<std::uint32_t>(tile.type);
    TT_FATAL(cb_id < no_cb_id, "Reduce plan args: auxiliary CB ID must fit in one byte");
    check_fits(type, auxiliary_configuration::tile_type_mask, "auxiliary tile type");
    check_fits(tile.num_valid_elements, auxiliary_configuration::valid_elements_mask, "auxiliary valid element count");
    TT_FATAL(
        tile.type == ReduceAuxiliaryTileType::Zero || tile.num_valid_elements > 0,
        "Reduce plan args: a non-zero auxiliary tile must contain at least one valid element");
    TT_FATAL(
        tile.type != ReduceAuxiliaryTileType::Zero ||
            (tile.num_valid_elements == 0 && std::bit_cast<std::uint32_t>(tile.value) == 0),
        "Reduce plan args: a zero auxiliary tile must have value 0 and zero valid elements");
    return insert(cb_id, auxiliary_configuration::cb_id_shift, auxiliary_configuration::cb_id_mask) |
           insert(type, auxiliary_configuration::tile_type_shift, auxiliary_configuration::tile_type_mask) |
           insert(
               tile.num_valid_elements,
               auxiliary_configuration::valid_elements_shift,
               auxiliary_configuration::valid_elements_mask);
}

}  // namespace

ReduceCallArgs::ReduceCallArgs(const ReduceCallPlan& call) {
    using reduce_plan_args::CallWord;
    const auto& plan = call.plan;
    const bool accumulates = call.accumulation_mode != ReduceAccumulationMode::None;
    TT_FATAL(
        accumulates == call.accumulator_cb_id.has_value(),
        "Reduce plan args: accumulation mode and accumulator CB presence disagree");
    TT_FATAL(
        call.accumulation_mode != ReduceAccumulationMode::None || call.accumulation_index == 0,
        "Reduce plan args: a non-accumulating call must use accumulation index zero");
    TT_FATAL(
        call.accumulation_mode != ReduceAccumulationMode::Intermediate || call.output_cb_id == call.accumulator_cb_id,
        "Reduce plan args: an intermediate call must write to its accumulator CB");
    TT_FATAL(
        call.accumulation_mode != ReduceAccumulationMode::Final || call.output_cb_id != call.accumulator_cb_id,
        "Reduce plan args: a final call must write outside its accumulator CB");
    TT_FATAL(
        plan.partial_mode == compute_kernel_lib::ReducePartialMode::None ||
            plan.partial_mode == compute_kernel_lib::ReducePartialMode::Scaler ||
            plan.partial_mode == compute_kernel_lib::ReducePartialMode::Mask,
        "Reduce plan args: call contains an unknown partial mode");
    TT_FATAL(
        plan.partial_mode != compute_kernel_lib::ReducePartialMode::Scaler ||
            plan.algorithm == ReduceAlgorithm::ReduceTile,
        "Reduce plan args: a partial-scaler call must use ReduceTile");
    TT_FATAL(
        plan.partial_mode != compute_kernel_lib::ReducePartialMode::Mask ||
            plan.algorithm == ReduceAlgorithm::AccumulateViaAdd,
        "Reduce plan args: a partial-mask call must use AccumulateViaAdd");
    TT_FATAL(
        plan.Ht > 0 && plan.Wt > 0 && plan.batches > 0 && plan.reduce_factor > 0 && plan.chunk.reduce_axis_tiles > 0 &&
            plan.chunk.output_tiles > 0 && !plan.auxiliary_tiles.empty(),
        "Reduce plan args: call contains zero-sized kernel geometry");

    compile_time_args_.reserve(reduce_plan_args::call_compile_time_arg_count(auxiliary_tile_count(plan)));
    const std::uint32_t record[] = {
        encode_configuration(call),
        encode_circular_buffers(call),
        plan.Ht,
        plan.Wt,
        plan.batches,
        plan.input_row_stride_tiles,
        plan.reduce_factor,
        plan.chunk.reduce_axis_tiles,
        encode_chunk_and_auxiliary(plan),
        std::bit_cast<std::uint32_t>(plan.post_scale),
        call.accumulation_index,
    };
    static_assert(std::size(record) == static_cast<std::size_t>(CallWord::Count));
    compile_time_args_.insert(compile_time_args_.end(), std::begin(record), std::end(record));

    for (const auto& tile : plan.auxiliary_tiles) {
        compile_time_args_.push_back(encode_auxiliary_tile_configuration(call.auxiliary_cb_id, tile));
        compile_time_args_.push_back(std::bit_cast<std::uint32_t>(tile.value));
    }
}

ReduceCallArgs::ReduceCallArgs(const ReducePlan& plan, const ReduceCallCbIds& cb_ids) :
    ReduceCallArgs(ReduceCallPlan{
        .input_cb_id = cb_ids.input_cb_id,
        .auxiliary_cb_id = cb_ids.auxiliary_cb_id,
        .output_cb_id = cb_ids.output_cb_id,
        .accumulator_cb_id = std::nullopt,
        .accumulation_mode = ReduceAccumulationMode::None,
        .accumulation_index = 0,
        .plan = plan,
    }) {}

void ReduceCallArgs::append_to(std::vector<std::uint32_t>& compile_time_args) const {
    compile_time_args.insert(compile_time_args.end(), compile_time_args_.begin(), compile_time_args_.end());
}

std::vector<std::uint32_t> ReduceCallArgs::get_compile_time_args() const { return compile_time_args_; }

void ReduceSequencePlan::append_to(std::vector<std::uint32_t>& compile_time_args) const {
    TT_FATAL(!calls.empty(), "Reduce plan args: a call sequence must not be empty");
    compile_time_args.push_back(checked_u32(calls.size(), "reduce call count"));
    for (const auto& call : calls) {
        ReduceCallArgs(call).append_to(compile_time_args);
    }
}

std::vector<std::uint32_t> ReduceSequencePlan::get_compile_time_args() const {
    std::vector<std::uint32_t> compile_time_args;
    append_to(compile_time_args);
    return compile_time_args;
}

}  // namespace ttnn::kernel_lib::host
