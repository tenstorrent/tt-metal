// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_host.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

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
    float scalar,
    std::uint32_t logical_reduce_elements,
    std::uint32_t partial_elements,
    bool has_partial) {
    if (plan.algorithm == ReduceAlgorithm::AccumulateViaAdd) {
        plan.reader_scaler = 1.0F;
        plan.reduce_factor = math == ReduceOpMath::AVG ? logical_reduce_elements : 1U;
        plan.post_scale = math == ReduceOpMath::AVG ? scalar * logical_reduce_elements : scalar;
        plan.auxiliary_kind = has_partial ? ReduceAuxiliaryKind::Mask : ReduceAuxiliaryKind::Scalar;
        plan.auxiliary_tile_count = 1;
    } else {
        TT_FATAL(
            dim != ReduceOpDim::HW || scalar >= 0.0F,
            "Reduce planner: ReduceTile HW reduction cannot represent negative scalar {}",
            scalar);
        plan.reader_scaler = dim == ReduceOpDim::HW ? std::sqrt(scalar) : scalar;
        plan.post_scale = 1.0F;
        plan.reduce_factor = 1;
        plan.auxiliary_kind = has_partial ? ReduceAuxiliaryKind::FullAndPartialScaler : ReduceAuxiliaryKind::Scalar;
        plan.auxiliary_tile_count = has_partial ? 2U : 1U;
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
    configure_scalar_and_aux(plan, math, dim, scalar, logical_reduce_elements, partial_elements, has_axis_partial);

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
    std::size_t aux_bytes = static_cast<std::size_t>(plan.auxiliary_tile_count) * aux_tile_bytes;
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
                    plan, math, dim, scalar, logical_reduce_elements, partial_elements, has_axis_partial);
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
        aux_bytes = static_cast<std::size_t>(plan.auxiliary_tile_count) * aux_tile_bytes;
    }

    add_requirement(
        plan,
        {.role = ReduceCbRole::Auxiliary,
         .data_format = aux_format,
         .page_size = aux_tile_bytes,
         .page_count = plan.auxiliary_tile_count,
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
    configure_scalar_and_aux(plan, math, dim, scalar, logical_reduce_elements, partial_elements, false);
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
    plan.reload_mode = plan.algorithm == ReduceAlgorithm::AccumulateViaAdd && !has_partial
                           ? compute_kernel_lib::AccumulateReloadMode::CopySeedZeroPair
                           : compute_kernel_lib::AccumulateReloadMode::CopySeedPairs;
    if (plan.algorithm == ReduceAlgorithm::AccumulateViaAdd && reduced_tiles != axis_chunk && !has_partial) {
        plan.auxiliary_kind = ReduceAuxiliaryKind::Zero;
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
         .page_count = plan.auxiliary_tile_count,
         .total_size_bytes = static_cast<std::size_t>(plan.auxiliary_tile_count) * aux_tile_bytes});
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

    if (input_spec.layout() == Layout::ROW_MAJOR) {
        return make_row_major_plan(
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
    return make_tiled_plan(
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

ReduceSequencePlan make_reduce_plan(
    const std::vector<ReduceCbConfig>& reductions,
    const ReduceSequenceCbIds& cb_ids,
    const ReduceHardwareConfig& hardware) {
    TT_FATAL(!reductions.empty(), "Reduce sequence planner: at least one input CB is required");

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
                plan.reader_scaler = 1.0F;
                plan.reduce_factor = grand_factor;
                plan.post_scale = common_post_multiplier;
            } else {
                plan.reader_scaler =
                    first_config.reduce_dim == ReduceOpDim::HW ? std::sqrt(grand_scalar) : grand_scalar;
                plan.reduce_factor = 1;
                plan.post_scale = 1.0F;
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
             .accumulation_index = static_cast<std::uint32_t>(i),
             .is_last = is_last,
             .plan = std::move(plans[i])});
    }
    return sequence;
}

void add_reduce_plan_defines(std::map<std::string, std::string>& defines, const ReducePlan& plan) {
    using compute_kernel_lib::AccumulateReloadMode;
    using compute_kernel_lib::ReduceAlgorithm;
    using compute_kernel_lib::ReduceInputPolicy;

    defines["REDUCE_ALGORITHM"] = plan.algorithm == ReduceAlgorithm::AccumulateViaAdd
                                      ? "compute_kernel_lib::ReduceAlgorithm::AccumulateViaAdd"
                                      : "compute_kernel_lib::ReduceAlgorithm::ReduceTile";
    switch (plan.input_policy) {
        case ReduceInputPolicy::WaitAndPopPerTile:
            defines["REDUCE_INPUT_POLICY"] = "compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile";
            break;
        case ReduceInputPolicy::BulkWaitBulkPop:
            defines["REDUCE_INPUT_POLICY"] = "compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop";
            break;
        case ReduceInputPolicy::WaitUpfrontNoPop:
            defines["REDUCE_INPUT_POLICY"] = "compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop";
            break;
        case ReduceInputPolicy::NoWaitNoPop:
            defines["REDUCE_INPUT_POLICY"] = "compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop";
            break;
        case ReduceInputPolicy::ChunkedWaitChunkedPop:
            defines["REDUCE_INPUT_POLICY"] = "compute_kernel_lib::ReduceInputPolicy::ChunkedWaitChunkedPop";
            break;
    }
    switch (plan.reload_mode) {
        case AccumulateReloadMode::FoldViaAdd:
            defines["REDUCE_RELOAD_MODE"] = "compute_kernel_lib::AccumulateReloadMode::FoldViaAdd";
            break;
        case AccumulateReloadMode::CopySeedPairs:
            defines["REDUCE_RELOAD_MODE"] = "compute_kernel_lib::AccumulateReloadMode::CopySeedPairs";
            break;
        case AccumulateReloadMode::CopySeedUniform:
            defines["REDUCE_RELOAD_MODE"] = "compute_kernel_lib::AccumulateReloadMode::CopySeedUniform";
            break;
        case AccumulateReloadMode::CopySeedSfpuAdd:
            defines["REDUCE_RELOAD_MODE"] = "compute_kernel_lib::AccumulateReloadMode::CopySeedSfpuAdd";
            break;
        case AccumulateReloadMode::CopySeedZeroPair:
            defines["REDUCE_RELOAD_MODE"] = "compute_kernel_lib::AccumulateReloadMode::CopySeedZeroPair";
            break;
    }
    defines["REDUCE_FACTOR"] = std::to_string(plan.reduce_factor);
    defines["REDUCE_INPUT_CHUNK_TILES"] = std::to_string(plan.chunk.reduce_axis_tiles);
    defines["REDUCE_OUTPUT_CHUNK_TILES"] = std::to_string(plan.chunk.output_tiles);
    defines["REDUCE_AUX_TILE_COUNT"] = std::to_string(plan.auxiliary_tile_count);
    defines["REDUCE_PARTIAL_VALID"] = std::to_string(plan.partial_reduce_axis_elements);
    switch (plan.auxiliary_kind) {
        case ReduceAuxiliaryKind::FullAndPartialScaler: defines["REDUCE_AUX_SCALER_PAIR"] = "1"; break;
        case ReduceAuxiliaryKind::Mask: defines["REDUCE_AUX_MASK"] = "1"; break;
        case ReduceAuxiliaryKind::Zero: defines["REDUCE_AUX_ZERO"] = "1"; break;
        case ReduceAuxiliaryKind::Scalar: break;
    }
}

}  // namespace ttnn::kernel_lib::host
