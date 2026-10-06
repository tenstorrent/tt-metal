// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <functional>
#include <optional>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "device/repeat_device_operation.hpp"
#include "device/repeat_utils.hpp"
#include "codegen/repeat_codegen_device_operation.hpp"
#include "codegen/repeat_codegen_supported.hpp"
#include "repeat.hpp"
#include "repeat_force.hpp"

namespace ttnn::operations::data_movement::detail {

struct UpperRepeatDims {
    static constexpr uint32_t collapsed_upper = 0;
    static constexpr uint32_t repeat = 1;
    static constexpr uint32_t collapsed_lower = 2;
    static constexpr uint32_t page_size = 3;
};

ttnn::Tensor repeat_upper_dims_rm(
    const ttnn::Tensor& tensor,
    const uint32_t dim,
    const uint32_t repetitions,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> optional_output_tensor = std::nullopt) {
    const auto& input_shape = tensor.logical_shape();
    ttsl::SmallVector<uint32_t> collapsed_shape_vector(4);

    collapsed_shape_vector[UpperRepeatDims::collapsed_upper] =
        std::accumulate(input_shape.cbegin(), input_shape.cbegin() + dim, 1, std::multiplies<uint32_t>());
    collapsed_shape_vector[UpperRepeatDims::repeat] = input_shape[dim];
    collapsed_shape_vector[UpperRepeatDims::collapsed_lower] =
        std::accumulate(input_shape.cbegin() + dim + 1, input_shape.cend() - 1, 1, std::multiplies<uint32_t>());
    collapsed_shape_vector[UpperRepeatDims::page_size] = input_shape[-1];

    auto input_tensor = ttnn::view(tensor, ttnn::Shape(collapsed_shape_vector));

    std::optional<Tensor> prim_output = std::nullopt;
    if (optional_output_tensor.has_value()) {
        auto collapsed_out = collapsed_shape_vector;
        collapsed_out[UpperRepeatDims::repeat] *= repetitions;
        prim_output = ttnn::view(optional_output_tensor.value(), ttnn::Shape(collapsed_out));
    }

    constexpr bool is_final_dim = false;
    auto out_tensor =
        ttnn::prim::repeat(input_tensor, repetitions, is_final_dim, output_mem_config, std::move(prim_output));
    auto expected_shape = input_shape;
    expected_shape[dim] *= repetitions;

    return ttnn::view(out_tensor, ttnn::Shape(expected_shape));
}

ttnn::Tensor repeat_last_dim_rm(
    const ttnn::Tensor& tensor,
    const uint32_t repetitions,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> optional_output_tensor = std::nullopt) {
    const auto& input_shape = tensor.logical_shape();
    ttsl::SmallVector<uint32_t> collapsed_shape_vector(2);

    collapsed_shape_vector[0] =
        std::accumulate(input_shape.cbegin(), input_shape.cend() - 1, 1, std::multiplies<uint32_t>());
    collapsed_shape_vector[1] = input_shape[-1];

    auto input_tensor = ttnn::view(tensor, ttnn::Shape(collapsed_shape_vector));

    std::optional<Tensor> prim_output = std::nullopt;
    if (optional_output_tensor.has_value()) {
        auto collapsed_out = collapsed_shape_vector;
        collapsed_out[1] *= repetitions;
        prim_output = ttnn::view(optional_output_tensor.value(), ttnn::Shape(collapsed_out));
    }

    constexpr bool is_final_dim = true;
    auto out_tensor =
        ttnn::prim::repeat(input_tensor, repetitions, is_final_dim, output_mem_config, std::move(prim_output));

    auto expected_shape = input_shape;
    expected_shape[-1] *= repetitions;

    return ttnn::view(out_tensor, ttnn::Shape(expected_shape));
}

std::tuple<ttnn::Tensor, ttsl::SmallVector<uint32_t>> match_input_rank(
    const ttnn::Tensor& tensor, const ttsl::SmallVector<uint32_t>& repetition_vector) {
    auto working_tensor = tensor;
    const auto& input_shape = working_tensor.logical_shape();
    ttsl::SmallVector<uint32_t> working_repetition_vector;

    const auto total_reps =
        std::accumulate(repetition_vector.cbegin(), repetition_vector.cend(), 1, std::multiplies<uint_fast32_t>());

    if (input_shape.rank() < repetition_vector.size()) {
        ttsl::SmallVector<uint32_t> new_shape_vec(repetition_vector.size(), 1);
        std::copy_backward(input_shape.cbegin(), input_shape.cend(), new_shape_vec.end());
        working_tensor = ttnn::view(working_tensor, ttnn::Shape(new_shape_vec));
        working_repetition_vector = repetition_vector;
    }
    // Pad repetition vector when shorter than tensor rank (torch errors; we allow it).
    else if (repetition_vector.size() < input_shape.rank()) {
        working_repetition_vector.resize(input_shape.rank(), 1);
        std::copy_backward(repetition_vector.cbegin(), repetition_vector.cend(), working_repetition_vector.end());
    }

    else {
        working_repetition_vector = repetition_vector;
    }

    TT_ASSERT(working_tensor.logical_volume() == tensor.logical_volume());
    TT_ASSERT(
        std::accumulate(
            working_repetition_vector.cbegin(),
            working_repetition_vector.cend(),
            1,
            std::multiplies<uint_fast32_t>()) == total_reps);

    return std::tie(working_tensor, working_repetition_vector);
}

bool is_tile_repeat_eligible(const ttnn::Tensor& tensor) {
    if (tensor.layout() != ttnn::TILE_LAYOUT) {
        return false;
    }
    const auto& shape = tensor.logical_shape();
    if (shape.rank() < 2) {
        return false;
    }
    return (shape[-1] % tt::constants::TILE_WIDTH == 0) && (shape[-2] % tt::constants::TILE_HEIGHT == 0);
}

ttnn::Tensor repeat_dim_tile(
    const ttnn::Tensor& tensor,
    const uint32_t dim,
    const uint32_t repetitions,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> optional_output_tensor = std::nullopt) {
    const auto& shape = tensor.logical_shape();
    const auto rank = shape.rank();

    uint32_t h_tiles = shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t w_tiles = shape[-1] / tt::constants::TILE_WIDTH;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
    uint32_t tile_page_size = tt::tile_size(cb_data_format);

    uint32_t higher, rep_dim_pages, lower;

    if (dim == rank - 1) {
        // W dimension: each tile-row's w_tiles get repeated
        higher = std::accumulate(shape.cbegin(), shape.cend() - 2, 1u, std::multiplies<uint32_t>()) * h_tiles;
        rep_dim_pages = w_tiles;
        lower = 1;
    } else if (dim == rank - 2) {
        // H dimension: tile-rows get repeated
        higher = std::accumulate(shape.cbegin(), shape.cend() - 2, 1u, std::multiplies<uint32_t>());
        rep_dim_pages = h_tiles;
        lower = w_tiles;
    } else {
        // Upper dimensions (batch, channel, etc.): groups of tiles get repeated
        higher = std::accumulate(shape.cbegin(), shape.cbegin() + dim, 1u, std::multiplies<uint32_t>());
        uint32_t lower_elements =
            std::accumulate(shape.cbegin() + dim + 1, shape.cend() - 2, 1u, std::multiplies<uint32_t>());
        rep_dim_pages = shape[dim];
        lower = lower_elements * h_tiles * w_tiles;
    }

    return ttnn::prim::repeat_tile(
        tensor,
        repetitions,
        dim,
        output_mem_config,
        higher,
        rep_dim_pages,
        lower,
        tile_page_size,
        std::move(optional_output_tensor));
}

// Single-dim codegen repeat step. prim::repeat_codegen's kernels index pages through a
// fixed 4D page map, so `tensor` is padded up to 4D here (prepending 1s) regardless of
// its original rank; the output is viewed back down to the true logical shape before
// returning.
ttnn::Tensor repeat_dim_codegen(
    const ttnn::Tensor& tensor,
    const uint32_t dim,
    const uint32_t repetitions,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> optional_output_tensor = std::nullopt) {
    const auto& shape = tensor.logical_shape();
    const uint32_t ndim = shape.rank();
    TT_FATAL(ndim <= 4, "RepeatCodegen supports rank <= 4, got {}", ndim);
    const uint32_t pad = ndim < 4 ? 4 - ndim : 0;

    ttnn::Tensor working = tensor;
    if (pad > 0) {
        ttsl::SmallVector<uint32_t> padded_shape(4, 1);
        std::copy(shape.cbegin(), shape.cend(), padded_shape.begin() + pad);
        working = ttnn::view(tensor, ttnn::Shape(padded_shape));
    }
    const uint32_t rep_dim_4d = dim + pad;
    const auto& shape4d = working.logical_shape();

    const auto page_map = ttnn::prim::derive_page_map(working, rep_dim_4d, repetitions);
    ttnn::prim::RepeatCodegenParams params{
        .rep_dim = rep_dim_4d,
        .num_repeats = repetitions,
        .lower_pages = page_map.lower_pages,
        .rep_dim_pages = page_map.rep_dim_pages,
        .total_out_pages = page_map.total_out_pages,
        .stick_size = page_map.stick_size,
        .output_mem_config = output_mem_config,
    };

    std::optional<Tensor> prim_output = std::nullopt;
    if (optional_output_tensor.has_value()) {
        auto expected4d = shape4d;
        expected4d[rep_dim_4d] *= repetitions;
        prim_output = ttnn::view(optional_output_tensor.value(), ttnn::Shape(expected4d));
    }

    auto out = ttnn::prim::repeat_codegen(working, params, std::move(prim_output));
    if (pad == 0) {
        return out;
    }

    auto expected_shape = shape;
    expected_shape[dim] *= repetitions;
    return ttnn::view(out, ttnn::Shape(expected_shape));
}

namespace {

// Strips shard_spec off a sharded input's memory config; the device op re-derives one for the
// new output shape. A preallocated output is where the result has to land, so its own memory
// config outranks both the requested one and the input's.
MemoryConfig derive_output_mem_config(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    if (optional_output_tensor.has_value()) {
        return optional_output_tensor->memory_config();
    }
    const auto& input_mc = input_tensor.memory_config();
    return memory_config.value_or(
        input_mc.is_sharded() ? MemoryConfig(input_mc.memory_layout(), input_mc.buffer_type()) : input_mc);
}

// Fills in a shard_spec for a sharded config that has none, sized to working_tensor. nullopt when no
// valid spec exists; each caller picks its own interleaved fallback.
std::optional<MemoryConfig> synthesize_sharded_mem_config(
    const ttnn::Tensor& working_tensor,
    const MemoryConfig& mem_config,
    std::optional<ShardOrientation> orientation_hint) {
    auto synth = repeat::generate_repeat_shard_spec(
        working_tensor, working_tensor.padded_shape(), mem_config.memory_layout(), orientation_hint);
    if (!synth.has_value()) {
        return std::nullopt;
    }
    return MemoryConfig(mem_config.memory_layout(), mem_config.buffer_type(), synth);
}

bool same_placement(const MemoryConfig& a, const MemoryConfig& b) {
    if (a.memory_layout() != b.memory_layout() || a.buffer_type() != b.buffer_type()) {
        return false;
    }
    return !a.is_sharded() || a.shard_spec() == b.shard_spec();
}

using repeat::interleaved_in;

// A sharded output with no shard_spec -- what a sharded input gets when no memory_config is passed --
// is given the spec native lands it in: the input's spec resized for the repeat when native repeats
// that input in place, otherwise one synthesized for the repeated shape, otherwise interleaved in the
// same buffer type. The codegen legs need the spec up front to decide where the final leg writes.
MemoryConfig resolve_codegen_output_mem_config(
    const ttnn::Tensor& working_tensor,
    const ttsl::SmallVector<uint32_t>& working_repetition_vector,
    const MemoryConfig& output_mem_config) {
    if (!output_mem_config.is_sharded() || output_mem_config.shard_spec().has_value() ||
        working_tensor.storage_type() != StorageType::DEVICE) {
        return output_mem_config;
    }
    const auto& shape = working_tensor.logical_shape();
    if (working_repetition_vector.size() != shape.rank()) {
        return output_mem_config;
    }
    auto out_shape = shape;
    for (size_t i = 0; i < working_repetition_vector.size(); ++i) {
        out_shape[i] *= working_repetition_vector[i];
    }
    const auto& input_shard_spec = working_tensor.shard_spec();
    const auto single = repeat::single_repeated_dim(working_repetition_vector);
    if (single.has_value() && input_shard_spec.has_value() &&
        repeat::is_native_repeat_sharding(
            working_tensor.tensor_spec(), output_mem_config, single->first, single->second)) {
        const auto adjusted = repeat::adjust_repeat_shard_spec_to_shape(
            *input_shard_spec, shape, out_shape, single->first, single->second);
        if (adjusted.has_value()) {
            return MemoryConfig(output_mem_config.memory_layout(), output_mem_config.buffer_type(), adjusted);
        }
    }

    const auto out_spec = repeat::spec_like(
        working_tensor, out_shape, working_tensor.layout(), interleaved_in(output_mem_config.buffer_type()));
    const std::optional<ShardOrientation> orientation_hint =
        input_shard_spec.has_value() ? std::optional{input_shard_spec->orientation} : std::nullopt;
    const auto synthesized = repeat::generate_repeat_shard_spec(
        working_tensor, out_spec.padded_shape(), output_mem_config.memory_layout(), orientation_hint);
    if (synthesized.has_value()) {
        return MemoryConfig(output_mem_config.memory_layout(), output_mem_config.buffer_type(), synthesized);
    }
    return interleaved_in(output_mem_config.buffer_type());
}

// Decomposes a (possibly multi-dim) repeat into single-dim prim::repeat_codegen legs. Each leg is
// independent (orthogonal axes), so leg order affects only intermediate sizes, never the result.
// Intermediate legs are interleaved, in DRAM when the input was unsharded and in the input's buffer
// type otherwise. Only the final leg writes the requested placement, and only when its pages
// line up with it; otherwise one placement hop runs at the end.
ttnn::Tensor repeat_via_codegen(
    const ttnn::Tensor& tensor,
    const ttsl::SmallVector<uint32_t>& repetition_vector,
    const MemoryConfig& output_mem_config,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    const auto plan = repeat_codegen::plan_codegen_legs(tensor, repetition_vector, output_mem_config);
    const std::optional<Tensor> final_out = plan.final_into_prealloc ? optional_output_tensor : std::nullopt;
    const size_t num_legs = plan.rep_dims.size();

    ttnn::Tensor working = tensor;
    if (plan.unshard_input) {
        working = ttnn::to_memory_config(working, interleaved_in(BufferType::DRAM), std::nullopt);
    }

    if (plan.round_trip) {
        // ttnn::untilize rather than to_layout, which bypasses untilize's codegen route on a padded tensor.
        working = ttnn::untilize(working);
    }

    for (size_t i = 0; i < num_legs; ++i) {
        const bool is_final = i + 1 == num_legs;
        const uint32_t d = plan.rep_dims[i];
        working = repeat_dim_codegen(
            working,
            d,
            plan.leg_repeats[d],
            is_final ? plan.final_mc : plan.intermediate_mc,
            is_final ? final_out : std::nullopt);
    }

    // A folded leg leaves its size-1 axis unexpanded; the pages are already in output order. A
    // row-major fold can land on H, and a TILE view cannot move rows across tile padding, so the
    // round trip restores the shape before it retilizes.
    auto out_shape = tensor.logical_shape();
    for (size_t d = 0; d < repetition_vector.size(); ++d) {
        out_shape[d] *= repetition_vector[d];
    }
    if (working.logical_shape() != out_shape) {
        working = ttnn::view(working, out_shape);
    }
    if (plan.round_trip) {
        working = ttnn::to_layout(working, ttnn::TILE_LAYOUT, tensor.dtype());
    }
    if (!same_placement(working.memory_config(), output_mem_config)) {
        working = ttnn::to_memory_config(working, output_mem_config, std::nullopt);
    }
    return working;
}

// Everything a preallocated output has to satisfy. Both routes run this, so a case that lands on
// codegen is held to the same standard as one that lands on native.
void validate_optional_output(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& working_tensor,
    const ttsl::SmallVector<uint32_t>& working_repetition_vector,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    if (!optional_output_tensor.has_value()) {
        return;
    }
    const auto& out = optional_output_tensor.value();

    if (memory_config.has_value()) {
        TT_FATAL(
            memory_config->buffer_type() == out.memory_config().buffer_type() &&
                memory_config->memory_layout() == out.memory_config().memory_layout(),
            "repeat: memory_config must match optional_output_tensor memory config");
        // An omitted shard_spec is derived from the prealloc; an explicit one has to agree with it.
        if (memory_config->shard_spec().has_value()) {
            TT_FATAL(
                memory_config->shard_spec() == out.memory_config().shard_spec(),
                "repeat: memory_config shard_spec must match optional_output_tensor");
        }
    }

    auto expected_logical_shape = working_tensor.logical_shape();
    for (size_t i = 0; i < working_repetition_vector.size(); ++i) {
        expected_logical_shape[i] *= working_repetition_vector[i];
    }
    TT_FATAL(out.device() == input_tensor.device(), "repeat optional output must be on the same device");
    TT_FATAL(out.dtype() == input_tensor.dtype(), "repeat optional output dtype mismatch");
    TT_FATAL(out.layout() == input_tensor.layout(), "repeat optional output layout mismatch");
    TT_FATAL(out.logical_shape() == expected_logical_shape, "repeat optional output shape mismatch");
    // Direct-write kernels read the input while writing the output, so a shared buffer corrupts both.
    TT_FATAL(out.buffer() != input_tensor.buffer(), "repeat: optional_output_tensor must not alias the input buffer");
}

// Copies into the preallocated output when the path that ran could not write it directly.
ttnn::Tensor finalize_into_preallocated(
    const ttnn::Tensor& result, const std::optional<Tensor>& optional_output_tensor) {
    if (!optional_output_tensor.has_value() || result.storage_type() != StorageType::DEVICE) {
        return result;
    }
    const auto& dst = optional_output_tensor.value();
    if (result.buffer() == dst.buffer()) {
        return result;
    }
    return ttnn::copy(result, dst);
}

}  // namespace

// The existing composite/native implementation, unconditionally. Callers that have already been
// routed here enter this rather than re-entering ttnn::repeat, so a call routed to native cannot
// be routed a second time and land on codegen partway through.
ttnn::Tensor repeat_native(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& repetition_vector,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    auto [working_tensor, working_repetition_vector] = match_input_rank(input_tensor, repetition_vector);
    MemoryConfig output_mem_config = derive_output_mem_config(input_tensor, memory_config, optional_output_tensor);
    auto working_output_mem_config = output_mem_config;

    validate_optional_output(
        input_tensor, working_tensor, working_repetition_vector, memory_config, optional_output_tensor);

    if (std::any_of(
            working_repetition_vector.cbegin(), working_repetition_vector.cend(), [](auto x) { return x == 0; })) {
        // Zero-repetition: allocate zeros with zero-volume shape.
        const auto& shape = working_tensor.logical_shape();
        std::transform(
            shape.cbegin(),
            shape.cend(),
            working_repetition_vector.cbegin(),
            working_repetition_vector.begin(),
            std::multiplies<uint32_t>());
        const MemoryConfig zero_mc =
            optional_output_tensor.has_value()
                ? optional_output_tensor->memory_config()
                : memory_config.value_or(
                      MemoryConfig{TensorMemoryLayout::INTERLEAVED, input_tensor.memory_config().buffer_type()});
        auto zeros_out = ttnn::zeros(
            ttnn::Shape(working_repetition_vector),
            input_tensor.dtype(),
            input_tensor.layout(),
            *input_tensor.device(),
            zero_mc);
        return finalize_into_preallocated(zeros_out, optional_output_tensor);
    }

    TT_FATAL(working_tensor.logical_shape().rank() > 0, "repeat does not support rank 0 tensors");

    // nothing to do!
    if (std::all_of(
            working_repetition_vector.cbegin(), working_repetition_vector.cend(), [](auto x) { return x == 1; })) {
        // working_tensor holds the rank expansion from match_input_rank. input_tensor does not.
        // A sharded config with neither a shard_spec nor an nd_shard_spec can't be handed to
        // to_memory_config: reuse the input's spec when it already has that layout (rank expansion
        // leaves it valid), else synthesize one, else stay interleaved like the composite path.
        // An ND config carries only nd_shard_spec and is passed through as requested.
        MemoryConfig all_ones_mc = output_mem_config;
        if (all_ones_mc.is_sharded() && !all_ones_mc.shard_spec().has_value() &&
            !all_ones_mc.nd_shard_spec().has_value()) {
            const auto& input_mc = input_tensor.memory_config();
            if (input_mc.memory_layout() == all_ones_mc.memory_layout() &&
                input_mc.buffer_type() == all_ones_mc.buffer_type()) {
                all_ones_mc = input_mc;
            } else {
                std::optional<ShardOrientation> orientation_hint;
                if (input_tensor.shard_spec().has_value()) {
                    orientation_hint = input_tensor.shard_spec()->orientation;
                }
                all_ones_mc = synthesize_sharded_mem_config(working_tensor, all_ones_mc, orientation_hint)
                                  .value_or(MemoryConfig(TensorMemoryLayout::INTERLEAVED, all_ones_mc.buffer_type()));
            }
        }
        return finalize_into_preallocated(ttnn::to_memory_config(working_tensor, all_ones_mc), optional_output_tensor);
    }

    // Direct prim write only when no later layout/reshard hop will reallocate.
    const bool needs_rm_tilize_roundtrip = input_tensor.layout() == ttnn::TILE_LAYOUT &&
                                           !is_tile_repeat_eligible(working_tensor);
    const bool needs_final_i2s = output_mem_config.is_sharded();  // refined after native_sharded below

    // Native path: sharded input, single-axis repeat, predicate accepts. Else composite.
    bool native_sharded = false;
    if (input_tensor.memory_config().is_sharded()) {
        const auto non_one_count = std::count_if(
            working_repetition_vector.cbegin(), working_repetition_vector.cend(), [](uint32_t r) { return r != 1; });
        if (non_one_count == 1) {
            int32_t native_dim = -1;
            uint32_t native_reps = 1;
            for (size_t i = 0; i < working_repetition_vector.size(); ++i) {
                if (working_repetition_vector[i] != 1) {
                    native_dim = static_cast<int32_t>(i);
                    native_reps = working_repetition_vector[i];
                    break;
                }
            }
            native_sharded = repeat::is_native_repeat_sharding(
                working_tensor.tensor_spec(), std::optional<MemoryConfig>{output_mem_config}, native_dim, native_reps);
        }
    }

    const bool will_i2s = !native_sharded && needs_final_i2s;
    // Prim can own the prealloc when no subsequent to_layout/i2s reallocates.
    const bool prim_can_land = optional_output_tensor.has_value() && !needs_rm_tilize_roundtrip && !will_i2s &&
                               working_tensor.buffer() != optional_output_tensor->buffer();

    // Snapshot orientation before the L1-interleaved staging hop strips it.
    std::optional<ShardOrientation> input_orientation_hint;
    if (!native_sharded) {
        if (input_tensor.shard_spec().has_value()) {
            input_orientation_hint = input_tensor.shard_spec()->orientation;
        }
        if (working_tensor.memory_config().is_sharded()) {
            // DRAM-sharded fallback via to_memory_config (sharded_to_interleaved is L1-only);
            // use working_tensor to keep rank padding from match_input_rank.
            const MemoryConfig l1_interleaved{TensorMemoryLayout::INTERLEAVED, BufferType::L1};
            working_tensor = ttnn::to_memory_config(working_tensor, l1_interleaved, std::nullopt);
        }
        if (working_output_mem_config.is_sharded()) {
            working_output_mem_config =
                MemoryConfig{TensorMemoryLayout::INTERLEAVED, working_output_mem_config.buffer_type()};
        }
    }

    if (is_tile_repeat_eligible(working_tensor)) {
        // Tile-native path; skip TILE->RM->TILE.
        for (auto it = working_repetition_vector.crbegin(); it != working_repetition_vector.crend(); ++it) {
            if (*it == 1) {
                continue;
            }
            auto dim = working_repetition_vector.crend() - it - 1;
            const bool is_last =
                std::none_of(std::next(it), working_repetition_vector.crend(), [](uint32_t r) { return r != 1; });
            auto step_out = (prim_can_land && is_last) ? optional_output_tensor : std::nullopt;
            working_tensor = repeat_dim_tile(
                working_tensor, dim, *it, working_output_mem_config, std::move(step_out));
        }
    } else {
        // RM path: TILE->RM, repeat, RM->TILE.
        if (working_tensor.layout() == ttnn::TILE_LAYOUT) {
            working_tensor = ttnn::to_layout(working_tensor, ttnn::ROW_MAJOR_LAYOUT);
        }

        for (auto it = working_repetition_vector.crbegin(); it != working_repetition_vector.crend(); ++it) {
            if (*it == 1) {
                continue;
            }
            const bool is_last =
                std::none_of(std::next(it), working_repetition_vector.crend(), [](uint32_t r) { return r != 1; });
            // RM prim lands only when final layout already matches (no later tilize).
            auto step_out = (prim_can_land && is_last) ? optional_output_tensor : std::nullopt;
            if (it == working_repetition_vector.crbegin()) {
                working_tensor = repeat_last_dim_rm(
                    working_tensor, *it, working_output_mem_config, std::move(step_out));
            } else {
                auto i = working_repetition_vector.crend() - it - 1;
                working_tensor = repeat_upper_dims_rm(
                    working_tensor, i, *it, working_output_mem_config, std::move(step_out));
            }
        }

        if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
            working_tensor = ttnn::to_layout(working_tensor, ttnn::TILE_LAYOUT, input_tensor.dtype());
        }
    }

    // Composite-only re-shard; native path already wrote sharded output.
    if (!native_sharded && output_mem_config.is_sharded()) {
        MemoryConfig final_mc = output_mem_config;
        if (!final_mc.shard_spec().has_value()) {
            auto synth = synthesize_sharded_mem_config(working_tensor, final_mc, input_orientation_hint);
            if (!synth.has_value()) {
                // No valid spec; keep interleaved.
                return finalize_into_preallocated(working_tensor, optional_output_tensor);
            }
            final_mc = *synth;
        }
        auto i2s_out = optional_output_tensor.has_value() ? optional_output_tensor : std::nullopt;
        working_tensor = ttnn::interleaved_to_sharded(
            working_tensor, final_mc, /*data_type_arg=*/std::nullopt, /*keep_l1_aligned=*/std::nullopt, i2s_out);
    }

    return finalize_into_preallocated(working_tensor, optional_output_tensor);
}

ttnn::Tensor repeat_force_native(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& repetition_vector,
    const std::optional<MemoryConfig>& memory_config) {
    return repeat_native(input_tensor, repetition_vector, memory_config, std::nullopt);
}

ttnn::Tensor repeat_force_codegen(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& repetition_vector,
    const std::optional<MemoryConfig>& memory_config) {
    auto [working_tensor, working_repetition_vector] = match_input_rank(input_tensor, repetition_vector);
    const MemoryConfig output_mem_config = resolve_codegen_output_mem_config(
        working_tensor, working_repetition_vector, derive_output_mem_config(input_tensor, memory_config));
    // Never falls back to native, so a forced call measures codegen even where routing would demote it.
    TT_FATAL(
        repeat_codegen::supported_by_codegen(working_tensor, working_repetition_vector, output_mem_config),
        "repeat_force_codegen invoked for a case the codegen path does not support "
        "(repeat_codegen::supported_by_codegen); use ttnn::repeat to route it");
    return repeat_via_codegen(working_tensor, working_repetition_vector, output_mem_config);
}

}  // namespace ttnn::operations::data_movement::detail

namespace ttnn {

ttnn::Tensor repeat(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& repetition_vector,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    namespace detail = operations::data_movement::detail;
    namespace repeat_codegen = operations::data_movement::repeat_codegen;

    auto [working_tensor, working_repetition_vector] = detail::match_input_rank(input_tensor, repetition_vector);
    const MemoryConfig output_mem_config =
        detail::derive_output_mem_config(input_tensor, memory_config, optional_output_tensor);
    // Native derives its own spec from output_mem_config; only the codegen route needs it resolved.
    const MemoryConfig codegen_output_mem_config =
        detail::resolve_codegen_output_mem_config(working_tensor, working_repetition_vector, output_mem_config);
    // Ahead of the routing decision, so a rejected preallocated output raises the same way whichever
    // route the case would have taken.
    detail::validate_optional_output(
        input_tensor, working_tensor, working_repetition_vector, memory_config, optional_output_tensor);

    // compute_output_specs() hands a preallocated output's spec straight back, so a tile the input
    // does not share would reach kernels generated for the input's pages.
    const bool output_page_ok = !optional_output_tensor.has_value() ||
                                repeat_codegen::output_matches_input_page(input_tensor, *optional_output_tensor);
    // The demotion predicates are cheap placement and shape tests; the support gate budgets L1 per leg,
    // so it runs only for a call that could still take the codegen route.
    if (output_page_ok &&
        !repeat_codegen::is_demoted(working_tensor, working_repetition_vector, codegen_output_mem_config) &&
        repeat_codegen::supported_by_codegen(working_tensor, working_repetition_vector, codegen_output_mem_config) &&
        repeat_codegen::row_major_cbs_fit_free_l1(
            working_tensor, working_repetition_vector, codegen_output_mem_config, optional_output_tensor.has_value())) {
        // The final leg lands in the prealloc when it can write that placement directly; otherwise the
        // result is copied in.
        return detail::finalize_into_preallocated(
            detail::repeat_via_codegen(
                working_tensor, working_repetition_vector, codegen_output_mem_config, optional_output_tensor),
            optional_output_tensor);
    }

    return detail::repeat_native(input_tensor, repetition_vector, memory_config, optional_output_tensor);
}

ttnn::Tensor repeat(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& repeat_dims,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::repeat(
        input_tensor,
        ttsl::SmallVector<uint32_t>(repeat_dims.cbegin(), repeat_dims.cend()),
        std::nullopt,
        optional_output_tensor);
}

}  // namespace ttnn
