// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <functional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
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

namespace ttnn::operations::data_movement::detail {

struct UpperRepeatDims {
    static constexpr uint32_t collapsed_upper = 0;
    static constexpr uint32_t repeat = 1;
    static constexpr uint32_t collapsed_lower = 2;
    static constexpr uint32_t page_size = 3;
};

ttnn::Tensor repeat_upper_dims_rm(
    const ttnn::Tensor& tensor, const uint32_t dim, const uint32_t repetitions, const MemoryConfig& output_mem_config) {
    const auto& input_shape = tensor.logical_shape();
    ttsl::SmallVector<uint32_t> collapsed_shape_vector(4);

    collapsed_shape_vector[UpperRepeatDims::collapsed_upper] =
        std::accumulate(input_shape.cbegin(), input_shape.cbegin() + dim, 1, std::multiplies<uint32_t>());
    collapsed_shape_vector[UpperRepeatDims::repeat] = input_shape[dim];
    collapsed_shape_vector[UpperRepeatDims::collapsed_lower] =
        std::accumulate(input_shape.cbegin() + dim + 1, input_shape.cend() - 1, 1, std::multiplies<uint32_t>());
    collapsed_shape_vector[UpperRepeatDims::page_size] = input_shape[-1];

    auto input_tensor = ttnn::view(tensor, ttnn::Shape(collapsed_shape_vector));

    constexpr bool is_final_dim = false;
    auto out_tensor = ttnn::prim::repeat(input_tensor, repetitions, is_final_dim, output_mem_config);
    auto expected_shape = input_shape;
    expected_shape[dim] *= repetitions;

    return ttnn::view(out_tensor, ttnn::Shape(expected_shape));
}

ttnn::Tensor repeat_last_dim_rm(
    const ttnn::Tensor& tensor, const uint32_t repetitions, const MemoryConfig& output_mem_config) {
    const auto& input_shape = tensor.logical_shape();
    ttsl::SmallVector<uint32_t> collapsed_shape_vector(2);

    collapsed_shape_vector[0] =
        std::accumulate(input_shape.cbegin(), input_shape.cend() - 1, 1, std::multiplies<uint32_t>());
    collapsed_shape_vector[1] = input_shape[-1];

    auto input_tensor = ttnn::view(tensor, ttnn::Shape(collapsed_shape_vector));

    constexpr bool is_final_dim = true;
    auto out_tensor = ttnn::prim::repeat(input_tensor, repetitions, is_final_dim, output_mem_config);

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
    const ttnn::Tensor& tensor, const uint32_t dim, const uint32_t repetitions, const MemoryConfig& output_mem_config) {
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
        tensor, repetitions, dim, output_mem_config, higher, rep_dim_pages, lower, tile_page_size);
}

// Single-dim codegen repeat step. repeat_codegen's kernels assume a 4D input
// (ops/repeat/spec.py's _page_map / build_repeat_rm_factory), so `tensor` is
// padded up to 4D here (prepending 1s) regardless of its original rank; the
// output is viewed back down to the true logical shape before returning.
ttnn::Tensor repeat_dim_codegen(
    const ttnn::Tensor& tensor, const uint32_t dim, const uint32_t repetitions, const MemoryConfig& output_mem_config) {
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

    uint32_t lower_pages = 0;
    uint32_t rep_dim_pages = 0;
    uint32_t total_out_pages = 0;
    uint32_t stick_size = 0;

    if (working.layout() == ttnn::TILE_LAYOUT) {
        const uint32_t Ht = (shape4d[2] + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
        const uint32_t Wt = (shape4d[3] + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
        const std::array<uint32_t, 4> dim_pages = {shape4d[0], shape4d[1], Ht, Wt};
        lower_pages = 1;
        for (uint32_t d = rep_dim_4d + 1; d < 4; ++d) {
            lower_pages *= dim_pages[d];
        }
        rep_dim_pages = dim_pages[rep_dim_4d];
        total_out_pages = dim_pages[0] * dim_pages[1] * dim_pages[2] * dim_pages[3] * repetitions;
        // stick_size stays 0: unused on the TILE branch.
    } else {
        stick_size = shape4d[3] * working.element_size();
        if (rep_dim_4d == 3) {
            // Last-dim (within-stick) path: page count is unaffected by the repeat.
            total_out_pages = shape4d[0] * shape4d[1] * shape4d[2];
        } else {
            const std::array<uint32_t, 4> dim_pages = {shape4d[0], shape4d[1], shape4d[2], 1};
            lower_pages = 1;
            for (uint32_t d = rep_dim_4d + 1; d < 4; ++d) {
                lower_pages *= dim_pages[d];
            }
            const uint32_t total_src_pages = dim_pages[0] * dim_pages[1] * dim_pages[2] * dim_pages[3];
            rep_dim_pages = dim_pages[rep_dim_4d];
            total_out_pages = total_src_pages * repetitions;
        }
    }

    ttnn::prim::RepeatCodegenParams params{
        .rep_dim = rep_dim_4d,
        .num_repeats = repetitions,
        .lower_pages = lower_pages,
        .rep_dim_pages = rep_dim_pages,
        .total_out_pages = total_out_pages,
        .stick_size = stick_size,
        .output_mem_config = output_mem_config,
    };

    auto out = ttnn::prim::repeat_codegen(working, params);

    auto expected_shape = shape;
    expected_shape[dim] *= repetitions;
    return ttnn::view(out, ttnn::Shape(expected_shape));
}

// Decomposes a (possibly multi-dim) repeat into a sequence of single-dim
// prim::repeat_codegen calls, mirroring the reverse-order per-dim loops
// above (native TILE/RM) and ops/repeat/repeat.py's RepeatCodegen.repeat --
// each single-dim step is independent (orthogonal axes), so iteration order
// doesn't affect correctness.
ttnn::Tensor repeat_via_codegen(
    const ttnn::Tensor& tensor,
    const ttsl::SmallVector<uint32_t>& repetition_vector,
    const MemoryConfig& output_mem_config) {
    ttnn::Tensor working_tensor = tensor;
    for (auto it = repetition_vector.crbegin(); it != repetition_vector.crend(); ++it) {
        if (*it == 1) {
            continue;
        }
        const auto dim = repetition_vector.crend() - it - 1;
        working_tensor = repeat_dim_codegen(working_tensor, dim, *it, output_mem_config);
    }
    return working_tensor;
}

}  // namespace ttnn::operations::data_movement::detail

namespace ttnn {

ttnn::Tensor repeat(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& repetition_vector,
    const std::optional<MemoryConfig>& memory_config,
    const std::string& implementation) {
    namespace repeat_codegen = operations::data_movement::repeat_codegen;
    // Validate before any early return so invalid values fail consistently.
    const auto sel = repeat_codegen::parse_implementation(implementation);

    auto [working_tensor, working_repetition_vector] =
        operations::data_movement::detail::match_input_rank(input_tensor, repetition_vector);
    // Strip shard_spec from sharded input; device op re-derives for new output shape.
    const auto& input_mc = input_tensor.memory_config();
    MemoryConfig output_mem_config = memory_config.value_or(
        input_mc.is_sharded() ? MemoryConfig(input_mc.memory_layout(), input_mc.buffer_type()) : input_mc);
    auto working_output_mem_config = output_mem_config;

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
        const MemoryConfig zero_mc = memory_config.value_or(
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, input_tensor.memory_config().buffer_type()});
        return ttnn::zeros(
            ttnn::Shape(working_repetition_vector),
            input_tensor.dtype(),
            input_tensor.layout(),
            *input_tensor.device(),
            zero_mc);
    }

    TT_FATAL(working_tensor.logical_shape().rank() > 0, "repeat does not support rank 0 tensors");

    // nothing to do!
    if (std::all_of(
            working_repetition_vector.cbegin(), working_repetition_vector.cend(), [](auto x) { return x == 1; })) {
        return input_tensor;
    }

    {
        // Sharded *output* has no resharding path in this port (codegen only ever produces an
        // interleaved output tensor); gate it here rather than in supported_by_codegen(), which
        // is about the input side per the manifest's hand-authored sharded case.
        const bool codegen_output_ok = !output_mem_config.is_sharded();
        if (sel != repeat_codegen::ImplementationSelector::Native) {
            const bool supported =
                codegen_output_ok && repeat_codegen::supported_by_codegen(working_tensor, working_repetition_vector);
            if (sel == repeat_codegen::ImplementationSelector::Codegen) {
                TT_FATAL(
                    supported,
                    "repeat: implementation=\"codegen\" requires a supported input and an interleaved output "
                    "memory configuration");
                return operations::data_movement::detail::repeat_via_codegen(
                    working_tensor, working_repetition_vector, output_mem_config);
            }
            // Auto: codegen iff supported and not perf-demoted; else fall through to native below.
            if (supported && !repeat_codegen::is_demoted(working_tensor, working_repetition_vector)) {
                return operations::data_movement::detail::repeat_via_codegen(
                    working_tensor, working_repetition_vector, output_mem_config);
            }
        }
    }

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
            native_sharded = operations::data_movement::repeat::is_native_repeat_sharding(
                working_tensor.tensor_spec(), std::optional<MemoryConfig>{output_mem_config}, native_dim, native_reps);
        }
    }

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

    if (operations::data_movement::detail::is_tile_repeat_eligible(working_tensor)) {
        // Tile-native path; skip TILE->RM->TILE.
        for (auto it = working_repetition_vector.crbegin(); it != working_repetition_vector.crend(); ++it) {
            if (*it == 1) {
                continue;
            }
            auto dim = working_repetition_vector.crend() - it - 1;
            working_tensor =
                operations::data_movement::detail::repeat_dim_tile(working_tensor, dim, *it, working_output_mem_config);
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
            if (it == working_repetition_vector.crbegin()) {
                working_tensor = operations::data_movement::detail::repeat_last_dim_rm(
                    working_tensor, *it, working_output_mem_config);
            } else {
                auto i = working_repetition_vector.crend() - it - 1;
                working_tensor = operations::data_movement::detail::repeat_upper_dims_rm(
                    working_tensor, i, *it, working_output_mem_config);
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
            auto synth = operations::data_movement::repeat::generate_repeat_shard_spec(
                working_tensor, working_tensor.padded_shape(), final_mc.memory_layout(), input_orientation_hint);
            if (synth.has_value()) {
                final_mc = MemoryConfig(final_mc.memory_layout(), final_mc.buffer_type(), synth);
            } else {
                return working_tensor;  // No valid spec; keep interleaved.
            }
        }
        working_tensor = ttnn::interleaved_to_sharded(working_tensor, final_mc, std::nullopt);
    }

    return working_tensor;
}

ttnn::Tensor repeat(
    const ttnn::Tensor& input_tensor, const ttnn::Shape& repeat_dims, const std::string& implementation) {
    return ttnn::repeat(
        input_tensor,
        ttsl::SmallVector<uint32_t>(repeat_dims.cbegin(), repeat_dims.cend()),
        std::nullopt,
        implementation);
}

}  // namespace ttnn
