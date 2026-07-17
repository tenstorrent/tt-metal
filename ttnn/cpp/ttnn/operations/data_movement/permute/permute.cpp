// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute.hpp"

#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/transpose/device/transpose_utils.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include "ttnn/operations/data_movement/permute/codegen/permute_codegen_device_operation.hpp"
#include "ttnn/operations/data_movement/permute/codegen/permute_codegen_supported.hpp"

#include <tt-metalium/hal.hpp>
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn::operations::data_movement::detail {

inline bool is_rm_block_or_width_sharded(const ttnn::Tensor& t) {
    if (t.layout() != Layout::ROW_MAJOR || !t.is_sharded()) {
        return false;
    }
    auto ml = t.memory_config().memory_layout();
    return ml == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED ||
           ml == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
}

ttnn::Tensor permute_impl(
    const ttnn::Tensor& a,
    const ttsl::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& output_mem_config,
    float pad_value = 0.0f) {
    uint32_t rank = a.logical_shape().rank();

    // Irregular RM block/width sharded hits a pages_per_shard misread in noc_async_*_sharded.
    // Input-side guard only; irregular output shapes are safe (writers emit full rows, compute_output_specs synthesises
    // a valid spec).
    const auto& input_logical = a.logical_shape();
    const bool irregular_hw = input_logical.rank() >= 2 && (input_logical[-1] % tt::constants::TILE_WIDTH != 0 ||
                                                            input_logical[-2] % tt::constants::TILE_HEIGHT != 0);
    if (is_rm_block_or_width_sharded(a) && irregular_hw) {
        // Snapshot orientation before the L1-interleaved staging hop strips it.
        std::optional<tt::tt_metal::ShardOrientation> input_orientation_hint;
        if (a.shard_spec().has_value()) {
            input_orientation_hint = a.shard_spec()->orientation;
        }
        auto resolved_mc = output_mem_config;
        if (output_mem_config.has_value() && output_mem_config->is_sharded() &&
            !output_mem_config->shard_spec().has_value()) {
            const auto& padded_in = a.padded_shape();
            ttsl::SmallVector<uint32_t> out_padded_vec(padded_in.rank());
            for (size_t i = 0; i < dims.size(); ++i) {
                out_padded_vec[i] = padded_in[dims[i]];
            }
            auto output_padded_shape = ttnn::Shape(std::move(out_padded_vec));
            auto spec = ttnn::operations::data_movement::transpose::generate_transpose_shard_spec(
                a, output_padded_shape, output_mem_config->memory_layout(), input_orientation_hint);
            resolved_mc =
                MemoryConfig(output_mem_config->memory_layout(), output_mem_config->buffer_type(), std::move(spec));
        }
        const auto interleaved_l1 =
            MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);
        auto x = ttnn::to_memory_config(a, interleaved_l1, std::nullopt);
        return permute_impl(x, dims, resolved_mc, pad_value);
    }

    auto prim_permute = [&](const ttnn::Tensor& input) -> ttnn::Tensor {
        return ttnn::prim::permute(input, dims, output_mem_config.value_or(a.memory_config()), std::nullopt, pad_value);
    };

    if (rank > 4) {
        if (a.is_sharded()) {
            // Preserve input's shard layout; compute_output_specs derives a valid spec.
            auto effective_config = output_mem_config.value_or(
                MemoryConfig(a.memory_config().memory_layout(), a.memory_config().buffer_type()));
            return ttnn::prim::permute(a, dims, effective_config, std::nullopt, pad_value);
        }
        return prim_permute(a);
    }

    TT_FATAL(dims.size() == 4, "Only 4D tensor are supported for permute.");
    uint32_t N = dims[0], C = dims[1], H = dims[2], W = dims[3];

    auto formatted_input_tensor = a;
    // WH and CN should be supported without typecast.
    bool wh = N == 0 && C == 1 && H == 3 && W == 2;
    bool cn = N == 1 && C == 0 && H == 2 && W == 3;
    bool cnwh = N == 1 && C == 0 && H == 3 && W == 2;
    bool bfloat8_supported = wh || cn || cnwh;
    bool typecast = formatted_input_tensor.dtype() == DataType::BFLOAT8_B and !bfloat8_supported && !a.is_sharded();
    formatted_input_tensor =
        typecast ? ttnn::typecast(formatted_input_tensor, DataType::BFLOAT16) : formatted_input_tensor;

    auto output = formatted_input_tensor;
    auto transpose_wh = [&](const ttnn::Tensor& input,
                            const std::optional<MemoryConfig>& mem_config = std::nullopt) -> ttnn::Tensor {
        return ttnn::transpose(input, -2, -1, mem_config, 0.0f);
    };

    auto transpose_hc = [&](const ttnn::Tensor& input,
                            const std::optional<MemoryConfig>& mem_config = std::nullopt) -> ttnn::Tensor {
        return ttnn::transpose(input, 1, -2, mem_config, pad_value);
    };

    auto transpose_cn = [&](const ttnn::Tensor& input) -> ttnn::Tensor {
        return ttnn::transpose(input, 0, 1, output_mem_config, 0.0f);
    };

    // Sharded: decompose into transpose chains or fall back to prim::permute.
    if (a.is_sharded()) {
        if (N == 0 && C == 1 && H == 2 && W == 3) {
            output = formatted_input_tensor;
        } else if (N == 0 && C == 1 && H == 3 && W == 2) {
            output = transpose_wh(formatted_input_tensor, output_mem_config);
        } else if (N == 0 && C == 2 && H == 1 && W == 3) {
            output = transpose_hc(formatted_input_tensor, output_mem_config);
        } else if (N == 0 && C == 2 && H == 3 && W == 1) {
            output = transpose_wh(transpose_hc(formatted_input_tensor), output_mem_config);
        } else if (N == 0 && C == 3 && H == 1 && W == 2) {
            output = transpose_hc(transpose_wh(formatted_input_tensor), output_mem_config);
        } else if (N == 0 && C == 3 && H == 2 && W == 1) {
            output = transpose_wh(transpose_hc(transpose_wh(formatted_input_tensor)), output_mem_config);
        } else if (N == 1 && C == 0 && H == 2 && W == 3) {
            output = transpose_cn(formatted_input_tensor);
        } else {
            // Preserve input's shard layout; compute_output_specs derives a valid spec.
            auto effective_config = output_mem_config.value_or(
                MemoryConfig(a.memory_config().memory_layout(), a.memory_config().buffer_type()));
            output = ttnn::prim::permute(formatted_input_tensor, dims, effective_config, std::nullopt, pad_value);
        }
    } else {
        if (N == 0 && C == 1 && H == 2 && W == 3) {
            output = formatted_input_tensor;
        } else if (N == 0 && C == 1 && H == 3 && W == 2) {
            output = transpose_wh(formatted_input_tensor, output_mem_config);
        } else if (N == 0 && C == 2 && H == 1 && W == 3) {
            output = transpose_hc(formatted_input_tensor, output_mem_config);
        } else if (N == 1 && C == 0 && H == 2 && W == 3) {
            output = transpose_cn(formatted_input_tensor);
        } else {
            output = prim_permute(formatted_input_tensor);
        }
    }
    // Convert back to original dtype if typecast was performed.
    output = typecast ? ttnn::typecast(output, DataType::BFLOAT8_B) : output;
    return output;
}

ttnn::Tensor permute_launch(
    const ttnn::Tensor& a,
    const ttsl::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& output_mem_config,
    float pad_value = 0.0f) {
    return permute_impl(a, dims, output_mem_config, pad_value);
}

bool is_permute_nop(const ttnn::Tensor& a, const ttsl::SmallVector<uint32_t>& dims) {
    // Rank <= 1 is always a no-op.
    const auto rank = a.logical_shape().rank();
    if (rank <= 1) {
        return true;
    }

    // Identity permutation.
    ttsl::SmallVector<uint32_t> seq_dims(rank);
    std::iota(seq_dims.begin(), seq_dims.end(), 0);
    if (dims == seq_dims) {
        return true;
    }

    // TILE: last two dims must stay; RM: last dim must stay.
    if ((a.layout() == Layout::TILE && (dims[rank - 1] != rank - 1 || dims[rank - 2] != rank - 2)) ||
        (a.layout() == Layout::ROW_MAJOR && dims[rank - 1] != rank - 1)) {
        return false;
    }

    // Shape change → not a no-op.
    const auto& shape = a.logical_shape();
    ttsl::SmallVector<uint32_t> perm_shape(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        perm_shape[i] = shape[dims[i]];
    }

    if (perm_shape != shape) {
        return false;
    }

    // Moving a dim with size > 1 changes physical layout.
    for (uint32_t i = 0; i < rank; ++i) {
        const uint32_t j = dims[i];
        if (i != j && shape[i] > 1) {
            return false;
        }
    }

    return true;
}

// A permutation is a layout no-op (safe to route through reshape instead of an actual data
// move) iff the size>1 axes keep their relative order in the output. This must be checked on
// axis INDICES, not on the sizes themselves: comparing only the size>1 *values* would
// false-positive whenever two or more size>1 axes share the same extent (e.g. permuting
// [0, 3, 1, 2] on shape (1, 64, 64, 64)) — the size lists would match even though the axes were
// genuinely swapped, silently routing a real transpose through a zero-copy reshape and
// producing wrong data.
bool is_permute_layout_nop(const ttnn::Tensor& a, const ttsl::SmallVector<uint32_t>& dims) {
    // Only ROW_MAJOR has a physical layout where this axis-order check is valid; TILE's
    // tiled last-two-dims storage can make an axis reorder non-trivial even when it satisfies
    // the check below, so it must never reach this shortcut.
    if (a.layout() != Layout::ROW_MAJOR) {
        return false;
    }
    if (a.is_sharded()) {
        return false;
    }

    const auto& shape = a.logical_shape();
    bool has_previous_axis = false;
    uint32_t previous_axis = 0;
    for (const uint32_t axis : dims) {
        if (shape[axis] <= 1) {
            continue;
        }
        if (has_previous_axis && axis < previous_axis) {
            return false;
        }
        previous_axis = axis;
        has_previous_axis = true;
    }
    return true;
}

}  // namespace ttnn::operations::data_movement::detail

namespace ttnn {

ttnn::Tensor permute(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    float pad_value,
    const std::string& implementation) {
    const auto input_rank = input_tensor.logical_shape().rank();
    TT_FATAL(
        input_rank == dims.size(),
        "The number of dimensions in the tensor input does not match the length of the desired ordering");
    TT_FATAL(is_device_tensor(input_tensor), "Tensor must already be on device");

    ttsl::SmallVector<uint32_t> normalized_dims(dims.size());
    std::transform(dims.begin(), dims.end(), normalized_dims.begin(), [input_tensor](std::int64_t idx) {
        return input_tensor.logical_shape().get_normalized_index(idx);
    });

    namespace permute_codegen = operations::data_movement::permute_codegen;
    const auto selector = permute_codegen::parse_implementation(implementation);

    if (operations::data_movement::detail::is_permute_nop(input_tensor, normalized_dims)) {
        return ttnn::to_memory_config(input_tensor, memory_config.value_or(input_tensor.memory_config()));
    }

    if (operations::data_movement::detail::is_permute_layout_nop(input_tensor, normalized_dims)) {
        const auto& input_shape = input_tensor.logical_shape();
        ttsl::SmallVector<uint32_t> output_shape(normalized_dims.size());
        std::transform(
            normalized_dims.begin(), normalized_dims.end(), output_shape.begin(), [&input_shape](uint32_t dim) {
                return input_shape[dim];
            });
        return ttnn::reshape(input_tensor, ttnn::Shape(std::move(output_shape)), memory_config);
    }

    const auto& output_memory_config = memory_config.value_or(input_tensor.memory_config());
    if (selector == permute_codegen::ImplementationSelector::Codegen) {
        TT_FATAL(
            permute_codegen::supported_by_codegen(input_tensor, normalized_dims, output_memory_config),
            "ttnn::permute: implementation=\"codegen\" requested but this input is not supported by the "
            "codegen implementation");
        return ttnn::prim::permute_codegen(input_tensor, normalized_dims, memory_config, std::nullopt);
    }
    if (selector == permute_codegen::ImplementationSelector::Auto &&
        permute_codegen::supported_by_codegen(input_tensor, normalized_dims, output_memory_config) &&
        !permute_codegen::is_demoted(input_tensor, normalized_dims)) {
        return ttnn::prim::permute_codegen(input_tensor, normalized_dims, memory_config, std::nullopt);
    }

    auto adjust_order = [](const ttsl::SmallVector<uint32_t>& dims) {
        ttsl::SmallVector<uint32_t> new_order;
        TT_FATAL(dims.size() <= 4, "Minimum rank of tensor required is 4");
        int additional_ranks = 4 - dims.size();
        for (int i = 0; i < additional_ranks; i++) {
            new_order.push_back(i);
        }
        for (unsigned int dim : dims) {
            new_order.push_back(dim + additional_ranks);
        }
        return new_order;
    };
    auto itensor = (input_tensor.logical_shape().rank() < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
    auto iorder = normalized_dims.size() < 4 ? adjust_order(normalized_dims) : normalized_dims;

    const auto input_layout = input_tensor.layout();

    if (input_layout == Layout::ROW_MAJOR && memory_config.has_value() && memory_config->is_sharded() &&
        memory_config->shard_spec().has_value()) {
        uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
        TT_FATAL(
            memory_config->shard_spec()->shape[1] * input_tensor.element_size() % (l1_alignment) == 0,
            "Shard page size must be aligned to {}B for L1 Tensor",
            l1_alignment);
    }
    auto output_tensor = operations::data_movement::detail::permute_launch(itensor, iorder, memory_config, pad_value);
    output_tensor = ttnn::to_layout(output_tensor, input_layout);

    if (input_rank < 4) {
        output_tensor = ttnn::squeeze_from_4D(output_tensor, input_rank);
    }

    return output_tensor;
}

ttnn::Tensor permute(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& dims,
    float pad_value,
    const std::string& implementation) {
    return permute(input_tensor, dims, std::nullopt, pad_value, implementation);
}

}  // namespace ttnn
