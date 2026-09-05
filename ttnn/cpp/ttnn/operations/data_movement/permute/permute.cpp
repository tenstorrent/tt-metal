// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute.hpp"
#include "permute_force.hpp"

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
            const auto& input_memory_config = a.memory_config();
            auto effective_config = output_mem_config.value_or(
                input_memory_config.created_with_nd_shard_spec()
                    ? MemoryConfig(input_memory_config.buffer_type(), input_memory_config.nd_shard_spec())
                    : MemoryConfig(input_memory_config.memory_layout(), input_memory_config.buffer_type()));
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
            const auto& input_memory_config = a.memory_config();
            auto effective_config = output_mem_config.value_or(
                input_memory_config.created_with_nd_shard_spec()
                    ? MemoryConfig(input_memory_config.buffer_type(), input_memory_config.nd_shard_spec())
                    : MemoryConfig(input_memory_config.memory_layout(), input_memory_config.buffer_type()));
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

// Asserts the preconditions every leg shares and resolves negative axes, so the routed entry and
// both forced entries reach the gate and the prims on identical arguments.
ttsl::SmallVector<uint32_t> normalize_permute_dims(
    const ttnn::Tensor& input_tensor, const ttsl::SmallVector<int64_t>& dims) {
    TT_FATAL(
        input_tensor.logical_shape().rank() == dims.size(),
        "The number of dimensions in the tensor input does not match the length of the desired ordering");
    TT_FATAL(is_device_tensor(input_tensor), "Tensor must already be on device");

    ttsl::SmallVector<uint32_t> normalized_dims(dims.size());
    std::transform(dims.begin(), dims.end(), normalized_dims.begin(), [&input_tensor](std::int64_t idx) {
        return input_tensor.logical_shape().get_normalized_index(idx);
    });
    return normalized_dims;
}

ttnn::Tensor permute_native(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& normalized_dims,
    const std::optional<MemoryConfig>& memory_config,
    float pad_value) {
    // A permutation that moves no bytes is answered from host state rather than dispatched. This
    // belongs to the native path: a forced-codegen call has to reach a program.
    if (is_permute_nop(input_tensor, normalized_dims)) {
        return ttnn::to_memory_config(input_tensor, memory_config.value_or(input_tensor.memory_config()));
    }

    const auto input_rank = input_tensor.logical_shape().rank();

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
    auto itensor = (input_rank < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
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
    auto output_tensor = permute_launch(itensor, iorder, memory_config, pad_value);
    output_tensor = ttnn::to_layout(output_tensor, input_layout);

    if (input_rank < 4) {
        output_tensor = ttnn::squeeze_from_4D(output_tensor, input_rank);
    }

    return output_tensor;
}

ttnn::Tensor permute_force_native(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    float pad_value) {
    return permute_native(input_tensor, normalize_permute_dims(input_tensor, dims), memory_config, pad_value);
}

ttnn::Tensor permute_force_codegen(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    float /*pad_value*/) {
    // pad_value is inert here: the supported scope is row-major, where a permutation introduces no
    // padding. The routed entry likewise passes it to the native leg only.
    const auto normalized_dims = normalize_permute_dims(input_tensor, dims);
    TT_FATAL(
        permute_codegen::supported_by_codegen(input_tensor, normalized_dims, memory_config),
        "permute_force_codegen invoked for a case the codegen path does not support (requires a "
        "device-resident, unsharded rank-2..{} bfloat16/float32/int32 ROW_MAJOR input, an "
        "interleaved output, a non-degenerate shape, a permutation that is not the transpose op's "
        "fused width-height case, and -- when the last axis stays put -- a stick narrow enough for "
        "{} circular-buffer slots in one core's L1). This entry never falls back to native, because "
        "a forced leg that quietly served native would make any comparison against native vacuous. "
        "Use ttnn::permute if you want the case routed.",
        PermuteCodegenDeviceOperation::kMaxDims,
        permute_codegen::kRmCbSlots);
    return ttnn::prim::permute_codegen(input_tensor, normalized_dims, memory_config, std::nullopt);
}

}  // namespace ttnn::operations::data_movement::detail

namespace ttnn {

ttnn::Tensor permute(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    float pad_value) {
    namespace detail = operations::data_movement::detail;
    namespace permute_codegen = operations::data_movement::permute_codegen;

    const auto normalized_dims = detail::normalize_permute_dims(input_tensor, dims);

    // The no-op shortcut is native's own answer, so a call it covers is left to native rather than
    // dispatched -- but it is not consulted as part of the support scope, so a forced-codegen call
    // still reaches a program.
    // is_demoted() reads only the rank and dims and is three integer operations; supported_by_codegen()
    // walks the shape and, on the row-invariant path, queries live L1 occupancy. Demotion rejects the
    // whole blocked-generic class, so asking the cheap predicate first keeps a demoted caller -- which
    // gets nothing from this path -- off the expensive one. Both are pure predicates over the same
    // inputs and neither depends on the other having run.
    if (!detail::is_permute_nop(input_tensor, normalized_dims) &&
        !permute_codegen::is_demoted(input_tensor, normalized_dims) &&
        permute_codegen::supported_by_codegen(input_tensor, normalized_dims, memory_config)) {
        return ttnn::prim::permute_codegen(input_tensor, normalized_dims, memory_config, std::nullopt);
    }
    return detail::permute_native(input_tensor, normalized_dims, memory_config, pad_value);
}

ttnn::Tensor permute(const ttnn::Tensor& input_tensor, const ttsl::SmallVector<int64_t>& dims, float pad_value) {
    return permute(input_tensor, dims, std::nullopt, pad_value);
}

}  // namespace ttnn
