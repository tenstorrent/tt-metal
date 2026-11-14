// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>

#include <tt-logger/tt-logger.hpp>

#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "permute_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/transpose/device/transpose_utils.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::data_movement {
PermuteDeviceOperation::program_factory_t PermuteDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& dims = operation_attributes.dims;
    if (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR) {
        // Last dimension unchanged — rows can be copied without reordering elements.
        if (dims.back() == tensor_args.input_tensor.logical_shape().rank() - 1) {
            return MultiCoreRowInvariant{};
        }
        // Last dimension moved — need a blocked transpose in RM.
        return MultiCoreBlockedGeneric{};
    }
    // Tiled layout selection.
    uint32_t rank = tensor_args.input_tensor.logical_shape().rank();
    // Tile dims stay in the last two positions (identity or WH swap).
    if ((dims[rank - 1] == rank - 1 && dims[rank - 2] == rank - 2) ||
        (dims[rank - 1] == rank - 2 && dims[rank - 2] == rank - 1)) {
        return MultiCoreTileInvariant{};
    }
    // Only one tile dimension is moved out.
    if (dims[rank - 1] == rank - 1 || dims[rank - 1] == rank - 2) {
        return MultiCoreTileRowInvariant{};
    }
    // Both tile dimensions are moved.
    return MultiCoreTiledGeneric{};
}

void PermuteDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        attributes.dims.size() == tensor_args.input_tensor.logical_shape().rank(),
        "Permute dimensions must match input tensor rank");
}

void PermuteDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensor_args*/) {}

PermuteDeviceOperation::spec_return_value_t PermuteDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    auto input_shape = input_tensor.logical_shape();

    ttsl::SmallVector<uint32_t> output_shape_vec(attributes.dims.size());
    std::transform(attributes.dims.begin(), attributes.dims.end(), output_shape_vec.begin(), [&](auto dim) {
        return input_shape[dim];
    });
    auto output_shape = Shape(std::move(output_shape_vec));

    auto output_mem_config = attributes.output_mem_config;

    // ND-sharded fallback: permute is a dimension reorder, so the shard follows the
    // same permutation. Keep ND provenance instead of dropping into legacy synthesis,
    // otherwise the output loses created_with_nd_shard_spec for 4D / rank-greater-4.
    if (output_mem_config.created_with_nd_shard_spec() && output_mem_config.nd_shard_spec().has_value()) {
        auto nd_spec = *output_mem_config.nd_shard_spec();
        if (input_tensor.memory_config().created_with_nd_shard_spec() && input_tensor.nd_shard_spec().has_value()) {
            // An ND-sharded input keeps the existing derivation: the shard shape is reordered
            // by the same permutation as the tensor. The attributes cannot tell whether this
            // config was supplied by the caller or defaulted from the input, so both behave
            // alike here.
            ttsl::SmallVector<uint32_t> shard_shape_vec(attributes.dims.size());
            std::transform(attributes.dims.begin(), attributes.dims.end(), shard_shape_vec.begin(), [&](auto dim) {
                return input_tensor.nd_shard_spec()->shard_shape[dim];
            });
            nd_spec = nd_spec.with_shard_shape(Shape(std::move(shard_shape_vec)));
        }
        // A non-ND input cannot have produced an ND config by defaulting, so this one came from
        // the caller and is already expressed in the output frame: honor it verbatim.
        return tt::tt_metal::TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(),
                tt::tt_metal::PageConfig(input_tensor.layout()),
                MemoryConfig(output_mem_config.buffer_type(), nd_spec)));
    }

    // Derive shard_spec when sharded output lacks one.
    if (output_mem_config.is_sharded() && !output_mem_config.shard_spec().has_value()) {
        ttsl::SmallVector<uint32_t> output_padded_vec(output_shape.view().begin(), output_shape.view().end());
        if (input_tensor.layout() == Layout::TILE && output_shape.rank() >= 2) {
            const auto& tile = input_tensor.tensor_spec().tile();
            auto r = output_shape.rank();
            output_padded_vec[r - 2] = tt::round_up(output_padded_vec[r - 2], tile.get_height());
            output_padded_vec[r - 1] = tt::round_up(output_padded_vec[r - 1], tile.get_width());
        }
        auto output_padded_shape = Shape(std::move(output_padded_vec));

        // Adapt input shard spec only when input/output share the same shard layout.
        bool derived = false;
        if (input_tensor.is_sharded() && input_tensor.shard_spec().has_value() &&
            input_tensor.memory_config().memory_layout() == output_mem_config.memory_layout()) {
            const auto& from_shape = input_tensor.padded_shape();
            auto vol_except_last = [](const auto& shape) {
                return std::accumulate(shape.cbegin(), shape.cend() - 1, uint64_t{1}, std::multiplies<uint64_t>());
            };
            uint64_t from_vol = vol_except_last(from_shape);
            uint64_t to_vol = vol_except_last(output_padded_shape);
            uint64_t from_w = from_shape[-1], to_w = output_padded_shape[-1];
            auto shard = input_tensor.shard_spec()->shape;
            uint64_t h_num = static_cast<uint64_t>(shard[0]) * to_vol;
            uint64_t w_num = static_cast<uint64_t>(shard[1]) * to_w;
            if (from_vol > 0 && from_w > 0 && h_num % from_vol == 0 && w_num % from_w == 0) {
                auto adjusted =
                    transpose::adjust_shard_spec_to_shape(*input_tensor.shard_spec(), from_shape, output_padded_shape);
                if (adjusted.has_value()) {
                    const bool tile_layout = input_tensor.layout() == Layout::TILE;
                    const bool tile_aligned = adjusted->shape[0] % tt::constants::TILE_HEIGHT == 0 &&
                                              adjusted->shape[1] % tt::constants::TILE_WIDTH == 0;
                    if (!tile_layout || tile_aligned) {
                        output_mem_config = MemoryConfig(
                            output_mem_config.memory_layout(), output_mem_config.buffer_type(), std::move(adjusted));
                        derived = true;
                    }
                }
            }
        }
        if (!derived) {
            // Generate a fresh shard spec for the permuted shape.
            auto shard_spec = transpose::generate_transpose_shard_spec(
                input_tensor, output_padded_shape, output_mem_config.memory_layout());
            output_mem_config =
                MemoryConfig(output_mem_config.memory_layout(), output_mem_config.buffer_type(), shard_spec);
        }
    }

    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), output_mem_config));
}

tt::tt_metal::operation::OpPerformanceModelGeneral<PermuteDeviceOperation::tensor_return_value_t>
PermuteDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*op_attr*/, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input_tensor;
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output, false, 0, true);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

uint32_t permute_row_invariant_rows_per_batch(const Tensor& input_tensor, const tt::tt_metal::TensorSpec& output_spec) {
    constexpr uint32_t kMaxRows = 256;
    constexpr uint32_t kMinRows = 32;
    constexpr uint32_t kScratchBytes = 16 * 1024;
    const auto& shape = input_tensor.logical_shape();
    if (input_tensor.device()->arch() != tt::ARCH::WORMHOLE_B0 || input_tensor.layout() != Layout::ROW_MAJOR ||
        shape.rank() < 2 || input_tensor.is_sharded() || output_spec.memory_config().is_sharded()) {
        return 0;
    }
    const auto grid = input_tensor.device()->compute_with_storage_grid_size();
    if (shape[-1] == 0 || shape.volume() / shape[-1] < 2ull * grid.x * grid.y) {
        return 0;
    }
    auto row_bytes = [&](tt::tt_metal::BufferType buffer_type) {
        const uint32_t alignment = buffer_type == tt::tt_metal::BufferType::DRAM
                                       ? tt::tt_metal::hal::get_dram_alignment()
                                       : tt::tt_metal::hal::get_l1_alignment();
        return tt::round_up(shape[-1] * input_tensor.element_size(), alignment);
    };
    const uint32_t page_size = row_bytes(input_tensor.buffer()->buffer_type());
    if (page_size != row_bytes(output_spec.memory_config().buffer_type())) {
        return 0;
    }
    const uint32_t rows = std::min(kMaxRows, kScratchBytes / page_size);
    if (rows < kMinRows || 2ull * rows * page_size > get_max_l1_space(input_tensor)) {
        return 0;
    }
    return rows;
}

ttsl::hash::hash_t PermuteDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    uint32_t rows_per_batch = 0;
    if (input_tensor.device()->arch() == tt::ARCH::WORMHOLE_B0 && input_tensor.layout() == Layout::ROW_MAJOR &&
        !input_tensor.is_sharded() && operation_attributes.dims.back() == input_tensor.logical_shape().rank() - 1) {
        rows_per_batch =
            permute_row_invariant_rows_per_batch(input_tensor, compute_output_specs(operation_attributes, tensor_args));
    }
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<PermuteDeviceOperation>, operation_attributes, tensor_args, rows_per_batch);
}

PermuteDeviceOperation::tensor_return_value_t PermuteDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::PermuteDeviceOperation::tensor_return_value_t permute(
    const Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    float pad_value) {
    using OperationType = ttnn::operations::data_movement::PermuteDeviceOperation;

    // BH 8-bit int RM pack_untilize drops the top 24 bits of each 32-bit Dest lane (#61827).
    // Workaround: stage via UInt32 in DRAM, permute, cast back. Lives in prim (not user API)
    // because fold/transpose/quasar call prim::permute directly; a user-API-only workaround
    // would need to be duplicated at every call site. Delete this block once #61827 lands.
    if ((input_tensor.dtype() == DataType::UINT8 || input_tensor.dtype() == DataType::INT8) &&
        input_tensor.layout() == Layout::ROW_MAJOR && input_tensor.device() != nullptr &&
        input_tensor.device()->arch() == tt::ARCH::BLACKHOLE && !dims.empty() &&
        dims.back() != static_cast<uint32_t>(dims.size() - 1)) {
        const auto src_dtype = input_tensor.dtype();
        const auto& input_mc = input_tensor.memory_config();
        const auto final_memory_config = memory_config.value_or(input_mc);
        // A sharded preallocated output tensor would be silently dropped by the final cast;
        // reject it so callers do not get a newly-allocated tensor they did not ask for.
        TT_FATAL(
            !(optional_output_tensor.has_value() && final_memory_config.is_sharded()),
            "prim::permute BH 8-bit RM workaround cannot honor a sharded optional_output_tensor; "
            "pass interleaved or omit it (see #61827).");
        const auto interleaved_l1 = tt::tt_metal::MemoryConfig(
            tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);
        auto staged = input_tensor.is_sharded() ? ttnn::to_memory_config(input_tensor, interleaved_l1, std::nullopt)
                                                : input_tensor;
        auto as_u32 = ttnn::typecast(staged, DataType::UINT32, ttnn::DRAM_MEMORY_CONFIG);
        auto permuted_u32 = ttnn::device_operation::launch<OperationType>(
            OperationType::operation_attributes_t{
                .dims = dims, .output_mem_config = ttnn::DRAM_MEMORY_CONFIG, .pad_value = pad_value},
            OperationType::tensor_args_t{.input_tensor = as_u32, .optional_output_tensor = std::nullopt});
        const auto cast_mc = final_memory_config.is_sharded() ? interleaved_l1 : final_memory_config;
        auto as_8bit = optional_output_tensor.has_value()
                           ? ttnn::typecast(permuted_u32, src_dtype, cast_mc, optional_output_tensor)
                           : ttnn::typecast(permuted_u32, src_dtype, cast_mc);
        return final_memory_config.is_sharded() ? ttnn::to_memory_config(as_8bit, final_memory_config, std::nullopt)
                                                : as_8bit;
    }

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dims = dims,
            .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
            .pad_value = pad_value},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor, .optional_output_tensor = std::move(optional_output_tensor)});
}
}  // namespace ttnn::prim
