// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/argmax_decode/device/argmax_decode_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b/argmax_decode/device/argmax_decode_table.hpp"

namespace ttnn::experimental::llama31_8b {

using namespace tt::tt_metal;

namespace {

bool generic_supported(const Tensor& input) {
    if (input.storage_type() != StorageType::DEVICE || input.dtype() != DataType::BFLOAT16 ||
        input.layout() != Layout::TILE) {
        return false;
    }
    const auto tile = input.tensor_spec().tile().get_tile_shape();
    if (tile[0] != tt::constants::TILE_HEIGHT || tile[1] != tt::constants::TILE_WIDTH) {
        return false;
    }
    const auto& mem = input.memory_config();
    if (mem.memory_layout() != TensorMemoryLayout::WIDTH_SHARDED || mem.buffer_type() != BufferType::L1 ||
        !mem.shard_spec().has_value()) {
        return false;
    }
    const auto& shard = mem.shard_spec().value();
    if (shard.shape[0] != tt::constants::TILE_HEIGHT || shard.shape[1] == 0 ||
        shard.shape[1] % tt::constants::TILE_WIDTH != 0) {
        return false;
    }
    const auto padded = input.padded_shape();
    const auto logical = input.logical_shape();
    if (padded.rank() < 2 || padded[-2] != tt::constants::TILE_HEIGHT || logical[-2] != tt::constants::TILE_HEIGHT ||
        logical[-1] == 0) {
        return false;
    }
    for (int i = 0; i < static_cast<int>(padded.rank()) - 2; ++i) {
        if (padded[i] != 1) {
            return false;
        }
    }
    return true;
}

}  // namespace

const argmax_decode_table::Shape* argmax_decode_shape(const Tensor& input) {
    if (!generic_supported(input) || input.device()->arch() != tt::ARCH::BLACKHOLE) {
        return nullptr;
    }
    const auto& shard = input.memory_config().shard_spec().value();
    if (shard.orientation != ShardOrientation::ROW_MAJOR) {
        return nullptr;
    }
    const uint32_t cols = input.logical_shape()[-1];
    for (const auto& s : argmax_decode_table::kShapes) {
        if (s.cols != cols || input.padded_shape()[-1] != cols || shard.shape[1] != s.shard_cols ||
            shard.grid.num_cores() != s.workers) {
            continue;
        }
        const auto grid = input.device()->compute_with_storage_grid_size();
        if (shard.grid != num_cores_to_corerangeset(s.workers, grid, true)) {
            continue;
        }
        return &s;
    }
    return nullptr;
}

namespace {

TensorSpec output_spec(const Tensor& input, const MemoryConfig& mem) {
    const auto in = input.logical_shape();
    ttsl::SmallVector<uint32_t> dims;
    for (int i = 0; i < static_cast<int>(in.rank()) - 1; ++i) {
        dims.push_back(in[i]);
    }
    return TensorSpec(ttnn::Shape(dims), TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), mem));
}

const char* output_problem(const Tensor& input, const MemoryConfig& mem, const std::optional<Tensor>& output_tensor) {
    if (mem.memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        return "memory_config must be interleaved";
    }
    if (!output_tensor.has_value()) {
        return nullptr;
    }
    const auto& out = output_tensor.value();
    if (out.storage_type() != StorageType::DEVICE || !out.is_allocated()) {
        return "output_tensor must be allocated on the device";
    }
    if (out.device() != input.device()) {
        return "output_tensor must be on the input's device";
    }
    if (out.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        return "output_tensor must be interleaved";
    }
    if (out.tensor_spec() != output_spec(input, out.memory_config())) {
        return "output_tensor must be uint32 ROW_MAJOR with the input shape minus the last dim";
    }
    return nullptr;
}

}  // namespace

bool argmax_decode_supported(
    const Tensor& input, const std::optional<MemoryConfig>& memory_config, const std::optional<Tensor>& output_tensor) {
    return argmax_decode_shape(input) != nullptr &&
           output_problem(input, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG), output_tensor) == nullptr;
}

void ArgmaxDecodeOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        argmax_decode_shape(tensor_args.input) != nullptr,
        "llama31_8b.argmax_decode: input shape and sharding are not in the op's table; use ttnn.argmax otherwise");
    const char* problem =
        output_problem(tensor_args.input, operation_attributes.output_mem_config, tensor_args.optional_output_tensor);
    TT_FATAL(problem == nullptr, "llama31_8b.argmax_decode: {}; use ttnn.argmax otherwise", problem ? problem : "");
}

void ArgmaxDecodeOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.input.is_allocated(), "llama31_8b.argmax_decode: input must be allocated on the device");
    if (tensor_args.optional_output_tensor.has_value()) {
        const auto& out = tensor_args.optional_output_tensor.value();
        TT_FATAL(
            out.is_allocated() && out.device() == tensor_args.input.device(),
            "llama31_8b.argmax_decode: output_tensor must be allocated on the input's device");
    }
}

ArgmaxDecodeOperation::spec_return_value_t ArgmaxDecodeOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }
    return output_spec(tensor_args.input, operation_attributes.output_mem_config);
}

ArgmaxDecodeOperation::tensor_return_value_t ArgmaxDecodeOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::experimental::llama31_8b

namespace ttnn::prim {
ttnn::Tensor llama31_8b_argmax_decode(
    const Tensor& input,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<Tensor>& optional_output_tensor) {
    using OperationType = ttnn::experimental::llama31_8b::ArgmaxDecodeOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.input = input, .optional_output_tensor = optional_output_tensor});
}
}  // namespace ttnn::prim
