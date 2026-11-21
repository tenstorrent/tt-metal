// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_device_operation.hpp"

#include <bit>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

namespace {

// Data types whose pages are plain element arrays, so a DM thread can reorder them element by element.
bool is_plain_element_dtype(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::UINT16:
        case DataType::UINT8:
        case DataType::INT8: return true;
        default: return false;
    }
}

uint32_t pad_word_of(DataType dtype, float pad_value) {
    switch (dtype) {
        case DataType::BFLOAT16: return std::bit_cast<uint32_t>(pad_value) >> 16;
        case DataType::FLOAT32: return std::bit_cast<uint32_t>(pad_value);
        default: return static_cast<uint32_t>(static_cast<int32_t>(pad_value));
    }
}

// The shape the plan works on: the logical shape, except that a rank 1 TILE tensor carries an implicit
// tile-height dimension ([W] is stored as one row of tiles).
std::vector<uint32_t> plan_shape(const Tensor& input) {
    const auto& logical = input.logical_shape();
    std::vector<uint32_t> shape(logical.cbegin(), logical.cend());
    if (input.layout() == Layout::TILE && shape.size() == 1) {
        shape.insert(shape.begin(), 1);
    }
    return shape;
}

std::vector<uint32_t> plan_dims(const PermuteParams& params, const Tensor& input) {
    std::vector<uint32_t> dims(params.dims.begin(), params.dims.end());
    if (input.layout() == Layout::TILE && dims.size() == 1) {
        dims = {0, 1};
    }
    return dims;
}

bool is_minimally_padded(const Tensor& input) {
    const auto shape = plan_shape(input);
    const auto& padded = input.padded_shape();
    if (padded.rank() != shape.size()) {
        return false;
    }
    const bool tile = input.layout() == Layout::TILE;
    for (size_t d = 0; d < shape.size(); ++d) {
        const bool tile_dim = tile && d + 2 >= shape.size();
        const uint32_t expected = tile_dim ? tt::round_up(shape[d], tt::constants::TILE_HEIGHT) : shape[d];
        if (padded[d] != expected) {
            return false;
        }
    }
    return true;
}

}  // namespace

permute_plan::BlockPlan permute_block_plan(const PermuteParams& params, const Tensor& input, uint32_t target_blocks) {
    const bool tile = input.layout() == Layout::TILE;
    return permute_plan::build_block_plan(permute_plan::PlanRequest{
        .shape = plan_shape(input),
        .dims = plan_dims(params, input),
        .tile = tile,
        // Block-format tiles have no per-element size; their pages only ever move whole.
        .elem_size = tile && !is_plain_element_dtype(input.dtype()) ? 0 : input.element_size(),
        .tile_page_bytes = tile ? static_cast<uint32_t>(input.tensor_spec().compute_page_size_bytes()) : 0,
        .pad_word = pad_word_of(input.dtype(), params.pad_value),
        .max_window_bytes = kPermuteMaxWindowBytes,
        .target_blocks = target_blocks,
    });
}

PermuteDeviceOperation::program_factory_t PermuteDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return PermuteProgramFactory{};
}

void PermuteDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& dims = operation_attributes.dims;
    const auto& shape = input.logical_shape();
    const size_t rank = shape.rank();

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Quasar permute: the input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Quasar permute: the input must be allocated in a device buffer");
    TT_FATAL(
        rank >= 1 && rank <= permute_plan::kMaxRank,
        "Quasar permute supports rank 1 to {}, got rank {}",
        permute_plan::kMaxRank,
        rank);
    TT_FATAL(dims.size() == rank, "Quasar permute: {} dims given for a rank {} tensor", dims.size(), rank);
    uint32_t seen = 0;
    for (const uint32_t dim : dims) {
        TT_FATAL(dim < rank, "Quasar permute: dim {} is out of range for rank {}", dim, rank);
        TT_FATAL((seen & (1u << dim)) == 0, "Quasar permute: dim {} is repeated in the permutation", dim);
        seen |= 1u << dim;
    }
    TT_FATAL(input.logical_volume() > 0, "Quasar permute does not support empty tensors (logical shape {})", shape);
    TT_FATAL(
        !input.is_sharded() && !operation_attributes.output_mem_config.is_sharded(),
        "Quasar permute supports interleaved tensors only (input {}, output {})",
        input.memory_config(),
        operation_attributes.output_mem_config);
    TT_FATAL(
        input.dtype() != DataType::BFLOAT8_B && input.dtype() != DataType::BFLOAT4_B,
        "Quasar permute: {} is not a Quasar data format",
        input.dtype());
    TT_FATAL(
        operation_attributes.num_threads >= 1 && operation_attributes.num_threads <= kPermuteMaxThreads,
        "Quasar permute: {} DM threads requested, a cluster offers 1 to {}",
        operation_attributes.num_threads,
        kPermuteMaxThreads);

    const bool tile = input.layout() == Layout::TILE;
    if (tile) {
        TT_FATAL(
            input.tensor_spec().tile() == Tile{},
            "Quasar permute requires standard 32x32 tiles, got {}x{}",
            input.tensor_spec().tile().get_height(),
            input.tensor_spec().tile().get_width());
    }
    TT_FATAL(
        is_minimally_padded(input),
        "Quasar permute does not support padding beyond the last logical {} (logical {}, padded {})",
        tile ? "tile" : "element",
        shape,
        input.padded_shape());

    if (!permute_plan::moves_whole_pages(plan_dims(operation_attributes, input), tile)) {
        // Pages are rebuilt element by element in L1: the page must be a plain element array.
        TT_FATAL(
            is_plain_element_dtype(input.dtype()),
            "Quasar permute: {} pages cannot be reordered element by element; only permutations that keep the "
            "{} in place are supported for this data type (dims {})",
            input.dtype(),
            tile ? "last two dimensions" : "last dimension",
            dims);
    }
    const auto plan = permute_block_plan(operation_attributes, input, 1);
    TT_FATAL(
        plan.in_window_bytes <= kPermuteMaxWindowBytes && plan.out_window_bytes <= kPermuteMaxWindowBytes,
        "Quasar permute: a block of this permutation needs {} + {} bytes of L1 per DM thread, the cap is {} each",
        plan.in_window_bytes,
        plan.out_window_bytes,
        kPermuteMaxWindowBytes);
}

TensorSpec PermuteDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& in_shape = input.logical_shape();
    ttsl::SmallVector<uint32_t> out_shape(operation_attributes.dims.size());
    for (size_t k = 0; k < out_shape.size(); ++k) {
        out_shape[k] = in_shape[operation_attributes.dims[k]];
    }
    return TensorSpec(
        Shape(std::move(out_shape)),
        TensorLayout(input.dtype(), PageConfig(input.layout()), operation_attributes.output_mem_config));
}

Tensor PermuteDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

Tensor permute(
    const Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& dims,
    const MemoryConfig& output_mem_config,
    float pad_value,
    uint32_t num_threads) {
    return ttnn::device_operation::launch<PermuteDeviceOperation>(
        PermuteParams{
            .dims = dims,
            .output_mem_config = output_mem_config,
            .pad_value = pad_value,
            .num_threads = num_threads,
        },
        PermuteInputs{.input = input_tensor});
}

}  // namespace ttnn::prim::qsr
