// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b_v3/swiglu_decode/device/swiglu_decode_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b_v3/swiglu_decode/device/swiglu_decode_table.hpp"

namespace ttnn::experimental::llama31_8b_v3 {

using namespace tt::tt_metal;
namespace table = swiglu_decode_table;

namespace {

bool input_ok(const Tensor& t) {
    if (t.storage_type() != StorageType::DEVICE || !t.is_allocated() || t.dtype() != DataType::BFLOAT16 ||
        t.layout() != Layout::TILE || t.device()->arch() != tt::ARCH::BLACKHOLE) {
        return false;
    }
    const auto grid = t.device()->compute_with_storage_grid_size();
    if (grid.x != table::kGridX || grid.y != table::kGridY) {
        return false;
    }
    const auto tile = t.tensor_spec().tile().get_tile_shape();
    if (tile[0] != tt::constants::TILE_HEIGHT || tile[1] != tt::constants::TILE_WIDTH) {
        return false;
    }
    const ttnn::Shape want({1, 1, tt::constants::TILE_HEIGHT, table::kWidth});
    if (t.logical_shape() != want || t.padded_shape() != want) {
        return false;
    }
    const auto& mem = t.memory_config();
    if (mem.memory_layout() != TensorMemoryLayout::WIDTH_SHARDED || mem.buffer_type() != BufferType::L1 ||
        !mem.shard_spec().has_value()) {
        return false;
    }
    const auto& shard = mem.shard_spec().value();
    return shard.orientation == ShardOrientation::ROW_MAJOR && shard.shape[0] == tt::constants::TILE_HEIGHT &&
           shard.shape[1] == table::kSrcShardWidth &&
           shard.grid == num_cores_to_corerangeset(table::kSrcCores, grid, true);
}

}  // namespace

bool swiglu_decode_supported(const Tensor& gate, const Tensor& up) {
    return input_ok(gate) && input_ok(up) && gate.device() == up.device() && gate.memory_config() == up.memory_config();
}

void SwigluDecodeOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    TT_FATAL(
        swiglu_decode_supported(tensor_args.gate, tensor_args.up),
        "llama31_8b_v3.swiglu_decode: inputs must be two [1, 1, 32, {}] bfloat16 TILE tensors width sharded in L1 on "
        "the first {} cores of the Blackhole {}x{} grid (row major, {} wide); use ttnn.mul with a SILU activation "
        "otherwise",
        table::kWidth,
        table::kSrcCores,
        table::kGridX,
        table::kGridY,
        table::kSrcShardWidth);
}

void SwigluDecodeOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.gate.is_allocated() && tensor_args.up.is_allocated(),
        "llama31_8b_v3.swiglu_decode: inputs must be allocated on the device");
}

SwigluDecodeOperation::spec_return_value_t SwigluDecodeOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& gate = tensor_args.gate;
    return TensorSpec(
        gate.logical_shape(), TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE), gate.memory_config()));
}

SwigluDecodeOperation::tensor_return_value_t SwigluDecodeOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.gate.device());
}

}  // namespace ttnn::experimental::llama31_8b_v3

namespace ttnn::prim {
ttnn::Tensor llama31_8b_v3_swiglu_decode(const Tensor& gate, const Tensor& up) {
    using OperationType = ttnn::experimental::llama31_8b_v3::SwigluDecodeOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{}, OperationType::tensor_args_t{.gate = gate, .up = up});
}
}  // namespace ttnn::prim
