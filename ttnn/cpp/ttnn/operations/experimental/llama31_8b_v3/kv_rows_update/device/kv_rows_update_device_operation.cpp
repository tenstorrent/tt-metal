// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b_v3/kv_rows_update/device/kv_rows_update_device_operation.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b_v3/kv_rows_update/device/kv_rows_update_table.hpp"

namespace ttnn::experimental::llama31_8b_v3 {

using namespace tt::tt_metal;
namespace table = kv_rows_update_table;

namespace {

bool on_device(const Tensor& t) {
    return t.storage_type() == StorageType::DEVICE && t.is_allocated() && t.device()->arch() == tt::ARCH::BLACKHOLE;
}

bool dram_interleaved(const Tensor& t) {
    const auto& mem = t.memory_config();
    return mem.memory_layout() == TensorMemoryLayout::INTERLEAVED && mem.buffer_type() == BufferType::DRAM;
}

bool cache_ok(const Tensor& c) {
    if (!on_device(c) || c.dtype() != DataType::BFLOAT8_B || c.layout() != Layout::TILE || !dram_interleaved(c)) {
        return false;
    }
    const auto& s = c.padded_shape();
    return s.rank() == 4 && s[1] == table::kKvHeads && s[2] == table::kBlock && s[3] == table::kHeadDim &&
           c.logical_shape() == s;
}

bool heads_ok(const Tensor& t, uint32_t rows) {
    if (!on_device(t) || t.dtype() != DataType::BFLOAT16 || t.layout() != Layout::TILE) {
        return false;
    }
    const auto& s = t.padded_shape();
    if (s.rank() != 4 || s[0] != 1 || s[1] < rows || s[2] != tt::constants::TILE_HEIGHT || s[3] != table::kHeadDim) {
        return false;
    }
    const auto& mem = t.memory_config();
    if (mem.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED || mem.buffer_type() != BufferType::L1 ||
        !mem.shard_spec().has_value()) {
        return false;
    }
    const auto& shard = mem.shard_spec().value();
    return shard.orientation == ShardOrientation::ROW_MAJOR && shard.shape[0] == tt::constants::TILE_HEIGHT &&
           shard.shape[1] == table::kHeadDim;
}

bool int_row_ok(const Tensor& t, uint32_t width) {
    return on_device(t) && t.dtype() == DataType::INT32 && t.layout() == Layout::ROW_MAJOR && dram_interleaved(t) &&
           t.logical_shape()[-1] == width && t.padded_shape()[-1] == width;
}

}  // namespace

bool kv_rows_update_supported(
    const Tensor& keys,
    const Tensor& k,
    const Tensor& values,
    const Tensor& v,
    const std::vector<Tensor>& row_positions,
    const Tensor& page_table) {
    const auto rows = static_cast<uint32_t>(row_positions.size());
    if (std::find(table::kRows.begin(), table::kRows.end(), rows) == table::kRows.end()) {
        return false;
    }
    if (!cache_ok(keys) || !cache_ok(values) || keys.padded_shape() != values.padded_shape()) {
        return false;
    }
    if (!heads_ok(k, rows) || !heads_ok(v, rows) || k.padded_shape() != v.padded_shape()) {
        return false;
    }
    for (const auto& p : row_positions) {
        if (!int_row_ok(p, rows) || p.logical_volume() != rows || p.device() != keys.device() ||
            p.tensor_spec() != row_positions.front().tensor_spec()) {
            return false;
        }
    }
    if (!int_row_ok(page_table, table::kMaxBlocks)) {
        return false;
    }
    const auto grid = keys.device()->compute_with_storage_grid_size();
    return grid.x == table::kGridX && grid.y == table::kGridY && k.device() == keys.device() &&
           v.device() == keys.device() && values.device() == keys.device() && page_table.device() == keys.device();
}

void KvRowsUpdateOperation::validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t& a) {
    TT_FATAL(
        kv_rows_update_supported(a.keys, a.k, a.values, a.v, a.row_positions, a.page_table),
        "llama31_8b_v3.kv_rows_update: expects bfloat8_b [blocks, {}, {}, {}] DRAM caches, bfloat16 [1, >=rows, 32, "
        "{}] L1 height-sharded K / V heads, 1, 2, 4 or 8 int32 [rows] DRAM position rows, an int32 [*, {}] DRAM page "
        "table on the Blackhole {}x{} grid; use paged_fused_update_cache otherwise",
        table::kKvHeads,
        table::kBlock,
        table::kHeadDim,
        table::kHeadDim,
        table::kMaxBlocks,
        table::kGridX,
        table::kGridY);
}

void KvRowsUpdateOperation::validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t& a) {
    bool ok = a.keys.is_allocated() && a.values.is_allocated() && a.k.is_allocated() && a.v.is_allocated() &&
              a.page_table.is_allocated();
    for (const auto& p : a.row_positions) {
        ok = ok && p.is_allocated();
    }
    TT_FATAL(ok, "llama31_8b_v3.kv_rows_update: inputs must be allocated on the device");
}

KvRowsUpdateOperation::spec_return_value_t KvRowsUpdateOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& a) {
    return a.keys.tensor_spec();
}

KvRowsUpdateOperation::tensor_return_value_t KvRowsUpdateOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& a) {
    return a.keys;
}

}  // namespace ttnn::experimental::llama31_8b_v3

namespace ttnn::prim {
ttnn::Tensor llama31_8b_v3_kv_rows_update(
    const Tensor& keys,
    const Tensor& k,
    const Tensor& values,
    const Tensor& v,
    const std::vector<Tensor>& row_positions,
    const Tensor& page_table) {
    using OperationType = ttnn::experimental::llama31_8b_v3::KvRowsUpdateOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{},
        OperationType::tensor_args_t{
            .keys = keys, .k = k, .values = values, .v = v, .row_positions = row_positions, .page_table = page_table});
}
}  // namespace ttnn::prim
