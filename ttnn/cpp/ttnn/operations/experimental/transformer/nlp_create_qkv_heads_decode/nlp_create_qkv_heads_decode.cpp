// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode.hpp"

#include <utility>
#include <tt-metalium/hal.hpp>
#include "device/nlp_create_qkv_heads_decode_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"

namespace ttnn::experimental {

namespace {

// The interleaved reader kernel for nlp_create_qkv_heads_decode reads each
// face row as a single 16-element transaction, i.e. `16 * element_size` bytes
// per noc_async_read. On Wormhole that is 32 bytes for bfloat16, which equals
// the 32-byte DRAM read alignment so the read is well-formed. On Blackhole
// the DRAM read alignment is 64 bytes, so a 32-byte transaction is
// sub-alignment and silently returns wrong data — every odd-indexed Q/K/V
// head ends up reading the *previous* user's row instead of its own (see
// issue #43270 for the gpt-oss-20b symptom and isolation diagnostics).
//
// Until the kernel is taught to read in alignment-compliant chunks, promote
// the input tensor to L1 (which has 16-byte read alignment regardless of
// arch) when the per-face-row read size would be below the DRAM alignment.
// On Wormhole the predicate is false (32 < 32 is false), so the WH path is
// untouched. On any architecture / dtype combination where the kernel's
// read size already meets DRAM alignment (e.g. fp32 on Blackhole — 64 bytes)
// the predicate is false and the original DRAM path is used.
bool needs_l1_promotion_for_dram_alignment(const ttnn::Tensor& input_tensor) {
    if (input_tensor.is_sharded()) {
        return false;  // sharded inputs don't go through the interleaved reader
    }
    if (input_tensor.memory_config().buffer_type() != tt::tt_metal::BufferType::DRAM) {
        return false;
    }
    const uint32_t face_row_read_bytes = 16u * input_tensor.element_size();
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    return face_row_read_bytes < dram_alignment;
}

}  // namespace

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> nlp_create_qkv_heads_decode(
    const Tensor& input_tensor_in,
    const uint32_t num_heads,
    const std::optional<const uint32_t> num_kv_heads,
    std::optional<std::array<Tensor, 3>>& /*optional_output_tensors*/,
    const std::optional<const bool> overlap_qk_coregrid,
    const std::optional<const Tensor>& batch_offset,
    const std::optional<const uint32_t> slice_size,
    const std::optional<MemoryConfig>& memory_config) {
    Tensor input_tensor = input_tensor_in;
    if (needs_l1_promotion_for_dram_alignment(input_tensor)) {
        const tt::tt_metal::MemoryConfig l1_interleaved{
            tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1};
        input_tensor = ttnn::to_memory_config(input_tensor, l1_interleaved, std::nullopt);
    }
    const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_heads);
    const bool overlap_qk_coregrid_val = input_tensor.is_sharded() ? overlap_qk_coregrid.value_or(true) : true;
    // Check if input is on subcoregrids
    // Conditions to check:
    // - input is sharded
    // - input is sharded on more than 1 grid range
    // - input is sharded on single grid range but does not start from 0,0
    const bool input_on_subcoregrids =
        input_tensor.is_sharded() &&
        (input_tensor.shard_spec().value().grid.ranges().size() > 1 ||
         input_tensor.shard_spec().value().grid.bounding_box().start_coord != CoreCoord{0, 0});

    CoreRangeSet output_core_grid;
    if (memory_config.has_value() and memory_config.value().shard_spec().has_value()) {
        output_core_grid = memory_config.value().shard_spec().value().grid;
    } else {
        const auto device_grid_size = input_tensor.device()->compute_with_storage_grid_size();
        output_core_grid =
            CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{device_grid_size.x - 1, device_grid_size.y - 1}}};
    }

    MemoryConfig output_mem_config{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec{output_core_grid, {}}};
    // Infer head_dim
    TT_FATAL(
        input_tensor.padded_shape()[3] % (num_heads + 2 * num_kv_heads_val) == 0,
        "Input shape {} must be divisible by num_heads + 2*num_kv_heads = {}",
        input_tensor.padded_shape()[3],
        num_heads + 2 * num_kv_heads_val);
    uint32_t head_dim = input_tensor.padded_shape()[3] / (num_heads + 2 * num_kv_heads_val);

    auto out = ttnn::prim::nlp_create_qkv_heads_decode(
        input_tensor,
        num_heads,
        num_kv_heads_val,
        head_dim,
        overlap_qk_coregrid_val,
        input_on_subcoregrids,
        batch_offset,
        slice_size,
        output_mem_config);
    return {out.at(0), out.at(1), out.at(2)};
}

}  // namespace ttnn::experimental
