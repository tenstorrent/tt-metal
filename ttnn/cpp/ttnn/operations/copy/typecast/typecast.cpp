// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast.hpp"
#include "ttnn/operations/copy/typecast/device/typecast_device_op.hpp"
#include "ttnn/operations/core/core.hpp"  // for to_dtype, to_layout, to_memory_config
#include "ttnn/tensor/tensor_utils.hpp"   // for is_cpu_tensor

namespace ttnn::operations::copy::detail {

namespace {

// Check whether two memory configs have the same TensorMemoryLayout
// (e.g. both INTERLEAVED, or both HEIGHT_SHARDED, etc.)
bool memory_layouts_match(const MemoryConfig& a, const MemoryConfig& b) {
    return a.memory_layout() == b.memory_layout();
}

// Determine if input is in a block-float format that is tile-only (BFP8_B, BFP4_B).
bool is_tile_only_dtype(const DataType dtype) { return dtype == DataType::BFLOAT8_B || dtype == DataType::BFLOAT4_B; }

}  // namespace

inline Tensor typecast_impl(
    const Tensor& input_tensor,
    const DataType& output_dtype,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<Layout>& output_layout = std::nullopt) {
    // Handle host tensors
    if (is_cpu_tensor(input_tensor)) {
        TT_FATAL(
            !optional_output_tensor.has_value(),
            "Preallocated output tensor is not supported for host tensor typecast. "
            "Use to_dtype directly if you need this functionality.");
        TT_FATAL(
            !sub_core_grids.has_value(),
            "sub_core_grids is not supported for host tensor typecast (only applicable to device operations).");

        auto result = ttnn::to_dtype(input_tensor, output_dtype);

        // Apply layout change for host tensors if requested
        if (output_layout.has_value() && result.layout() != output_layout.value()) {
            result = ttnn::to_layout(result, output_layout.value());
        }
        return result;
    }

    // Device tensor path
    const auto target_layout = output_layout.value_or(input_tensor.layout());
    const auto target_mem_config = optional_output_tensor.has_value()
                                       ? optional_output_tensor.value().memory_config()
                                       : memory_config.value_or(input_tensor.memory_config());

    const bool needs_layout_change = (input_tensor.layout() != target_layout);
    const bool needs_memory_layout_change = !memory_layouts_match(input_tensor.memory_config(), target_mem_config);
    const bool needs_dtype_change = (input_tensor.dtype() != output_dtype);

    // Fast path: no layout or memory layout change needed - use prim::typecast directly
    if (!needs_layout_change && !needs_memory_layout_change) {
        if (!needs_dtype_change) {
            // Nothing to do at all, just return the input (or apply memory_config for buffer type change)
            if (input_tensor.memory_config() == target_mem_config) {
                return input_tensor;
            }
            return ttnn::to_memory_config(input_tensor, target_mem_config);
        }

        const auto input_dtype = input_tensor.dtype();
        const bool preserve_fp32_precision =
            (input_dtype == DataType::FLOAT32) or
            (output_dtype == DataType::UINT8 and
             (input_dtype == DataType::BFLOAT16 or input_dtype == DataType::BFLOAT8_B or
              input_dtype == DataType::BFLOAT4_B)) or
            (input_dtype == DataType::UINT16 and output_dtype == DataType::UINT8) or
            (input_dtype == DataType::UINT8 and output_dtype != DataType::BFLOAT16);
        const bool fp32_dest_acc_en = preserve_fp32_precision or output_dtype == DataType::UINT32 or
                                      output_dtype == DataType::INT32 or output_dtype == DataType::FLOAT32 or
                                      input_dtype == DataType::UINT32 or input_dtype == DataType::INT32;
        const bool bfp8_pack_precise = (output_dtype == DataType::BFLOAT8_B);
        const auto output_memory_config = target_mem_config;

        return ttnn::prim::typecast(
            input_tensor,
            output_dtype,
            output_memory_config,
            fp32_dest_acc_en,
            preserve_fp32_precision,
            bfp8_pack_precise,
            optional_output_tensor,
            sub_core_grids);
    }

    // Slow path: need layout and/or memory layout changes.
    // Preallocated output is not supported in this path since we compose multiple ops.
    TT_FATAL(
        !optional_output_tensor.has_value(),
        "Preallocated output tensor is not supported when typecast requires layout or memory layout transformation.");
    TT_FATAL(
        !sub_core_grids.has_value(),
        "sub_core_grids is not supported when typecast requires layout or memory layout transformation.");

    auto working = input_tensor;

    // Step 1: If memory layout differs and input is sharded, deshard to interleaved first.
    // This gives us a common ground for subsequent layout/dtype operations.
    if (needs_memory_layout_change && working.is_sharded()) {
        const auto interleaved_config =
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, working.memory_config().buffer_type()};
        working = ttnn::to_memory_config(working, interleaved_config);
    }

    // Step 2: Change layout if needed.
    // For BFP types: typecast must happen in TILE layout, so we sequence carefully:
    //   - If input is BFP and target is RM: typecast first (in TILE), then change layout
    //   - If input is non-BFP RM and target is BFP TILE: change to TILE first, then typecast
    //   - Otherwise: change layout, then typecast

    const bool input_is_bfp = is_tile_only_dtype(working.dtype());
    const bool output_is_bfp = is_tile_only_dtype(output_dtype);

    if (needs_layout_change && needs_dtype_change) {
        if (input_is_bfp && target_layout == Layout::ROW_MAJOR) {
            // BFP TILE -> non-BFP RM: typecast first (stays in TILE), then to_layout
            if (needs_dtype_change) {
                working = typecast_impl(working, output_dtype);
            }
            working = ttnn::to_layout(working, target_layout);
        } else if (output_is_bfp && working.layout() == Layout::ROW_MAJOR) {
            // non-BFP RM -> BFP TILE: to_layout first (go to TILE), then typecast
            working = ttnn::to_layout(working, target_layout);
            working = typecast_impl(working, output_dtype);
        } else {
            // General case: change layout first, then typecast
            working = ttnn::to_layout(working, target_layout);
            working = typecast_impl(working, output_dtype);
        }
    } else if (needs_layout_change) {
        // Only layout change, no dtype change
        working = ttnn::to_layout(working, target_layout);
    } else if (needs_dtype_change) {
        // Only dtype change (but memory layout differs - handled by recursive call which hits fast path
        // since we already desharded above)
        working = typecast_impl(working, output_dtype);
    }

    // Step 3: Apply target memory config if it doesn't match yet.
    if (working.memory_config() != target_mem_config) {
        working = ttnn::to_memory_config(working, target_mem_config);
    }

    return working;
}

}  // namespace ttnn::operations::copy::detail

namespace ttnn {

Tensor typecast(
    const Tensor& input,
    const DataType& output_dtype,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<Layout>& output_layout) {
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            output_dtype == optional_output_tensor.value().dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }

    return operations::copy::detail::typecast_impl(
        input, output_dtype, memory_config_arg, optional_output_tensor, sub_core_grids, output_layout);
}

Tensor typecast(
    const Tensor& input_tensor,
    const DataType& tt_input_dtype,
    const DataType& tt_output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<Layout>& output_layout) {
    TT_FATAL(tt_input_dtype == input_tensor.dtype(), "input dtype and input tensor's dtype provided should match");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            tt_output_dtype == optional_output_tensor.value().dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }
    return operations::copy::detail::typecast_impl(
        input_tensor, tt_output_dtype, memory_config, optional_output_tensor, sub_core_grids, output_layout);
}

}  // namespace ttnn
