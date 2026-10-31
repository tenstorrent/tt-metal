// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_indexed.hpp"

#include "device/rotary_embedding_indexed_device_operation.hpp"
#include "device/rope_chain_device_operation.hpp"

namespace ttnn::operations::bringup::rotary_embedding_indexed {

// Scalar form: no metadata tensor; kv_actual_global is a host scalar.
ttnn::Tensor rotary_embedding_indexed(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    uint32_t kv_actual_global,
    uint32_t cluster_axis,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<uint32_t>& seq_subshard_axis,
    const std::optional<uint32_t>& rotary_dim,
    uint32_t rotary_offset,
    const std::optional<ttnn::Tensor>& concat_prefix,
    bool uniform,
    const std::optional<ttnn::Tensor>& output_tensor,
    std::optional<CoreCoord> max_grid,
    uint32_t input_heads,
    const std::optional<tt::tt_metal::Layout>& output_layout,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    uint32_t group_heads) {
    const bool chain = input_heads > 0 || output_layout.has_value() || output_dtype.has_value() || group_heads > 1 ||
                       input.layout() == tt::tt_metal::Layout::ROW_MAJOR;
    if (chain) {  // bring-up chain mode (P.22): one fused program, device/rope_chain_device_operation.hpp
        TT_FATAL(
            kv_actual_global == 0 && !seq_subshard_axis.has_value() && !concat_prefix.has_value(),
            "rotary_embedding_indexed chain mode (input_heads / output_layout / output_dtype / group_heads / a "
            "ROW_MAJOR input) needs kv_actual_global 0, no seq_subshard_axis and no concat_prefix");
        TT_FATAL(rotary_dim.has_value(), "rotary_embedding_indexed chain mode needs rotary_dim");
        return ttnn::prim::bringup::rotary_embedding_indexed_chain(
            input,
            cos,
            sin,
            trans_mat,
            input_heads,
            *rotary_dim,
            rotary_offset,
            output_layout.value_or(tt::tt_metal::Layout::TILE),
            output_dtype.value_or(input.dtype()),
            group_heads,
            memory_config,
            compute_kernel_config,
            output_tensor,
            max_grid);
    }
    return ttnn::prim::bringup::rotary_embedding_indexed(
        input,
        cos,
        sin,
        trans_mat,
        /*metadata=*/std::nullopt,
        kv_actual_global,
        cluster_axis,
        memory_config,
        compute_kernel_config,
        seq_subshard_axis,
        rotary_dim,
        rotary_offset,
        concat_prefix,
        uniform,
        output_tensor,
        max_grid);
}

// Tensor form: kv_actual_global is a 1-element uint32 DRAM tensor read on-device (element [0]); the
// scalar attr is unused on this path.
ttnn::Tensor rotary_embedding_indexed(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    const ttnn::Tensor& kv_actual_global,
    uint32_t cluster_axis,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<uint32_t>& seq_subshard_axis,
    const std::optional<uint32_t>& rotary_dim,
    uint32_t rotary_offset,
    const std::optional<ttnn::Tensor>& concat_prefix) {
    return ttnn::prim::bringup::rotary_embedding_indexed(
        input,
        cos,
        sin,
        trans_mat,
        /*metadata=*/kv_actual_global,
        /*kv_actual_global=*/0,
        cluster_axis,
        memory_config,
        compute_kernel_config,
        seq_subshard_axis,
        rotary_dim,
        rotary_offset,
        concat_prefix);
}

}  // namespace ttnn::operations::bringup::rotary_embedding_indexed
