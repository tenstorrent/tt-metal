// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flash_mqa_decode.hpp"

#include "device/flash_mqa_decode_device_operation.hpp"

namespace ttnn::experimental::deepseek {

Tensor flash_mqa_decode(
    const Tensor& q,
    const Tensor& kv,
    const Tensor& lengths,
    const std::optional<Tensor>& selected_kv,
    uint32_t selected_row_offset,
    uint32_t ring_rows,
    const std::optional<Tensor>& attention_sink,
    std::optional<float> scale,
    uint32_t k_chunk_size,
    uint32_t num_s_blocks,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::flash_mqa_decode(
        q,
        kv,
        lengths,
        selected_kv,
        selected_row_offset,
        ring_rows,
        attention_sink,
        scale,
        k_chunk_size,
        num_s_blocks,
        memory_config,
        compute_kernel_config);
}

}  // namespace ttnn::experimental::deepseek
