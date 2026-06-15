// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode.hpp"

#include "device/matmul_decode_device_operation.hpp"

namespace ttnn {

Tensor matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded,
    std::optional<const DataType> dtype,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<uint32_t> out_subblock_h,
    std::optional<uint32_t> out_subblock_w,
    uint32_t in0_block_w,
    bool k_stream,
    uint32_t k_slice_tiles) {
    return ttnn::prim::matmul_decode(
        input_tensor_a,
        input_tensor_b,
        partial_width_sharded,
        dtype,
        compute_kernel_config,
        out_subblock_h,
        out_subblock_w,
        in0_block_w,
        k_stream,
        k_slice_tiles);
}

}  // namespace ttnn
