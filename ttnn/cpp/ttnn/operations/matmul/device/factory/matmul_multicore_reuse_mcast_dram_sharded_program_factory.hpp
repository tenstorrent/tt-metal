// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

// Multi-row DRAM-streamed matmul: MatmulMultiCoreReuseMultiCast1DProgramConfig::dram_stream_workers_per_bank > 0.
ttnn::device_operation::ProgramArtifacts create_program_dram_stream_mrows(
    tt::tt_metal::distributed::MeshDevice& device,
    const tt::tt_metal::MeshTensor& in0_tensor,
    const tt::tt_metal::MeshTensor& in1_tensor,
    const tt::tt_metal::MeshTensor& out_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t Mt,
    uint32_t Kt,
    uint32_t Nt,
    uint32_t k_block,
    uint32_t cores_per_bank,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat output_data_format,
    bool fuse_swiglu,
    uint32_t num_distributors,
    uint32_t x_chunk);

}  // namespace ttnn::prim
