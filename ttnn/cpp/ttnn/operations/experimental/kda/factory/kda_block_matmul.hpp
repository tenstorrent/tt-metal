// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

// Programs the KDA ops run alone or on cores another program of the same dispatch leaves idle.
namespace ttnn::experimental::prim::kda_factory_detail {

// a @ w into output's tile columns from output_column, on the rectangle `cores` of every device. Each core owns a tile
// block of the output and streams K in k_block-tile blocks: the first core of each row reads that row's A blocks and
// multicasts them along the row (NoC 1), the first core of each column multicasts W down the column (NoC 0). Compute
// accumulates the K blocks in an FP32 buffer with packer L1 accumulation; the last K block reloads it into DST (exact)
// and packs the output dtype subblock by subblock, each written as soon as it is packed.
tt::tt_metal::ProgramDescriptor block_matmul_program(
    const tt::tt_metal::distributed::MeshDevice& device,
    const Tensor& a,
    const Tensor& w,
    const Tensor& output,
    const tt::tt_metal::CoreRange& cores,
    uint32_t k_block,
    uint32_t output_column,
    const DeviceComputeKernelConfig& compute_kernel_config);

void check_block_matmul(
    const Tensor& a, const Tensor& w, const tt::tt_metal::CoreRange& cores, uint32_t k_block, std::string_view name);

// Copy `columns` tile columns of `source` from source_column into destination's first columns, split by tile rows over
// `cores`. Both are TILE tensors of one dtype.
tt::tt_metal::ProgramDescriptor copy_tile_columns_program(
    const Tensor& source,
    uint32_t source_column,
    const Tensor& destination,
    uint32_t columns,
    const std::vector<tt::tt_metal::CoreCoord>& cores);

}  // namespace ttnn::experimental::prim::kda_factory_detail
