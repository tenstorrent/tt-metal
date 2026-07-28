// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

void CombineFabric2dDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t&) {
    TT_FATAL(args.device != nullptr, "combine_fabric2d requires a mesh device in attributes");
    TT_FATAL(args.num_links >= 1 && args.num_links <= 4, "num_links must be between 1 and 4 (got {})", args.num_links);
    TT_FATAL(args.num_tokens >= 1, "num_tokens must be >= 1 (got {})", args.num_tokens);
    TT_FATAL(args.num_slots >= 1, "num_slots must be >= 1 (got {})", args.num_slots);
}

void CombineFabric2dDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t&) {}

CombineFabric2dDeviceOperation::spec_return_value_t CombineFabric2dDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t&) {
    // Baseline (L1-only) path: the receiver ring lives at a fixed L1 address managed by the program
    // factory, not in a tensor. This dummy output only gives the framework something to place on every
    // mesh coordinate so the op dispatches on all chips.
    //
    // Phase 3 DRAM modes (variant bit5 DRAM_DIRECT / bit6 DRAM_DRAIN): this tensor IS the interleaved
    // DRAM landing buffer. It is sized to hold one page (= chunk_size_bytes) per token for every worker
    // this device can host (2 neighbors x num_links). Its base address is uniform across the mesh, so a
    // producer on chip A can address the same buffer on chip B by page index.
    const bool dram_mode = (args.variant & (32u | 64u)) != 0;
    if (!dram_mode) {
        return TensorSpec(
            ttnn::Shape({1, 1}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::UINT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig{
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
    }
    TT_FATAL(
        args.chunk_size_bytes % sizeof(uint32_t) == 0,
        "combine_fabric2d: chunk_size_bytes {} must be a multiple of 4 for the DRAM output buffer",
        args.chunk_size_bytes);
    const uint32_t max_workers = 2u * args.num_links;                      // 2 neighbors, num_links each
    const uint32_t pages = max_workers * args.num_tokens;                  // one page per token per worker
    const uint32_t page_elems = args.chunk_size_bytes / sizeof(uint32_t);  // row = one page = one token
    return TensorSpec(
        ttnn::Shape({pages, page_elems}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

CombineFabric2dDeviceOperation::tensor_return_value_t CombineFabric2dDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), args.device);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d

namespace ttnn::prim {
ttnn::Tensor combine_fabric2d(
    ttnn::MeshDevice* device,
    uint32_t num_links,
    uint32_t num_tokens,
    uint32_t chunk_size_bytes,
    uint32_t num_slots,
    uint32_t axis,
    uint32_t stall_telemetry,
    uint32_t variant,
    tt::tt_fabric::Topology topology) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .device = device,
            .num_links = num_links,
            .num_tokens = num_tokens,
            .chunk_size_bytes = chunk_size_bytes,
            .num_slots = num_slots,
            .axis = axis,
            .stall_telemetry = stall_telemetry,
            .variant = variant,
            .topology = topology},
        OperationType::tensor_args_t{});
}
}  // namespace ttnn::prim
