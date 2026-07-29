// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

void CombineFabric2dDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(args.device != nullptr, "combine_fabric2d requires a mesh device in attributes");
    TT_FATAL(args.num_links >= 1 && args.num_links <= 4, "num_links must be between 1 and 4 (got {})", args.num_links);
    TT_FATAL(args.num_tokens >= 1, "num_tokens must be >= 1 (got {})", args.num_tokens);
    TT_FATAL(args.num_slots >= 1, "num_slots must be >= 1 (got {})", args.num_slots);
    if (tensor_args.input.has_value()) {
        const auto& in = *tensor_args.input;
        // Precooked tokens are only observable if the tokens actually land somewhere host-readable, i.e.
        // in one of the DRAM modes. In the L1-only baseline the receiver NOP-acks and drops them.
        TT_FATAL(
            (args.variant & (32u | 64u)) != 0,
            "combine_fabric2d: a precooked input tensor only has an observable effect in a DRAM mode "
            "(variant bit5 DRAM_DIRECT or bit6 DRAM_DRAIN); got variant {}",
            args.variant);
        TT_FATAL(in.storage_type() == tt::tt_metal::StorageType::DEVICE, "combine_fabric2d: input must be on device");
        TT_FATAL(
            in.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM &&
                in.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "combine_fabric2d: input must be an interleaved DRAM tensor");
        TT_FATAL(
            in.dtype() == tt::tt_metal::DataType::UINT32,
            "combine_fabric2d: input must be UINT32 (the op moves raw bytes; {} would only confuse the check)",
            in.dtype());
        TT_FATAL(
            in.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "combine_fabric2d: input must be ROW_MAJOR so one row is exactly one page (= one token)");
        // Page-size and page-count checks need the buffer and live in the program factory, where the
        // output buffer they are compared against exists.
    }
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
    tt::tt_fabric::Topology topology,
    const std::optional<ttnn::Tensor>& input) {
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
        OperationType::tensor_args_t{.input = input});
}
}  // namespace ttnn::prim
