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
    const operation_attributes_t& /*args*/, const tensor_args_t&) {
    // Dummy output — the receiver L1 ring lives at a fixed L1 address managed by the program factory
    // (uniform across the mesh), not in a tensor. This exists only to give the framework something to
    // place on every mesh coordinate so the op dispatches on all chips.
    auto output_shape = ttnn::Shape({1, 1});
    return TensorSpec(
        output_shape,
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
            .topology = topology},
        OperationType::tensor_args_t{});
}
}  // namespace ttnn::prim
