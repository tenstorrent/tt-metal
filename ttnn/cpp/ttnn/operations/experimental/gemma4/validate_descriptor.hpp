// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "request_descriptor.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::gemma4 {

inline void validate_request_descriptors(const Tensor& descriptors, const Tensor& input, uint32_t request_index) {
    using namespace tt::tt_metal;
    TT_FATAL(
        descriptors.storage_type() == StorageType::DEVICE && descriptors.buffer() != nullptr &&
            descriptors.device() == input.device(),
        "Request descriptors must be allocated on the input device");
    TT_FATAL(
        descriptors.dtype() == DataType::UINT32 && descriptors.layout() == Layout::ROW_MAJOR &&
            descriptors.memory_config() == ttnn::DRAM_MEMORY_CONFIG,
        "Request descriptors must be interleaved UINT32 ROW_MAJOR DRAM tensors");
    const auto& shape = descriptors.logical_shape();
    TT_FATAL(
        shape.rank() == 2 && shape[1] == request::words && request_index < shape[0] &&
            descriptors.buffer()->page_size() == request::bytes,
        "Request descriptors must have shape [requests, 8] with 32-byte pages and a valid request_index");
    for (const auto& placement : descriptors.tensor_topology().placements()) {
        TT_FATAL(
            std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Replicate>(placement),
            "Request descriptors must be replicated across the mesh");
    }
}

}  // namespace ttnn::experimental::gemma4
