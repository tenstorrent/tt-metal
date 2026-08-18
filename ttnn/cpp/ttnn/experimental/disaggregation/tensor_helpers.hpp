// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental_disaggregation {

// Wrap raw bfp8-packed bytes (uint32-aligned, TILE layout) as a host-side
// ttnn::Tensor with the given shape — no quantization round-trip.
// Used to compare KV-table reads against the live KV cache byte-for-byte.
ttnn::Tensor tensor_from_bfp8_bytes(std::span<const uint8_t> raw_bytes, const std::vector<uint32_t>& shape);

// Wrap raw bfloat16 bytes (2 bytes/element, ROW_MAJOR layout) as a host-side
// ttnn::Tensor with the given shape — no quantization round-trip. The RM/bf16
// analogue of tensor_from_bfp8_bytes, for uncompressed (bf16 ROW_MAJOR) KV caches.
ttnn::Tensor tensor_from_bf16_bytes(std::span<const uint8_t> raw_bytes, const std::vector<uint32_t>& shape);

}  // namespace ttnn::experimental_disaggregation
