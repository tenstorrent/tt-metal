// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include <tt_stl/assert.hpp>

#include "tt-metalium/allocator.hpp"
#include "tt-metalium/mesh_device.hpp"

namespace ttnn::prim::detail {

inline constexpr uint32_t kWaveletL1BudgetQuantumBytes = 16U * 1024U;

[[nodiscard]] inline uint32_t quantized_available_l1_bytes(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    TT_FATAL(mesh_device != nullptr, "Wavelet operations require device tensors");
    const uint64_t base = mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t frontier =
        mesh_device->lowest_occupied_compute_l1_address().value_or(mesh_device->l1_size_per_core());
    TT_FATAL(
        frontier >= base, "Wavelet allocator reports occupied L1 frontier {} below unreserved base {}", frontier, base);
    const uint64_t available = frontier - base;
    TT_FATAL(
        available <= std::numeric_limits<uint32_t>::max(),
        "Wavelet available static L1 size {} exceeds uint32_t",
        available);
    return static_cast<uint32_t>(available / kWaveletL1BudgetQuantumBytes) * kWaveletL1BudgetQuantumBytes;
}

}  // namespace ttnn::prim::detail
