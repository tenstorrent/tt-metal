// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"

namespace gemma4_sliding {

// Lanes: reads the first N elements of a metadata tensor (one uint32 per lane in page 0) in one DMA; dst_l1_addr
// must hold N words. Same contract as trace_metadata::read_metadata_scalar_u32, its N = 1 case.
template <typename NocT, typename AccessorArgsT, size_t N>
inline void read_metadata_vector_u32(
    NocT& noc,
    const AccessorArgsT& accessor_args,
    uint32_t tensor_addr,
    uint32_t dst_l1_addr,
    std::array<uint32_t, N>& out) {
    const auto accessor = TensorAccessor(accessor_args, tensor_addr);
    noc.async_read(accessor, CoreLocalMem<uint8_t>(dst_l1_addr), N * sizeof(uint32_t), {.page_id = 0}, {});
    noc.async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t i = 0; i < N; ++i) {
        out[i] = CoreLocalMem<volatile uint32_t>(dst_l1_addr)[i];
    }
}

}  // namespace gemma4_sliding
