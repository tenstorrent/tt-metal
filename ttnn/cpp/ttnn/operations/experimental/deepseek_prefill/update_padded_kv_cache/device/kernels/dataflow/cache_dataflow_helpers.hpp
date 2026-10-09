// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"

namespace kv_cache_dataflow {

// Caller owns the scratch CB reservation and lifetime; successive metadata reads reuse its first word.
template <uint32_t AccessorOffset>
inline uint32_t read_metadata(Noc& noc, const CircularBuffer& scratch, uint32_t address) {
    constexpr auto args = TensorAccessorArgs<AccessorOffset>();
    const auto metadata = TensorAccessor(args, address);
    noc.async_read(metadata, scratch, sizeof(uint32_t), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    // Trace replay overwrites metadata at fixed addresses. DMA completion does not evict the RISC cache.
    invalidate_l1_cache();
    return CoreLocalMem<volatile uint32_t>(scratch.get_write_ptr())[0];
}

}  // namespace kv_cache_dataflow
