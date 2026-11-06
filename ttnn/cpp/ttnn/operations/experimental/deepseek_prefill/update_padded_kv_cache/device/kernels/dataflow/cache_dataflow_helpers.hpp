// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/assert.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/chip_position/kernels/chip_position.hpp"

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

// This chip's position on the linearized sp*tp axis, from the chip table appended to the compile-time args.
inline uint32_t linear_coord_from_chip_table() {
    constexpr uint32_t offset = get_named_compile_time_arg_val("chip_table_offset");
    constexpr uint32_t size = get_named_compile_time_arg_val("chip_table_size");
    const uint32_t index = deepseek_prefill::chip_table_index<get_named_compile_time_arg_val("chip_lookup") != 0>();
    ASSERT(index < size);
    return kernel_compile_time_args[offset + index];
}

}  // namespace kv_cache_dataflow
