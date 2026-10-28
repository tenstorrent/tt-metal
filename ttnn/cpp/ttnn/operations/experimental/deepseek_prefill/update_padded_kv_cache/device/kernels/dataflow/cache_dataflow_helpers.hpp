// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#ifdef LINEAR_COORD_FROM_FABRIC
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/common/kernels/mesh_coord_from_fabric.hpp"
#endif

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

// This chip's position on the linearized sp*tp block-cyclic axis: common arg 0, or, in a program built once for
// every chip (LINEAR_COORD_FROM_FABRIC), sp_coord * tp_factor + tp_coord from the chip's mesh coordinate.
inline uint32_t chip_linear_coord() {
#ifdef LINEAR_COORD_FROM_FABRIC
    const uint32_t sp_coord = mesh_coord_from_fabric(LINEAR_COORD_SP_AXIS, LINEAR_COORD_MESH_COLS);
#ifdef LINEAR_COORD_TP_AXIS
    const uint32_t tp_coord = mesh_coord_from_fabric(LINEAR_COORD_TP_AXIS, LINEAR_COORD_MESH_COLS);
#else
    const uint32_t tp_coord = 0;
#endif
    return sp_coord * LINEAR_COORD_TP_FACTOR + tp_coord;
#else
    return get_common_arg_val<uint32_t>(0);
#endif
}

}  // namespace kv_cache_dataflow
