// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

namespace dataflow_kernel_lib {

// Face size in uint32 (128 u32 = 256 bf16 = 16x16 face)
constexpr uint32_t FACE_SIZE_U32 = 128;

// Face size in uint32 for float32 (256 u32 = 256 f32 = 16x16 face)
constexpr uint32_t FACE_SIZE_U32_FP32 = 256;

/**
 * @brief Convert an L1 address to a volatile L1 pointer
 *
 * @param addr L1 memory address
 * @return Volatile pointer to uint32_t in L1 memory
 */
FORCE_INLINE volatile tt_l1_ptr uint32_t* addr_to_l1_ptr(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

/**
 * @brief Create NOC source/destination args for a local L1 address on this core
 *
 * @param addr L1 memory address
 * @param noc_id NOC index (defaults to the current core's noc_index)
 * @return UnicastEndpoint src_args_type with this core's NOC coordinates and the given address
 */
FORCE_INLINE auto local_noc_addr(uint32_t addr, uint8_t noc_id = noc_index) {
    return noc_traits_t<UnicastEndpoint>::src_args_type{
        .noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = addr};
}

/**
 * @brief Zero out the exact tile size for a DFB's current write entry using the device zero API.
 *
 * @param dfb DataflowBuffer whose current write entry should be zeroed
 */
FORCE_INLINE void zero_tile(::DataflowBuffer dfb) {
    Noc noc;
    noc.async_write_zeros(dfb, dfb.get_tile_size());
    noc.write_zeros_l1_barrier();
}

/**
 * @brief Reserve, zero-fill, and push one tile into a DataflowBuffer
 *
 * @tparam dfb_id DataflowBuffer ID whose tile byte size should be used
 */
template <uint32_t dfb_id>
FORCE_INLINE void prepare_zero_tile() {
    ::DataflowBuffer dfb(dfb_id);
    dfb.reserve_back(1);
    zero_tile(dfb);
    dfb.push_back(1);
}

}  // namespace dataflow_kernel_lib
