// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"

namespace trace_metadata {

// Read one per-chunk metadata scalar -- page 0 of a 1-element uint32 DRAM tensor -- into L1 scratch and
// return its value.
//
// Such a tensor sits at a FIXED DRAM address that the host refreshes in place between trace replays,
// which is what lets a single captured program pick up new per-chunk scalars instead of having them baked
// in as runtime args. Two steps of the sequence are easy to omit and both fail *silently*, with
// plausible-but-wrong data rather than an error:
//
//   - `async_read_barrier()` orders the DMA but does NOT invalidate the RISC data cache. Since the
//     address is reused every chunk, a cached L1 line hands back the PRIOR chunk's value. Hence
//     `invalidate_l1_cache()` between the barrier and the load.
//   - the load itself must be volatile, or it can be hoisted above the DMA.
//
// `dst_l1_addr` is deliberately the caller's choice: which CB's L1 may be borrowed as scratch, and at
// which offset, is kernel-specific -- these small reads only land correctly at a CB page base on some
// platforms -- so the caller passes an address it has reasoned about rather than this helper guessing.
template <typename NocT, typename AccessorArgsT>
inline uint32_t read_metadata_scalar_u32(
    NocT& noc, const AccessorArgsT& accessor_args, uint32_t tensor_addr, uint32_t dst_l1_addr) {
    const auto accessor = TensorAccessor(accessor_args, tensor_addr);
    noc.async_read(accessor, CoreLocalMem<uint8_t>(dst_l1_addr), sizeof(uint32_t), {.page_id = 0}, {});
    noc.async_read_barrier();
    invalidate_l1_cache();
    return CoreLocalMem<volatile uint32_t>(dst_l1_addr)[0];
}

}  // namespace trace_metadata
