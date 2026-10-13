// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Data-movement helpers shared by the qkv_causal_conv1d_silu kernels.
#pragma once

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
#include "api/core_local_mem.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

namespace qkv_conv {

// The first word of `accessor`'s page 0, read through the aligned L1 slot `slot`.
template <typename Accessor>
inline uint32_t read_word(Noc& noc, const Accessor& accessor, uint32_t slot) {
    noc.async_read(accessor, CoreLocalMem<uint32_t>(slot), sizeof(uint32_t), {.page_id = 0}, {});
    noc.async_read_barrier();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot)[0];
}

// This rank's chronology over [actual_start, actual_end), or the full capacity without actual_end.
template <uint32_t has_actual_end, uint32_t sp_rank, uint32_t sp_size, uint32_t local_rows>
inline kda_chronology::Topology read_chronology(Noc& noc, uint32_t slot) {
    const uint32_t start = read_word(noc, TensorAccessor(tensor::actual_start), slot);
    uint32_t end = 0;
    if constexpr (has_actual_end) {
        end = read_word(noc, TensorAccessor(*tensor::get_token_if_present<"actual_end">()), slot);
    }
    return kda_chronology::derive_bounds<has_actual_end>(start, end, sp_rank, sp_size, local_rows);
}

}  // namespace qkv_conv
