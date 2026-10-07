// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"

namespace select_rows {

constexpr uint32_t no_record = ~0U;

// The selected row indices: read from the index tensor, or derived on device from the chronology for a history
// selection record. `scratch` is a 64-byte aligned L1 word buffer.
template <
    uint32_t rows,
    uint32_t record,
    uint32_t has_actual_end,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows>
FORCE_INLINE void resolve_rows(Noc& noc, uint32_t scratch, uint32_t (&indices)[rows]) {
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    if constexpr (record == no_record) {
        const auto index_tensor = TensorAccessor(*tensor::get_token_if_present<"indices">());
        noc.async_read(index_tensor, CoreLocalMem<uint32_t>(scratch), rows * sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        for (uint32_t i = 0; i < rows; ++i) {
            indices[i] = words[i];
        }
    } else {
        const auto actual_start = TensorAccessor(*tensor::get_token_if_present<"actual_start">());
        noc.async_read(actual_start, CoreLocalMem<uint32_t>(scratch), sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        const uint32_t start = words[0];
        auto topology = kda_chronology::derive(start, sp_rank, sp_size, local_rows);
        if constexpr (has_actual_end) {
            const auto actual_end = TensorAccessor(*tensor::get_token_if_present<"actual_end">());
            noc.async_read(actual_end, CoreLocalMem<uint32_t>(scratch), sizeof(uint32_t), {.page_id = 0}, {});
            noc.async_read_barrier();
            topology = kda_chronology::derive_interval(start, words[0], sp_rank, sp_size, local_rows);
        }
        uint32_t derived[kda_chronology::selection::packed_history_rows];
        kda_chronology::selection_history_rows(topology, record, sp_size, local_rows, derived);
        for (uint32_t i = 0; i < rows; ++i) {
            indices[i] = derived[i];
        }
    }
}

}  // namespace select_rows
