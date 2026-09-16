// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_topology/chronology.hpp"
using namespace kda_chronology;
template <uint32_t P, uint32_t C, uint32_t BH, uint32_t K, uint32_t V>
TT_KERNEL void derive() {
    Noc noc;
    Scratchpad<volatile uint32_t> scratch(scratch::scratch);
    const auto starts = TensorAccessor(tensor::start);
    const auto ranks = TensorAccessor(tensor::rank);
    const auto output = TensorAccessor(tensor::output);
    auto address = scratch.get_base_address();
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    noc.async_read(starts, CoreLocalMem<uint32_t>(address), 4, {.page_id = 0}, {});
    noc.async_read_barrier();
    const uint32_t start = words[0];
    noc.async_read(ranks, CoreLocalMem<uint32_t>(address), 4, {.page_id = 0}, {});
    noc.async_read_barrier();
    auto t = derive(start, words[0], P, C);
    for (uint32_t row = 0; row < 8 + 2 * P; ++row) {
        for (uint32_t i = 0; i < 8; ++i) {
            words[i] = 0;
        }
        if (row == 0) {
            words[0] = t.boundary;
            words[1] = t.head_rows;
            words[2] = t.local_split;
            words[3] = t.rank;
            words[4] = t.final_owner;
            words[5] = t.split;
            words[6] = C;
        } else if (row <= 3) {
            uint32_t base = row == 1 ? (t.local_split ? t.head_rows : C) - 3
                                     : (row == 2 ? (t.rank + P - 1) % P : t.final_owner) * 3;
            for (uint32_t i = 0; i < 3; ++i) {
                words[i] = base + i;
            }
        } else {
            uint32_t selected;
            if (row < 6) {
                selected = (t.rank + P - t.boundary) % P;
            } else if (row < 8) {
                selected = t.split ? t.final_owner : P;
            } else {
                selected = (t.boundary + (row - 8) / 2) % P;
            }
            const bool end = row % 2 != 0;
            words[0] = selected + uint32_t(end);
            if (end) {
                words[1] = BH;
                words[2] = K;
                words[3] = row >= 8 ? K + V : V;
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(address), output, 32, {}, {.page_id = row});
        noc.async_write_barrier();
    }
}
