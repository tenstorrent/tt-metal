// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_dataflow_common.hpp"

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

#include <cstdint>

void kernel_main() {
    // Runtime arguments
    const uint32_t id = get_arg(args::id);
    const uint32_t work_per_core = get_arg(args::work_per_core);

    // Compile time arguments
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t total_number_of_cores = get_arg(args::total_number_of_cores);
    constexpr bool uint16_output = get_arg(args::uint16_output) == 1;

#if not GENERATE_INDICES
    // Precomputed indices tensor accessor
    const auto indices_accessor = TensorAccessor(ta::indices);
#endif  // not GENERATE_INDICES

    // Constants
    constexpr uint32_t onetile = 1;

    // Tensor accessor
    const auto inout_tensor_accessor = TensorAccessor(ta::input);

    Noc noc;
    DataflowBuffer cb_in0(dfb::cb_in0);
    DataflowBuffer cb_index(dfb::cb_index);
    const uint32_t tile_bytes_in0 = cb_in0.get_tile_size();
#if not GENERATE_INDICES
    const uint32_t tile_bytes_index = cb_index.get_tile_size();
#endif

    // Read data and generate indices
    for (uint32_t core_loop = 0; core_loop < work_per_core; core_loop++) {
        const uint32_t row = id + core_loop * total_number_of_cores;
        for (uint32_t w = 0; w < Wt; ++w) {
            cb_in0.reserve_back(onetile);
            noc.async_read(
                inout_tensor_accessor, cb_in0, tile_bytes_in0, {.page_id = row * Wt + w}, {.offset_bytes = 0});
            noc.async_read_barrier();

            cb_in0.push_back(onetile);
#if GENERATE_INDICES
            if (uint16_output) {
                generate_index_tile<uint16_t>(dfb::cb_index, w);
            } else {
                generate_index_tile<uint32_t>(dfb::cb_index, w);
            }
#else
            // Read precomputed indices to circular buffer
            cb_index.reserve_back(onetile);
            noc.async_read(
                indices_accessor, cb_index, tile_bytes_index, {.page_id = row * Wt + w}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_index.push_back(onetile);
#endif  // GENERATE_INDICES
        }  // w loop
    }  // core_loop loop
}
