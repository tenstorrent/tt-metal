// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// [Quasar bring-up: concat interleaved metal2]
//
// NOTE: This is the Metal 2.0 fork of reader_concat_stick_layout_interleaved_start_id.cpp, which
// lives beside it. ConcatMetalV2ProgramFactory (selected on Quasar) binds this file; the original
// serves the legacy descriptor factory on Wormhole/Blackhole.
//
// CACHE-COHERENCE RULE (the reason this fork exists in the form it does):
// in the WIDTH_CONCAT branch below, one output stick is assembled from one sub-stick per input
// tensor. Every one of those sub-sticks is placed by a NOC read addressed at a byte offset INTO
// the DataflowBuffer entry. No CPU store ever touches the staged row. On Quasar the DM core's L1
// view is not coherent with RISC stores, so an assembly loop that used tt_memmove / direct stores
// would corrupt the row silently -- that is exactly why
// concat_tiled_unaligned_program_factory.cpp:76-82 gates its own factory off on Quasar.

#include <stdint.h>
#include <tuple>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_tensor = get_arg(args::start_tensor);
    const uint32_t start_tensor_id = get_arg(args::start_tensor_id);

    constexpr uint32_t num_tensors = std::tuple_size_v<decltype(tensor::inputs)>;
    auto accessor_tuple = make_tensor_accessors(tensor::inputs);
    // CAUTION (tensor_accessor.h): the wrappers alias `accessor_tuple`; it must outlive them.
    auto accessors = make_abstract_tensor_accessor_wrappers(accessor_tuple);

    // Runtime varargs, in the order the host packs them:
    //   [0    .. N-1 ]  num_pages_per_block[t]
    //   [N    .. 2N-1]  page_id_per_tensor[t]
    //   [2N   .. 3N-1]  page_size_per_tensor[t]   (bytes; differs per tensor only for WIDTH_CONCAT)
    [[maybe_unused]] uint32_t num_pages_per_block[num_tensors];
    uint32_t page_id_per_tensor[num_tensors];
    uint32_t page_size_per_tensor[num_tensors];
    for (uint32_t t = 0; t < num_tensors; ++t) {
        num_pages_per_block[t] = get_vararg(t);
        page_id_per_tensor[t] = get_vararg(num_tensors + t);
        page_size_per_tensor[t] = get_vararg(2 * num_tensors + t);
    }

    Noc noc;
    DataflowBuffer dfb_in(dfb::in);

    constexpr uint32_t ublock_size_pages = 1;
    // Both are unused under WIDTH_CONCAT, where the host's block arithmetic pins start_tensor to 0
    // and every row consumes one sub-stick from every input.
    [[maybe_unused]] uint32_t curr_tensor = start_tensor;
    [[maybe_unused]] uint32_t curr_tensor_id = start_tensor_id;
    for (uint32_t i = 0; i < num_pages; ++i) {
        dfb_in.reserve_back(ublock_size_pages);
#ifdef WIDTH_CONCAT
        // Width concat always starts a row at tensor 0, and num_pages_per_block[t] == 1, so each
        // output stick is exactly one sub-stick from each input, laid down left to right.
        // The host guarantees sum(page_size_per_tensor) <= the DFB entry size.
        uint32_t dest_off = 0;
        for (uint32_t j = 0; j < num_tensors; ++j) {
            const uint32_t page_size = page_size_per_tensor[j];
            noc.async_read(
                accessors[j],
                dfb_in,
                page_size,
                {.page_id = page_id_per_tensor[j]},
                {.offset_bytes = dest_off});
            dest_off += page_size;
            page_id_per_tensor[j]++;
        }
#else
        noc.async_read(
            accessors[curr_tensor],
            dfb_in,
            page_size_per_tensor[curr_tensor],
            {.page_id = page_id_per_tensor[curr_tensor]},
            {.offset_bytes = 0});

        page_id_per_tensor[curr_tensor]++;
        curr_tensor_id++;

        if (curr_tensor_id == num_pages_per_block[curr_tensor]) {
            curr_tensor_id = 0;
            curr_tensor++;
            if (curr_tensor == num_tensors) {
                curr_tensor = 0;
            }
        }
#endif
        noc.async_read_barrier();
        dfb_in.push_back(ublock_size_pages);
    }
}
