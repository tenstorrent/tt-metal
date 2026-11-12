// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// [Quasar bring-up: concat interleaved metal2]
//
// NOTE: This is the Metal 2.0 fork of reader_concat_interleaved_start_id.cpp, which lives beside it.
// ConcatMetalV2ProgramFactory (selected on Quasar) binds this file; the original serves the legacy
// descriptor factory on Wormhole/Blackhole. Until the original is retired, changes here likely
// belong there too.
//
// Differences from the legacy fork, and only these:
//   * named args (args::num_tiles, ...) instead of positional get_arg_val<>
//   * the per-tensor source base addresses are GONE from the arg list: they arrive through the
//     host's TensorParameter/TensorBinding plumbing, which also refreshes them on a program-cache
//     hit. The variadic tensor count is expressed with a TensorBindingSequence (tensor::inputs).
//   * the two per-tensor uint32 arrays arrive as runtime varargs rather than being read out of the
//     raw RTA block with get_arg_addr().
//   * the NOC read lands directly in the DataflowBuffer (dfb::in) rather than at a CoreLocalMem
//     built from get_write_ptr(). This matters on Quasar: the DM L1 is not coherent with CPU
//     stores there, so every byte of an assembled page must be placed by the NOC, never by the
//     RISC. See the TODO(ARCH_QUASAR) gate in concat_tiled_unaligned_program_factory.cpp:76-82 for
//     what the coherent-store rule costs when it is broken -- that factory corrupts silently.

#include <stdint.h>
#include <tuple>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_tensor = get_arg(args::start_tensor);
    const uint32_t start_tensor_id = get_arg(args::start_tensor_id);

    // One TensorAccessor per input tensor. The count is a property of the binding sequence the
    // host declared, so no separate count argument is needed.
    constexpr uint32_t num_tensors = std::tuple_size_v<decltype(tensor::inputs)>;
    auto accessor_tuple = make_tensor_accessors(tensor::inputs);
    // CAUTION (tensor_accessor.h): the wrappers alias `accessor_tuple`; it must outlive them.
    auto accessors = make_abstract_tensor_accessor_wrappers(accessor_tuple);

    // Runtime varargs, in the order the host packs them:
    //   [0        .. N-1 ]  num_tiles_per_block[t]
    //   [N        .. 2N-1]  tile_id_per_tensor[t]
    uint32_t num_tiles_per_block[num_tensors];
    uint32_t tile_id_per_tensor[num_tensors];
    for (uint32_t t = 0; t < num_tensors; ++t) {
        num_tiles_per_block[t] = get_vararg(t);
        tile_id_per_tensor[t] = get_vararg(num_tensors + t);
    }

    Noc noc;
    DataflowBuffer dfb_in(dfb::in);
    // Tiled path: one DFB entry is exactly one tile, for every input and for the output.
    const uint32_t tile_size_bytes = dfb_in.get_entry_size();

    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t curr_tensor = start_tensor;
    uint32_t curr_tensor_id = start_tensor_id;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_in.reserve_back(ublock_size_tiles);
        noc.async_read(
            accessors[curr_tensor],
            dfb_in,
            tile_size_bytes,
            {.page_id = tile_id_per_tensor[curr_tensor]},
            {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in.push_back(ublock_size_tiles);

        tile_id_per_tensor[curr_tensor]++;
        curr_tensor_id++;

        if (curr_tensor_id == num_tiles_per_block[curr_tensor]) {
            curr_tensor_id = 0;
            curr_tensor++;
            if (curr_tensor == num_tensors) {
                curr_tensor = 0;
            }
        }
    }
}
