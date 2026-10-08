// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tuple>
#include <utility>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
// NoC address of page `page_id` of input `idx`, where `idx` is only known at run time. Only the
// selected input's accessor is built, in place from its binding token, so stack use does not grow
// with the number of inputs. Materializing every input's accessor up front (a tuple of
// TensorAccessors, an array of AbstractTensorAccessorWrappers and two per-input arrays) grows the
// frame by ~40 B per input; on Quasar the DM stack is only what TLS leaves of the 8 KB per-DM local
// region (~1.4 KB), and beyond ~32 inputs the frame overran into TLS (rta/crta/DFB state).
template <typename Tokens, uint32_t... Is>
FORCE_INLINE uint64_t
input_page_noc_addr(const Tokens& tokens, uint32_t idx, uint32_t page_id, std::integer_sequence<uint32_t, Is...>) {
    uint64_t addr = 0;
    (void)((idx == Is ? (addr = TensorAccessor(std::get<Is>(tokens)).get_noc_addr(page_id), true) : false) || ...);
    return addr;
}
}  // namespace

// Make n reads defined by num_reads
// Writes to the bound Dataflow Buffer in L1
// Expects n input tensor bindings, reached positionally through the `inputs` binding sequence
void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_tensor = get_arg(args::start_tensor);
    const uint32_t start_tensor_id = get_arg(args::start_tensor_id);

    // The tensor binding sequence carries its own length, so the host passes no tensor count.
    constexpr uint32_t num_tensors = std::tuple_size_v<decltype(tensor::inputs)>;
    constexpr auto input_indices = std::make_integer_sequence<uint32_t, num_tensors>();

    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;

    // Two num_tensors-element runtime vararg blocks, in the order the host supplies them:
    // num_tiles_per_block first, then tile_id_per_tensor (each input's first page on this core).
    // Read on demand rather than copied into per-input stack arrays (see above).
    constexpr uint32_t tile_id_per_tensor_offset = num_tensors;

    DataflowBuffer dfb_in(dfb::in);
    // The tile size comes off the buffer object; the legacy free function took a circular-buffer
    // index, which no longer exists. Read after the buffer is constructed, since the value is not a
    // constant expression and so takes the member-getter form.
    const uint32_t tile_size_bytes = dfb_in.get_tile_size();
    Noc noc;

    // Inputs are visited in order, wrapping back to 0 after the last; `round` counts the wraps. An
    // input's page on its k-th visit is its starting page plus k blocks. For an input before
    // start_tensor the host's starting page already points at the next block (its first visit here
    // is in round 1), and start_tensor's starting page includes start_tensor_id.
    uint32_t curr_tensor = start_tensor;
    uint32_t curr_tensor_id = start_tensor_id;
    uint32_t round = 0;
    uint32_t curr_block_tiles = get_vararg(curr_tensor);
    uint32_t curr_page = get_vararg(tile_id_per_tensor_offset + curr_tensor);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_in.reserve_back(ublock_size_tiles);
        uint32_t l1_write_addr = dfb_in.get_write_ptr();
        noc.async_read(
            PrecomposedUnicastEndpoint{},
            CoreLocalMem<uint8_t>(l1_write_addr),
            tile_size_bytes,
            {.noc_addr = input_page_noc_addr(tensor::inputs, curr_tensor, curr_page, input_indices)},
            {});
        noc.async_read_barrier();
        dfb_in.push_back(ublock_size_tiles);

        curr_page++;
        curr_tensor_id++;

        if (curr_tensor_id == curr_block_tiles) {
            curr_tensor_id = 0;
            curr_tensor++;
            if (curr_tensor == num_tensors) {
                curr_tensor = 0;
                round++;
            }
            curr_block_tiles = get_vararg(curr_tensor);
            const uint32_t visits = (curr_tensor < start_tensor) ? round - 1 : round;
            curr_page = get_vararg(tile_id_per_tensor_offset + curr_tensor) + visits * curr_block_tiles;
            if (curr_tensor == start_tensor) {
                curr_page -= start_tensor_id;
            }
        }
    }
}
