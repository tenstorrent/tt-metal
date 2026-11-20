// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/correction_core.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/dataflow/gate_matmul_dataflow.hpp"

constexpr uint32_t cb_chron = get_compile_time_arg_val(0);
constexpr uint32_t cb_norm = get_compile_time_arg_val(1);
constexpr uint32_t kt = get_compile_time_arg_val(2);
constexpr uint32_t num_chunks = get_compile_time_arg_val(3);
// The cores' layout (correction_core.hpp).
constexpr uint32_t cores_per_head = get_compile_time_arg_val(4);
constexpr uint32_t heads_per_column = get_compile_time_arg_val(5);
constexpr uint32_t core_chunks = get_compile_time_arg_val(6);
constexpr uint32_t first_row = get_compile_time_arg_val(7);
// The normalized outputs are token-major [B, T, H*V]: head h's tiles of chunk c at tile row (h / heads) * num_chunks +
// c, from tile column (h % heads) * kt.
constexpr uint32_t heads = get_compile_time_arg_val(8);
constexpr auto output_args = TensorAccessorArgs<9>();
// The output gate's operands (gate_matmul_dataflow.hpp): this kernel moves x of the core's chunks, on NoC 1.
using Gate = kda_gate::Operands<output_args.next_compile_time_args_offset()>;

// Every correction core of a head moves x of its chunks for its matmul (in the group of the head batch's cores at its
// place), then writes its chunk outputs.
void kernel_main() {
    // Every core takes the same args (common, per device): its head and share follow from its place in the grid.
    const auto self = kda_relay::correction_core<heads_per_column, first_row, cores_per_head>();
    const uint32_t head = self.head;
    const uint32_t place = self.place;
    const uint32_t first_chunk = place * core_chunks;
    Gate::move_x((head / heads) * cores_per_head + place, (head / heads) * num_chunks + first_chunk, head);

    // The reader's chunk count.
    cb_wait_front(cb_chron, 1);
    const uint32_t chunks = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_chron))[0];
    cb_pop_front(cb_chron, 1);

    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(0));
    const uint32_t out_tile = get_tile_size(cb_norm);
    for (uint32_t c = 0; c < chunks; ++c) {
        cb_wait_front(cb_norm, kt);
        const uint32_t page = (((head / heads) * num_chunks + first_chunk + c) * heads + head % heads) * kt;
        for (uint32_t j = 0; j < kt; ++j) {
            noc_async_write_page(page + j, output, get_read_ptr(cb_norm) + j * out_tile);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_norm, kt);
    }
    noc_async_write_barrier();
}
