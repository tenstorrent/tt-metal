// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_tile writer.
//   fetch_affine_rows: once per kernel, builds one row-0 tile per tile-column of gamma and beta (only
//                      face-0 row 0 and face-1 row 0 are meaningful; the Row broadcast reads nothing else).
//                      Runs here, not on the reader, so the reader issues block 0 first (stall-shadow).
//   store_block:       drains cb_output_tiles one tile-row at a time.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

namespace {

constexpr uint32_t ELEM_BYTES = 2;  // bf16 (validated)
constexpr uint32_t FACE_ROW_BYTES = 16 * ELEM_BYTES;
constexpr uint32_t FACE_BYTES = 16 * 16 * ELEM_BYTES;
// One NoC read covers a 64 B aligned chunk: DRAM reads need 64 B alignment (BH), so the transfer is
// widened from the 32 B face row to 64 B; the extra bytes land in rows that are never read.
constexpr uint32_t CHUNK_BYTES = 64;
constexpr uint32_t CHUNK_ELEMS = 2 * FACE_ROW_BYTES / ELEM_BYTES;  // 32 elements: one tile-column

template <bool is_row_major, uint32_t stick_elems, typename Accessor>
FORCE_INLINE void fetch_affine_rows(Noc noc, const Accessor& accessor, uint32_t cb_id, uint32_t tensor_w_tiles) {
    DataflowBuffer rows(cb_id);
    const uint32_t tile_bytes = rows.get_tile_size();
    rows.reserve_back(tensor_w_tiles);
    const uint32_t base = rows.get_write_ptr();
    for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
        const uint32_t tile_addr = base + col * tile_bytes;
        if constexpr (is_row_major) {
            const uint32_t first_elem = col * CHUNK_ELEMS;
            noc_async_read(
                accessor.get_noc_addr(first_elem / stick_elems, (first_elem % stick_elems) * ELEM_BYTES),
                tile_addr,
                CHUNK_BYTES);
        } else {
            noc_async_read(accessor.get_noc_addr(col, 0), tile_addr, CHUNK_BYTES);
            noc_async_read(accessor.get_noc_addr(col, FACE_BYTES), tile_addr + FACE_BYTES, CHUNK_BYTES);
        }
    }
    noc_async_read_barrier();
    if constexpr (is_row_major) {
        // Elements 16..31 arrived as face-0 row 1; face-1 row 0 needs them.
        for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
            const uint32_t tile_offset = col * tile_bytes;
            noc.async_read(
                UnicastEndpoint{},
                rows,
                FACE_ROW_BYTES,
                dataflow_kernel_lib::local_addr(base + tile_offset + FACE_ROW_BYTES, noc.get_noc_id()),
                {.offset_bytes = tile_offset + FACE_BYTES});
        }
        noc.async_read_barrier();
    }
    rows.push_back(tensor_w_tiles);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_rows = get_compile_time_arg_val(1);
    constexpr uint32_t cb_beta_rows = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block_height = get_compile_time_arg_val(4);
    constexpr uint32_t output_tile_bytes = get_compile_time_arg_val(5);
    constexpr bool gamma_is_rm = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t gamma_stick_elems = get_compile_time_arg_val(7);
    constexpr uint32_t gamma_page_bytes = get_compile_time_arg_val(8);
    constexpr bool beta_is_rm = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t beta_stick_elems = get_compile_time_arg_val(10);
    constexpr uint32_t beta_page_bytes = get_compile_time_arg_val(11);
    constexpr auto output_args = TensorAccessorArgs<12>();
    constexpr auto gamma_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_row = get_arg_val<uint32_t>(3);
    const uint32_t num_blocks = get_arg_val<uint32_t>(4);
    const uint32_t last_block_extent = get_arg_val<uint32_t>(5);

    Noc noc;
    const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr, beta_page_bytes);
    fetch_affine_rows<gamma_is_rm, gamma_stick_elems>(noc, gamma_accessor, cb_gamma_rows, tensor_w_tiles);
    fetch_affine_rows<beta_is_rm, beta_stick_elems>(noc, beta_accessor, cb_beta_rows, tensor_w_tiles);

    const auto output_accessor = TensorAccessor(output_args, output_addr, output_tile_bytes);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_rows = (block_idx + 1 == num_blocks) ? last_block_extent : block_height;
        const uint32_t block_first_row = start_row + block_idx * block_height;
        for (uint32_t row = 0; row < block_rows; ++row) {
            const uint32_t first_page = (block_first_row + row) * tensor_w_tiles;
            cb_wait_front(cb_output_tiles, tensor_w_tiles);
            uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
            for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
                noc_async_write(l1_read_addr, output_accessor.get_noc_addr(first_page + col), output_tile_bytes);
                l1_read_addr += output_tile_bytes;
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_output_tiles, tensor_w_tiles);
        }
    }
    noc_async_write_barrier();
}
