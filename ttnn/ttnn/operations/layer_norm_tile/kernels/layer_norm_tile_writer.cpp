// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_tile writer.
//   fetch_affine_rows: once per kernel, builds one row-0 tile per tile-column of gamma and beta (only
//                      face-0 row 0 and face-1 row 0 are meaningful; the Row broadcast reads nothing else).
//                      Runs here, not on the reader, so the reader issues block 0 first (stall-shadow).
//   load_block (residual stream): tops up cb_residual_tiles on this RISC's NoC, in parallel with the
//                      reader's x stream on the other NoC.
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
// RM: one 64 B read per tile-column (DRAM reads need 64 B alignment on BH, so the 32 B face row is
// widened; the extra bytes land in rows that are never read). TILE: face-0 row 0 and face-1 row 0 are
// FACE_BYTES apart, so one read spanning both costs a single request per tile-column.
constexpr uint32_t CHUNK_BYTES = 64;
constexpr uint32_t TILE_SPAN_BYTES = FACE_BYTES + CHUNK_BYTES;
constexpr uint32_t CHUNK_ELEMS = 2 * FACE_ROW_BYTES / ELEM_BYTES;  // 32 elements: one tile-column

// Issues the reads for one tensor's row-0 tiles; the caller owns the barrier so gamma and beta share one.
template <bool is_row_major, uint32_t stick_elems, typename Accessor, typename Rows>
FORCE_INLINE void issue_affine_reads(const Accessor& accessor, Rows& rows, uint32_t tensor_w_tiles) {
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
            noc_async_read(accessor.get_noc_addr(col), tile_addr, TILE_SPAN_BYTES);
        }
    }
}

// RM only: elements 16..31 arrived as face-0 row 1; face-1 row 0 needs them.
template <bool is_row_major, typename Rows>
FORCE_INLINE void issue_face1_copies(Noc noc, Rows& rows, uint32_t tensor_w_tiles) {
    if constexpr (is_row_major) {
        const uint32_t tile_bytes = rows.get_tile_size();
        const uint32_t base = rows.get_write_ptr();
        for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
            const uint32_t tile_offset = col * tile_bytes;
            noc.async_read(
                UnicastEndpoint{},
                rows,
                FACE_ROW_BYTES,
                dataflow_kernel_lib::local_addr(base + tile_offset + FACE_ROW_BYTES, noc.get_noc_id()),
                {.offset_bytes = tile_offset + FACE_BYTES});
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_rows = get_compile_time_arg_val(1);
    constexpr uint32_t cb_beta_rows = get_compile_time_arg_val(2);
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t output_tile_bytes = get_compile_time_arg_val(5);
    // True when the residual stream is fetched here (the other NoC) rather than by the reader.
    constexpr bool fetch_residual = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t residual_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t residual_depth_rows = get_compile_time_arg_val(8);
    constexpr bool gamma_is_rm = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t gamma_stick_elems = get_compile_time_arg_val(10);
    constexpr uint32_t gamma_page_bytes = get_compile_time_arg_val(11);
    constexpr bool beta_is_rm = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t beta_stick_elems = get_compile_time_arg_val(13);
    constexpr uint32_t beta_page_bytes = get_compile_time_arg_val(14);
    constexpr auto output_args = TensorAccessorArgs<15>();
    constexpr auto gamma_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto residual_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t residual_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_row = get_arg_val<uint32_t>(4);
    const uint32_t core_rows = get_arg_val<uint32_t>(5);

    // load_block, residual stream (when fetch_residual): runs on this RISC's NoC, concurrently with the
    // reader's x stream on the other NoC. Rows are topped up ahead of each output drain; reserving row r + depth - 1
    // never blocks for long, because compute popped residual rows < r before producing output row r - 1.
    [[maybe_unused]] const auto residual_accessor = TensorAccessor(residual_args, residual_addr, residual_tile_bytes);
    uint32_t residual_rows_pushed = 0;
    auto top_up_residual = [&](uint32_t row_limit) {
        if constexpr (fetch_residual) {
            for (; residual_rows_pushed < row_limit; ++residual_rows_pushed) {
                const uint32_t first_page = (start_row + residual_rows_pushed) * tensor_w_tiles;
                cb_reserve_back(cb_residual_tiles, tensor_w_tiles);
#ifndef LAYER_NORM_ABLATE_READS  // perf ablation: keep the CB handshake, drop the NoC payload
                uint32_t l1_write_addr = get_write_ptr(cb_residual_tiles);
                for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
                    noc_async_read(
                        residual_accessor.get_noc_addr(first_page + col), l1_write_addr, residual_tile_bytes);
                    l1_write_addr += residual_tile_bytes;
                }
                noc_async_read_barrier();
#endif
                cb_push_back(cb_residual_tiles, tensor_w_tiles);
            }
        }
    };
    auto residual_row_limit = [&](uint32_t output_row) {
        const uint32_t limit = output_row + residual_depth_rows;
        return limit < core_rows ? limit : core_rows;
    };

    // Compute needs residual row 0 first (add_residual_block) and gamma/beta only at scale_block. Issuing
    // the affine reads ahead of it was measured slower: the grid-wide fetch of a few hot gamma/beta banks
    // delays the residual responses.
    top_up_residual(1);  // every assigned core owns >= 1 tile-row

    Noc noc;
    const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr, beta_page_bytes);
    DataflowBuffer gamma_rows(cb_gamma_rows);
    DataflowBuffer beta_rows(cb_beta_rows);
    issue_affine_reads<gamma_is_rm, gamma_stick_elems>(gamma_accessor, gamma_rows, tensor_w_tiles);
    issue_affine_reads<beta_is_rm, beta_stick_elems>(beta_accessor, beta_rows, tensor_w_tiles);
    noc_async_read_barrier();
    if constexpr (gamma_is_rm || beta_is_rm) {
        issue_face1_copies<gamma_is_rm>(noc, gamma_rows, tensor_w_tiles);
        issue_face1_copies<beta_is_rm>(noc, beta_rows, tensor_w_tiles);
        noc.async_read_barrier();
    }
    gamma_rows.push_back(tensor_w_tiles);
    beta_rows.push_back(tensor_w_tiles);

    // store_block
    const auto output_accessor = TensorAccessor(output_args, output_addr, output_tile_bytes);
    for (uint32_t row = 0; row < core_rows; ++row) {
        top_up_residual(residual_row_limit(row));
        const uint32_t first_page = (start_row + row) * tensor_w_tiles;
        cb_wait_front(cb_output_tiles, tensor_w_tiles);
#ifndef LAYER_NORM_ABLATE_WRITES  // perf ablation: keep the CB handshake, drop the NoC payload
        uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
        for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
            noc_async_write(l1_read_addr, output_accessor.get_noc_addr(first_page + col), output_tile_bytes);
            l1_read_addr += output_tile_bytes;
        }
#endif
        noc_async_writes_flushed();
        cb_pop_front(cb_output_tiles, tensor_w_tiles);
    }
    noc_async_write_barrier();
}
