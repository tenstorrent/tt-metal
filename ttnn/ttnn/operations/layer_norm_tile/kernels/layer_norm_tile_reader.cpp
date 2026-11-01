// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm reader (RISCV_1; NoC chosen per core by the host, see _core_nocs).
//   load_block   : R_b x Wt tiles of x (and residual, unless the writer RISC streams it) in C-tile
//                  chunks on a trid ring of `read_trid_depth` chunks (layer_norm_tile_stream.hpp).
//   affine       : gamma/beta do not vary along the row split, so ONE injector core reads them,
//                  issued untagged behind its block 0's last chunk, into a compact block (TILE: per tile
//                  column 64 B at offset 0 + 64 B at FACE_BYTES; ROW_MAJOR: the 64 B stick span) and
//                  multicasts it with one linked McastArgs send (no receiver handshake: the block lands
//                  in a reader-private scratch CB untouched before the flag). Every core then scatters
//                  row 0 of faces 0/1 into its gamma/beta tile slots with local self-NoC reads.
//                  A receiver with a next block does this right after issuing that block's first chunk,
//                  so the receive overlaps the block's DRAM reads (the scatter has its own trid; a plain
//                  barrier would also wait on the chunk). Compute needs gamma/beta only at block 0's
//                  affine pass, and in_depth = 2 lets block 1 be reserved before block 0 is popped.
//                  Every DRAM read is 64 B at a 64 B-aligned offset (BH DRAM alignment).
//
// The whole-CB multicast (2 x 48 KB, 3 KB live) held the NoC0 path for ~2-4.7 us and per-row
// multicasts cost ~250 ns each, hence compact + local scatter. The scatter is a strided self-NoC
// read with a per-tile source address (~2 us for Wt = 24). A stateful read (one set_read_state, per-read
// source/destination only) measured ~0.3 us slower and an L1 word copy on the RISC ~3.6x slower.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "vendor/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "layer_norm_tile_stream.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_beta_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block_row_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t io_chunk_tiles = get_compile_time_arg_val(6);
    constexpr bool has_residual = get_compile_time_arg_val(7) != 0;
    constexpr bool affine_is_tile = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t affine_page_elems = get_compile_time_arg_val(9);
    constexpr uint32_t input_tile_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t affine_page_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t affine_elem_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t read_trid_depth = get_compile_time_arg_val(13);
    constexpr bool residual_on_writer = get_compile_time_arg_val(14) != 0;
    constexpr auto input_args = TensorAccessorArgs<15>();
    [[maybe_unused]] constexpr auto residual_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr auto affine_mc = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("amc_ct_offset"), get_named_compile_time_arg_val("amc_rt_offset")>();
    constexpr uint32_t cb_affine_scratch = get_named_compile_time_arg_val("affine_scratch_cb");

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t residual_addr = get_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(2);
    const uint32_t beta_addr = get_arg_val<uint32_t>(3);
    const uint32_t row_tile_start = get_arg_val<uint32_t>(4);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(5);
    const uint32_t rotation = get_arg_val<uint32_t>(6);

    constexpr bool read_residual = has_residual && !residual_on_writer;

    static_assert(tensor_w_tiles % io_chunk_tiles == 0, "every block push must be a whole number of chunks");

    constexpr uint32_t TILE_DIM = 32;
    constexpr uint32_t FACE_DIM = 16;
    constexpr uint32_t affine_row_bytes = TILE_DIM * affine_elem_bytes;  // 64 B
    constexpr uint32_t face_row_bytes = FACE_DIM * affine_elem_bytes;    // 32 B
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * affine_elem_bytes;
    constexpr uint32_t affine_tile_bytes = 4 * face_bytes;
    constexpr uint32_t entry_bytes = affine_is_tile ? 2 * affine_row_bytes : affine_row_bytes;
    constexpr uint32_t tensor_block_bytes = tensor_w_tiles * entry_bytes;
    constexpr uint32_t affine_trid = read_trid_depth + 1;  // outside the chunk ring's ids 1..depth
    static_assert(affine_trid <= NOC_MAX_TRANSACTION_ID, "affine scatter trid out of the NoC id range");
    static_assert(affine_is_tile || affine_page_elems % TILE_DIM == 0, "RM affine stick must hold whole tile rows");

    DataflowBuffer input_tiles(cb_input_tiles);
    [[maybe_unused]] DataflowBuffer residual_tiles(cb_residual_tiles);
    DataflowBuffer gamma_tiles(cb_gamma_tiles);
    DataflowBuffer beta_tiles(cb_beta_tiles);
    // Reader-private landing area: never pushed, only its (uniform) L1 address is used.
    DataflowBuffer scratch(cb_affine_scratch);
    const uint32_t gamma_block = scratch.get_write_ptr();
    const uint32_t beta_block = gamma_block + tensor_block_bytes;

    const auto input_accessor = TensorAccessor(input_args, input_addr, input_tile_bytes);
    [[maybe_unused]] const auto residual_accessor = TensorAccessor(residual_args, residual_addr, input_tile_bytes);
    const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, affine_page_bytes);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr, affine_page_bytes);

    auto issue_entry_reads = [&](const auto& accessor, uint32_t block) {
        for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
            const uint32_t entry = block + col * entry_bytes;
            if constexpr (affine_is_tile) {
                noc_async_read(accessor.get_noc_addr(col, 0), entry, affine_row_bytes);
                noc_async_read(accessor.get_noc_addr(col, face_bytes), entry + affine_row_bytes, affine_row_bytes);
            } else {
                const uint32_t elem = col * TILE_DIM;
                noc_async_read(
                    accessor.get_noc_addr(elem / affine_page_elems, (elem % affine_page_elems) * affine_elem_bytes),
                    entry,
                    affine_row_bytes);
            }
        }
    };

    const Noc noc;

    // Entry -> slot: face-0 row 0 at slot+0, face-1 row 0 at slot+FACE_BYTES.
    constexpr uint32_t face1_src = affine_is_tile ? affine_row_bytes : face_row_bytes;
    auto scatter = [&](uint32_t block, DataflowBuffer& dst) {
        dst.reserve_back(tensor_w_tiles);
        uint32_t slot = dst.get_write_ptr();
        for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
            const uint32_t entry = block + col * entry_bytes;
            noc_async_read(get_noc_addr(entry), slot, face_row_bytes);
            noc_async_read(get_noc_addr(entry + face1_src), slot + face_bytes, face_row_bytes);
            slot += affine_tile_bytes;
        }
    };
    auto scatter_and_push = [&]() {
        dataflow_kernel_lib::set_read_trid(noc, affine_trid);
        scatter(gamma_block, gamma_tiles);
        scatter(beta_block, beta_tiles);
        dataflow_kernel_lib::async_read_barrier_with_trid(noc, affine_trid);
        dataflow_kernel_lib::set_read_trid(noc, 0);
        gamma_tiles.push_back(tensor_w_tiles);
        beta_tiles.push_back(tensor_w_tiles);
    };

    const bool is_sender = affine_mc.can_send();

    auto issue_stage = [&]() {
        issue_entry_reads(gamma_accessor, gamma_block);
        issue_entry_reads(beta_accessor, beta_block);
    };
    auto complete_and_send = [&]() {
        MaybeDeviceZoneScope("reader_affine_send");
        noc_async_read_barrier();
        if constexpr (affine_mc.sender_available) {
            affine_mc.sender(noc).send(gamma_block, gamma_block, 2 * tensor_block_bytes);
        }
        scatter_and_push();
    };
    auto receive_affine = [&]() {
        MaybeDeviceZoneScope("reader_affine");
        if constexpr (affine_mc.receiver_available) {
            affine_mc.receiver(noc).receive();
        }
        scatter_and_push();
    };

    bool send_pending = is_sender;
    bool receive_pending = !is_sender;

    for (uint32_t block_row = 0; block_row < core_row_tiles; block_row += block_row_tiles) {
        const uint32_t rows =
            (core_row_tiles - block_row) < block_row_tiles ? (core_row_tiles - block_row) : block_row_tiles;
        const uint32_t first_page = (row_tile_start + block_row) * tensor_w_tiles;
        const uint32_t num_chunks = rows * tensor_w_tiles / io_chunk_tiles;

        uint32_t input_base = 0;
        uint32_t residual_base = 0;
        {
            MaybeDeviceZoneScope("reader_reserve");
            input_tiles.reserve_back(rows * tensor_w_tiles);
            input_base = input_tiles.get_write_ptr();
            if constexpr (read_residual) {
                residual_tiles.reserve_back(rows * tensor_w_tiles);
                residual_base = residual_tiles.get_write_ptr();
            }
        }

        auto issue = [&](uint32_t chunk) {
            const uint32_t page = first_page + chunk * io_chunk_tiles;
            const uint32_t offset = chunk * io_chunk_tiles * input_tile_bytes;
            layer_norm_tile::read_chunk<io_chunk_tiles, input_tile_bytes>(
                input_accessor, page, input_base + offset, rotation);
            if constexpr (read_residual) {
                layer_norm_tile::read_chunk<io_chunk_tiles, input_tile_bytes>(
                    residual_accessor, page, residual_base + offset, rotation);
            }
            if (send_pending && chunk + 1 == num_chunks) {
                dataflow_kernel_lib::set_read_trid(noc, 0);
                issue_stage();
            }
            if (receive_pending && block_row != 0 && chunk == 0) {
                receive_affine();
                receive_pending = false;
            }
        };
        auto retire = [&]() {
            input_tiles.push_back(io_chunk_tiles);
            if constexpr (read_residual) {
                residual_tiles.push_back(io_chunk_tiles);
            }
        };
        {
            MaybeDeviceZoneScope("reader_block");
            layer_norm_tile::read_block_chunks<read_trid_depth>(noc, num_chunks, issue, retire);
        }

        if (send_pending) {
            complete_and_send();
            send_pending = false;
        }
        if (receive_pending && block_row + rows >= core_row_tiles) {
            receive_affine();
            receive_pending = false;
        }
    }
}
