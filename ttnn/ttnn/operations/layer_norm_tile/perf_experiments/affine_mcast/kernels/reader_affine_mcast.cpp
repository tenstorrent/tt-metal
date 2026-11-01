// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// affine_mcast reader: the op's reader with gamma/beta staged ONCE on an injector core and
// multicast to every active core (two McastArgs channels, one per affine CB).
//   injector  : stages row 0 of every gamma/beta tile exactly like the op (incl. the RM self-NoC
//               fix-up), before its own block 0, then sends the whole Wt-tile CB region to the grid.
//   receivers : no affine reads at all; after block 0's chunks (where the op barriers on its own
//               affine reads) they wait the data-ready flag and push the CBs.
// Every block/chunk path is unchanged from kernels/layer_norm_tile_reader.cpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "../../../kernels/layer_norm_tile_stream.hpp"

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
    constexpr auto gamma_mc = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("gmc_ct_offset"), get_named_compile_time_arg_val("gmc_rt_offset")>();
    constexpr auto beta_mc = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("bmc_ct_offset"), get_named_compile_time_arg_val("bmc_rt_offset")>();
    // Receivers wait for the broadcast before block 0 (required to keep a handshake channel from
    // stalling the injector behind the slowest core's block 0) or after it (no-handshake channel).
    constexpr bool receive_early = get_named_compile_time_arg_val("affine_recv_early") != 0;
    // Injector stages behind its own block 0's last chunk (as the op does) instead of before block 0.
    // With no remote receivers (1-core call) staging early only serializes it ahead of block 0, so the
    // lone core stages behind block 0 exactly like the op.
    constexpr bool send_late = get_named_compile_time_arg_val("affine_send_late") != 0 || !gamma_mc.has_receivers;
    // Raw payload modes (the helper's send_signal() still publishes the flag after a write barrier):
    //   sparse : only the live 64 B rows of each slot (face-0 row 0, face-1 row 0), 3 KB per tensor.
    //   chunk  : the whole CB region as UNLINKED multicasts of affine_chunk_bytes each.
    // Raw because SenderPipe::send() issues the region as one linked multicast (the path stays
    // reserved for the whole 48 KB) and takes one contiguous region per data-ready signal.
    constexpr bool sparse = get_named_compile_time_arg_val("affine_sparse") != 0;
    constexpr uint32_t chunk_bytes = get_named_compile_time_arg_val("affine_chunk_bytes");
    constexpr bool raw_payload = sparse || chunk_bytes != 0;

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t residual_addr = get_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(2);
    const uint32_t beta_addr = get_arg_val<uint32_t>(3);
    const uint32_t row_tile_start = get_arg_val<uint32_t>(4);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(5);
    const uint32_t rotation = get_arg_val<uint32_t>(6);
    // affine_sparse: arg 7 = rectangle count, then (sx, sy, ex, ey, num_dests) per receiver rectangle.
    constexpr uint32_t RECT_ARGS = 7;

    constexpr bool read_residual = has_residual && !residual_on_writer;

    static_assert(tensor_w_tiles % io_chunk_tiles == 0, "every block push must be a whole number of chunks");

    constexpr uint32_t TILE_DIM = 32;
    constexpr uint32_t FACE_DIM = 16;
    constexpr uint32_t affine_row_bytes = TILE_DIM * affine_elem_bytes;
    constexpr uint32_t face_row_bytes = FACE_DIM * affine_elem_bytes;
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * affine_elem_bytes;
    constexpr uint32_t affine_tile_bytes = 4 * face_bytes;
    constexpr uint32_t affine_cb_bytes = tensor_w_tiles * affine_tile_bytes;
    static_assert(affine_is_tile || affine_page_elems % TILE_DIM == 0, "RM affine stick must hold whole tile rows");

    DataflowBuffer input_tiles(cb_input_tiles);
    [[maybe_unused]] DataflowBuffer residual_tiles(cb_residual_tiles);
    DataflowBuffer gamma_tiles(cb_gamma_tiles);
    DataflowBuffer beta_tiles(cb_beta_tiles);

    const auto input_accessor = TensorAccessor(input_args, input_addr, input_tile_bytes);
    [[maybe_unused]] const auto residual_accessor = TensorAccessor(residual_args, residual_addr, input_tile_bytes);
    const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, affine_page_bytes);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr, affine_page_bytes);

    auto issue_affine_reads = [&](const auto& accessor, DataflowBuffer& dst) {
        dst.reserve_back(tensor_w_tiles);
        uint32_t slot_addr = dst.get_write_ptr();
        for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
            if constexpr (affine_is_tile) {
                noc_async_read(accessor.get_noc_addr(col, 0), slot_addr, affine_row_bytes);
                noc_async_read(accessor.get_noc_addr(col, face_bytes), slot_addr + face_bytes, affine_row_bytes);
            } else {
                const uint32_t elem = col * TILE_DIM;
                noc_async_read(
                    accessor.get_noc_addr(elem / affine_page_elems, (elem % affine_page_elems) * affine_elem_bytes),
                    slot_addr,
                    affine_row_bytes);
            }
            slot_addr += affine_tile_bytes;
        }
    };

    auto finish_affine = [&](DataflowBuffer& dst) {
        if constexpr (!affine_is_tile) {
            uint32_t slot_addr = dst.get_write_ptr();
            for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
                noc_async_read(get_noc_addr(slot_addr + face_row_bytes), slot_addr + face_bytes, face_row_bytes);
                slot_addr += affine_tile_bytes;
            }
        }
    };

    const Noc noc;
    const bool is_sender = gamma_mc.can_send();

    // Injector: barrier the staged rows, RM fix-up, then broadcast both CBs and feed its own compute.
    auto complete_and_send = [&]() {
        MaybeDeviceZoneScope("reader_affine_send");
        {
            MaybeDeviceZoneScope("reader_affine_stage");
            noc_async_read_barrier();
            if constexpr (!affine_is_tile) {
                finish_affine(gamma_tiles);
                finish_affine(beta_tiles);
                noc_async_read_barrier();
            }
        }
        const uint32_t gamma_base = gamma_tiles.get_write_ptr();
        const uint32_t beta_base = beta_tiles.get_write_ptr();
        if constexpr (gamma_mc.sender_available) {
            if constexpr (raw_payload) {
                const uint32_t num_rects = get_arg_val<uint32_t>(RECT_ARGS);
                for (uint32_t rect = 0; rect < num_rects; ++rect) {
                    const uint32_t base_arg = RECT_ARGS + 1 + rect * 5;
                    const uint32_t num_dests = get_arg_val<uint32_t>(base_arg + 4);
                    const uint64_t mcast_base = get_noc_multicast_addr(
                        get_arg_val<uint32_t>(base_arg),
                        get_arg_val<uint32_t>(base_arg + 1),
                        get_arg_val<uint32_t>(base_arg + 2),
                        get_arg_val<uint32_t>(base_arg + 3),
                        0);
                    for (uint32_t base : {gamma_base, beta_base}) {
                        if constexpr (!sparse) {
                            for (uint32_t offset = 0; offset < affine_cb_bytes; offset += chunk_bytes) {
                                const uint32_t bytes =
                                    affine_cb_bytes - offset < chunk_bytes ? affine_cb_bytes - offset : chunk_bytes;
                                noc_async_write_multicast(
                                    base + offset, mcast_base | (base + offset), bytes, num_dests);
                            }
                            continue;
                        }
                        for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
                            const uint32_t slot = base + col * affine_tile_bytes;
                            noc_async_write_multicast(slot, mcast_base | slot, affine_row_bytes, num_dests);
                            noc_async_write_multicast(
                                slot + face_bytes, mcast_base | (slot + face_bytes), affine_row_bytes, num_dests);
                        }
                    }
                }
                noc_async_write_barrier();
                gamma_mc.sender(noc).send_signal();
                beta_mc.sender(noc).send_signal();
            } else {
                gamma_mc.sender(noc).send(gamma_base, gamma_base, affine_cb_bytes);
                beta_mc.sender(noc).send(beta_base, beta_base, affine_cb_bytes);
            }
        }
        gamma_tiles.push_back(tensor_w_tiles);
        beta_tiles.push_back(tensor_w_tiles);
    };
    if (is_sender && !send_late) {
        issue_affine_reads(gamma_accessor, gamma_tiles);
        issue_affine_reads(beta_accessor, beta_tiles);
        complete_and_send();
    }
    bool send_pending = is_sender && send_late;

    auto receive_affine = [&]() {
        MaybeDeviceZoneScope("reader_affine");
        gamma_tiles.reserve_back(tensor_w_tiles);
        beta_tiles.reserve_back(tensor_w_tiles);
        // A 1-core call has no receiver role, so the receiver face is not instantiable there.
        if constexpr (gamma_mc.receiver_available) {
            gamma_mc.receiver(noc).receive();
            beta_mc.receiver(noc).receive();
        }
        gamma_tiles.push_back(tensor_w_tiles);
        beta_tiles.push_back(tensor_w_tiles);
    };
    if (receive_early && !is_sender) {
        receive_affine();
    }

    bool affine_pending = !is_sender && !receive_early;
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
                issue_affine_reads(gamma_accessor, gamma_tiles);
                issue_affine_reads(beta_accessor, beta_tiles);
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
        if (affine_pending) {
            receive_affine();
            affine_pending = false;
        }
    }
}
