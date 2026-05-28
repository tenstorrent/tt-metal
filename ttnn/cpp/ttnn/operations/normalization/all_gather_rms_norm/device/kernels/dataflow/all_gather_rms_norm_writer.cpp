// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer dataflow kernel for the generic fused all_gather_rms_norm op.
//
// For ring_size == 1 (single device) it simply writes the normalized output tiles from cb_output back to
// interleaved DRAM. For ring_size > 1 it must additionally all-gather the per-device stats over fabric
// (send cb_local_stats to the ring neighbors' cb_gathered_stats slots, then the semaphore handshake) --
// that path is a TODO; see all_gather_rms_norm_program_factory.cpp and broadcast_tile_writer.cpp.
//
// Ported from: layernorm_distributed/.../writer_unary_interleaved_start_id_blocked.cpp (output write).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // Compile-time args (order must match all_gather_rms_norm_program_factory.cpp: writer_ct_args).
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t ring_size = get_compile_time_arg_val(2);
    constexpr uint32_t cb_local_stats = get_compile_time_arg_val(3);
    constexpr uint32_t cb_gathered_stats = get_compile_time_arg_val(4);
    constexpr uint32_t cb_packet_header = get_compile_time_arg_val(5);
    constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(6);
    constexpr uint32_t num_links = get_compile_time_arg_val(7);
    constexpr auto dst_args = TensorAccessorArgs<8>();

    (void)cb_local_stats;
    (void)cb_gathered_stats;
    (void)cb_packet_header;
    (void)num_packet_headers_storable;
    (void)num_links;

    // Runtime args.
    uint32_t ai = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(ai++);
    const uint32_t tile_offset = get_arg_val<uint32_t>(ai++);

    if constexpr (ring_size > 1) {
        // TODO(fabric): before writing the output, all-gather the per-device stats:
        //   - open the fabric connection(s) from the appended connection RT args (start at index 3),
        //   - send cb_local_stats to each ring neighbor's cb_gathered_stats slot using packet headers
        //     from cb_packet_header,
        //   - atomic-inc the out-ready semaphore and wait until it reaches ring_size * num_links,
        //   - close the connection(s).
        // See broadcast_tile_writer.cpp for the canonical pattern.
    }

    const auto s = TensorAccessor(dst_args, dst_addr);
    Noc noc;
    CircularBuffer cb_out_buf(cb_out);

    const uint32_t tile_bytes = get_tile_size(cb_out);
    uint32_t tile_id = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += blk) {
        cb_out_buf.wait_front(blk);
        uint32_t write_offset = 0;
        for (uint32_t j = 0; j < blk; j++) {
            noc.async_write(
                use<CircularBuffer::AddrSelector::READ_PTR>(cb_out_buf),
                s,
                tile_bytes,
                {.offset_bytes = write_offset},
                {.page_id = tile_id});
            tile_id++;
            write_offset += tile_bytes;
        }
        noc.async_write_barrier();
        cb_out_buf.pop_front(blk);
    }
}
