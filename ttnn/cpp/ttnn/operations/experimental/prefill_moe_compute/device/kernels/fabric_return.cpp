// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fabric return kernel: gathers result rows from TILE_LAYOUT out_bufs
// and writes unscaled rows to output slots (local) or sends to remote
// staging buffer via fabric (remote).
//
// This is a PURE DATA MOVEMENT kernel — no floating-point math.
// Scaling and accumulation are done externally via ttnn.mul + ttnn.sum
// on compute cores (matching the selective_reduce_combine pattern).
//
// Runs on return core (2, grid_y) as dm1 (RISCV_1).
//
// Output layout: [NUM_EXPERTS, 1, P, D] ROW_MAJOR
//   Page dest_page = e * P + dest_token (pre-computed by Python)
//
// Metadata format (packed uint32 words, stored in DRAM tensor):
//   Per expert e:
//     out_buf_addr, M_e, [src_row, dest_device, dest_page, recv_slot_id] * M_e
//
// Runtime args:
//   [0] output_addr
//   [1] D_tiles
//   [2] D_bytes
//   [3] num_experts
//   [4] my_device_id
//   [5] recv_core_phys_x
//   [6] recv_core_phys_y
//   [7] staging_addr         (DRAM address of recv_staging_buf, 0 if no fabric)
//   [8] metadata_addr        (DRAM address of metadata tensor)
//   [9] metadata_word_count  (number of uint32 metadata words)
//   [10+] FabricConnectionManager RT args (if ENABLE_FABRIC_SEND)
//
// Compile-time args:
//   TensorAccessorArgs<0>: out_buf accessor
//   TensorAccessorArgs<1>: output accessor
//   TensorAccessorArgs<2>: metadata tensor accessor
//   TensorAccessorArgs<3>: staging accessor (if ENABLE_FABRIC_SEND)
//
// Circular buffers:
//   CB0: tile batch buffer (D_tiles tiles for batch reads)
//   CB1: row extraction buffer (D_bytes)
//   CB3: metadata buffer
//
// Semaphores:
//   SEM_EXPERT_READY (id=3): Incremented by reader leader after each expert's
//     barrier B. Return kernel waits for e+1 before processing expert e,
//     enabling pipelined overlap with the next expert's compute.
//   SEM_RECV (id=4): Incremented on remote recv core via fabric.

#include "api/dataflow/dataflow_api.h"

#ifdef ENABLE_FABRIC_SEND
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

using namespace tt::tt_fabric::linear::experimental;
#endif

inline void extract_row_from_tile(uint32_t tile_l1, uint32_t dst, uint32_t row) {
    uint32_t face_pair = row >> 4;
    uint32_t local_row = row & 0xF;
    uint32_t face0_offset = (face_pair * 2) * 512 + local_row * 32;
    uint32_t face1_offset = (face_pair * 2 + 1) * 512 + local_row * 32;

    volatile uint32_t* d0 = reinterpret_cast<volatile uint32_t*>(dst);
    volatile uint32_t* s0 = reinterpret_cast<volatile uint32_t*>(tile_l1 + face0_offset);
    for (uint32_t w = 0; w < 8; ++w) {
        d0[w] = s0[w];
    }

    volatile uint32_t* d1 = reinterpret_cast<volatile uint32_t*>(dst + 32);
    volatile uint32_t* s1 = reinterpret_cast<volatile uint32_t*>(tile_l1 + face1_offset);
    for (uint32_t w = 0; w < 8; ++w) {
        d1[w] = s1[w];
    }
}

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t D_tiles = get_arg_val<uint32_t>(1);
    const uint32_t D_bytes = get_arg_val<uint32_t>(2);
    const uint32_t num_experts = get_arg_val<uint32_t>(3);
    const uint32_t my_device_id = get_arg_val<uint32_t>(4);
    const uint32_t recv_core_phys_x = get_arg_val<uint32_t>(5);
    const uint32_t recv_core_phys_y = get_arg_val<uint32_t>(6);
    const uint32_t staging_addr = get_arg_val<uint32_t>(7);
    const uint32_t metadata_addr = get_arg_val<uint32_t>(8);
    const uint32_t metadata_word_count = get_arg_val<uint32_t>(9);

    constexpr auto outbuf_args = TensorAccessorArgs<0>();
    constexpr auto output_args = TensorAccessorArgs<1>();
    constexpr auto meta_args = TensorAccessorArgs<2>();

    constexpr uint32_t cb_tile = 0;
    constexpr uint32_t cb_row = 1;
    constexpr uint32_t cb_meta = 3;
    constexpr uint32_t SEM_EXPERT_READY = 3;
    constexpr uint32_t SEM_RECV = 4;

    const uint32_t tile_page_bytes = get_local_cb_interface(cb_tile).fifo_page_size;
    const auto output_accessor = TensorAccessor(output_args, output_addr, D_bytes);

    // Reserve L1 buffers — CB0 holds D_tiles tiles for batch reads
    cb_reserve_back(cb_tile, D_tiles);
    const uint32_t tile_buf = get_write_ptr(cb_tile);
    cb_reserve_back(cb_row, 1);
    const uint32_t row_l1 = get_write_ptr(cb_row);

    // Read metadata from DRAM into L1
    cb_reserve_back(cb_meta, 1);
    const uint32_t meta_l1 = get_write_ptr(cb_meta);
    const uint32_t meta_page_bytes = metadata_word_count * sizeof(uint32_t);
    const auto meta_accessor = TensorAccessor(meta_args, metadata_addr, meta_page_bytes);
    noc_async_read_page(0, meta_accessor, meta_l1);
    noc_async_read_barrier();

    // Metadata is now in L1 — interpret as uint32 array
    volatile uint32_t* meta = reinterpret_cast<volatile uint32_t*>(meta_l1);

    // Per-expert semaphore: pipelined with compute (wait for each expert individually)
    volatile tt_l1_ptr uint32_t* ready_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_EXPERT_READY));

#ifdef ENABLE_FABRIC_SEND
    constexpr uint32_t MAX_FABRIC_CHUNK = 4096;

    constexpr auto staging_args = TensorAccessorArgs<3>();
    const auto staging_accessor = TensorAccessor(staging_args, staging_addr, D_bytes);

    // Fabric args at fixed RT arg offset 10
    std::size_t fabric_arg_idx = 10;
    auto fabric_connection = FabricConnectionManager::build_from_args(fabric_arg_idx);
    fabric_connection.open();

    // Get both senders (null if not available for edge devices)
    auto* forward_sender =
        fabric_connection.has_forward_connection() ? &fabric_connection.get_forward_connection() : nullptr;
    auto* backward_sender =
        fabric_connection.has_backward_connection() ? &fabric_connection.get_backward_connection() : nullptr;

    auto pkt_hdr_data = PacketHeaderPool::allocate_header();
    auto pkt_hdr_sem = PacketHeaderPool::allocate_header();

    // Remote SEM_RECV NOC address (same physical coords on all devices)
    uint64_t remote_recv_sem_noc_addr = safe_get_noc_addr(
        static_cast<uint8_t>(recv_core_phys_x), static_cast<uint8_t>(recv_core_phys_y), get_semaphore(SEM_RECV), 0);
#endif

    // Process experts — metadata parsed from L1
    // Format per expert: out_buf_addr, M_e, [src_row, dest_device, dest_page, recv_slot_id] * M_e
    uint32_t meta_idx = 0;

    for (uint32_t e = 0; e < num_experts; ++e) {
        // Wait for this expert's output to be ready (pipelined with next expert's compute)
        noc_semaphore_wait_min(ready_sem, e + 1);

        uint32_t out_buf_addr = meta[meta_idx++];
        uint32_t M_e = meta[meta_idx++];

        if (M_e == 0) {
            continue;
        }

        const auto outbuf_accessor = TensorAccessor(outbuf_args, out_buf_addr, tile_page_bytes);

        // Batch-read ALL D_tiles tiles for tile_row=0 into L1 (single barrier).
        // Since P <= 32, all tokens fit in one tile row (tile_row = src_row >> 5 = 0).
        // This replaces per-token tile reads: 4 experts * 90 tiles vs 128 tokens * 90 tiles.
        for (uint32_t c = 0; c < D_tiles; ++c) {
            noc_async_read_page(c, outbuf_accessor, tile_buf + c * tile_page_bytes);
        }
        noc_async_read_barrier();

        uint32_t tokens_base = meta_idx;

        for (uint32_t i = 0; i < M_e; ++i) {
            uint32_t src_row = meta[tokens_base + i * 4 + 0];
            uint32_t dest_device = meta[tokens_base + i * 4 + 1];
            uint32_t dest_page = meta[tokens_base + i * 4 + 2];
            // recv_slot_id at tokens_base + i * 4 + 3 (used only for remote sends)

            uint32_t row_in_tile = src_row & 0x1F;

            // Extract row from L1 tiles (no DRAM read — tiles already cached)
            for (uint32_t c = 0; c < D_tiles; ++c) {
                extract_row_from_tile(tile_buf + c * tile_page_bytes, row_l1 + c * 64, row_in_tile);
            }

            if (dest_device == my_device_id) {
                // Simple write — no read, no scale, no accumulate
                noc_async_write_page(dest_page, output_accessor, row_l1);
                noc_async_write_barrier();
            }
#ifdef ENABLE_FABRIC_SEND
            else {
                uint32_t recv_slot_id = meta[tokens_base + i * 4 + 3];

                // Compute direction and hops for 1xN linear mesh
                uint8_t num_hops;
                auto* sender = forward_sender;  // default
                if (dest_device > my_device_id) {
                    num_hops = static_cast<uint8_t>(dest_device - my_device_id);
                    sender = forward_sender;
                } else {
                    num_hops = static_cast<uint8_t>(my_device_id - dest_device);
                    sender = backward_sender;
                }

                // Write raw (unscaled) row to remote staging buffer at recv_slot_id
                uint64_t dest_noc_addr =
                    tt::tt_fabric::linear::addrgen_detail::get_noc_address(staging_accessor, recv_slot_id);

                uint32_t bytes_sent = 0;
                while (bytes_sent < D_bytes) {
                    uint32_t chunk = D_bytes - bytes_sent;
                    if (chunk > MAX_FABRIC_CHUNK) {
                        chunk = MAX_FABRIC_CHUNK;
                    }

                    fabric_unicast_noc_unicast_write(
                        sender,
                        pkt_hdr_data,
                        row_l1 + bytes_sent,
                        chunk,
                        tt::tt_fabric::NocUnicastCommandHeader{dest_noc_addr + bytes_sent},
                        num_hops);
                    noc_async_write_barrier();
                    bytes_sent += chunk;
                }

                // Signal remote recv core
                fabric_unicast_noc_unicast_atomic_inc(
                    sender,
                    pkt_hdr_sem,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_recv_sem_noc_addr, 1, true},
                    num_hops);
                noc_async_write_barrier();
            }
#endif
        }
        meta_idx += M_e * 4;
    }

#ifdef ENABLE_FABRIC_SEND
    fabric_connection.close();
#endif

    cb_push_back(cb_tile, D_tiles);
    cb_pop_front(cb_tile, D_tiles);
    cb_push_back(cb_row, 1);
    cb_pop_front(cb_row, 1);
    cb_push_back(cb_meta, 1);
    cb_pop_front(cb_meta, 1);
}
