// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Reader kernel for the row-major dispatch path — runs on the sender core's
// RISCV_1, paired with writer_dispatch.cpp on RISCV_0.  (The tile-layout path
// uses untilize cores instead and has no sender-side reader kernel.)
//
// Streams row-major input/indices from DRAM and, per routed token, either
// writes it straight to local DRAM or hands (route_info, payload, metadata) to the
// writer for a fabric send to the destination device.
//
// Tokens [token_start_idx, token_end_idx) are processed in batches of
// read_batch_size.  Reads are pipelined: the next batch's DRAM reads are issued
// before the current batch's write barrier so the read and write NOC channels
// overlap.  (The input/indices scratch is a single region reused per
// batch, not a FIFO — see the reservation note below.)
//
// Startup: load offsets[] (per-expert DRAM page allocators) and the read-only
// dispatch_table[] (expert → destination chip, or -1) into local L1 scratch.
//
// For each batch, for each (token, top-k expert) routed to this dispatch core:
//   - Allocate a destination DRAM page from offsets[expert]; if the dispatch
//     buffer is full, bump the counter and drop the token (prevents OOB writes).
//   - Local (expert maps to this device): write payload + metadata directly to
//     local DRAM.
//   - Cross-device: push route_info (route/distance/page), payload, and metadata
//     to the writer CBs; the writer forwards them over the fabric.
//
// After the last batch: push ROUTE_INFO_SENTINEL to the writer so it stops.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#define ENABLE_DISPATCH_DEBUG 0

#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_DISPATCH(...)
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile Time Args =====
    // CB IDs (indices 0-8)
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_payload_for_writer_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_metadata_for_writer_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_metadata_temp_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_dispatch_table_id = get_compile_time_arg_val(8);

    // Page counts (indices 9-14)
    constexpr uint32_t input_pages = get_compile_time_arg_val(9);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(10);
    constexpr uint32_t offsets_pages = get_compile_time_arg_val(11);
    constexpr uint32_t output_pages = get_compile_time_arg_val(12);
    constexpr uint32_t metadata_pages = get_compile_time_arg_val(13);
    constexpr uint32_t dispatch_table_pages = get_compile_time_arg_val(14);

    // Page sizes (indices 15-20; offsets/dispatch_table page sizes unused by the reader)
    constexpr uint32_t input_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(16);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(19);

    // Operation parameters (indices 21-27)
    constexpr uint32_t num_devices = get_compile_time_arg_val(21);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(22);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(23);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(24);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(25);
    constexpr uint32_t metadata_len = get_compile_time_arg_val(26);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(27);

    // Mesh information (indices 28-32)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(28);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(29);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(30);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(31);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(32);

    // Aligned page sizes (indices 33-38)
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(33);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(34);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(36);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(37);
    constexpr uint32_t aligned_dispatch_table_page_size = get_compile_time_arg_val(38);

    // Fabric configuration (indices 39-42)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(39);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(40);
    constexpr uint32_t num_links = get_compile_time_arg_val(41);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(42);

    // Batch configuration (index 43)
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(43);

    // Total dispatch buffer token capacity (shared across all local experts).
    constexpr uint32_t max_dispatch_buffer_token_size = get_compile_time_arg_val(44);

    // TensorAccessorArgs for all 6 tensors (starting at index 45, after the
    // trailing max_dispatch_buffer_token_size arg).
    constexpr auto input_args = TensorAccessorArgs<45>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto dispatch_table_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();

#ifdef HAS_PADDING_CONFIG
    // padding_config TensorAccessorArgs + scratch CB id are appended LAST in the compile-time args
    // (after the 7 tensor accessors) so they never shift the existing index layout.
    constexpr auto padding_cfg_args = TensorAccessorArgs<dispatch_table_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_padding_config_id =
        get_compile_time_arg_val(padding_cfg_args.next_compile_time_args_offset());
#endif

    // ===== Runtime Args =====
    uint32_t rt_args = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t offsets_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t dispatch_table_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t cross_device_semaphore_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t dispatch_core_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t num_dispatch_cores = get_arg_val<uint32_t>(rt_args++);
    uint32_t core_mask = num_dispatch_cores - 1;

#ifdef HAS_PADDING_CONFIG
    // padding_config base address sits right after the 13 base runtime args (fixed index 13).
    uint32_t padding_config_address = get_arg_val<uint32_t>(rt_args++);
#endif

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t col = linearized_mesh_coord % mesh_cols;
    constexpr uint32_t device_begin_idx = axis == ReplicateGroup::COLS ? col : row * mesh_cols;
    constexpr uint32_t device_end_idx =
        (axis == ReplicateGroup::COLS) ? (col + mesh_rows * mesh_cols) : (row * mesh_cols + mesh_cols);
    constexpr uint32_t device_stride = axis == ReplicateGroup::COLS ? mesh_cols : 1;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t device_begin_idx = 0;
    constexpr uint32_t device_end_idx = num_devices;
    constexpr uint32_t device_stride = 1;
#endif

    DPRINT_DISPATCH(
        "Reader kernel: tokens=[{},{}] dispatch_core={}/{}\n",
        token_start_idx,
        token_end_idx,
        dispatch_core_idx,
        num_dispatch_cores);

    // Read offsets tensor into local scratch
    const auto offsets_addr_gen = TensorAccessor(offsets_args, offsets_tensor_address);
    cb_reserve_back(cb_offsets_id, offsets_pages);
    uint32_t offsets_base_addr = get_write_ptr(cb_offsets_id);
    for (uint32_t i = 0; i < offsets_pages; i++) {
        noc_async_read_page(i, offsets_addr_gen, offsets_base_addr + i * aligned_offsets_page_size);
    }
    noc_async_read_barrier();
    tt_l1_ptr uint32_t* offsets = reinterpret_cast<tt_l1_ptr uint32_t*>(offsets_base_addr);

    // Read dispatch table into local scratch
    const auto dispatch_table_addr_gen = TensorAccessor(dispatch_table_args, dispatch_table_tensor_address);
    cb_reserve_back(cb_dispatch_table_id, dispatch_table_pages);
    uint32_t dispatch_table_base_addr = get_write_ptr(cb_dispatch_table_id);
    for (uint32_t i = 0; i < dispatch_table_pages; i++) {
        noc_async_read_page(
            i, dispatch_table_addr_gen, dispatch_table_base_addr + i * aligned_dispatch_table_page_size);
    }
    noc_async_read_barrier();
    tt_l1_ptr int32_t* expert_dispatch_table = reinterpret_cast<tt_l1_ptr int32_t*>(dispatch_table_base_addr);

#ifdef HAS_PADDING_CONFIG
    // Bound the token loop to this device's real (unpadded) tokens. The per-device padding_config
    // row is [local_real_tokens, pad_side]; for right padding (pad_side == 0) we shrink token_end_idx
    // to the next 32-aligned multiple of the real count. Left padding keeps the full range
    // (unsupported fast path). The batch loop below derives total_batches from token_end_idx, so this
    // single override shrinks the work. Padded tokens in the last partial batch keep their sentinel
    // expert indices, so expert_dispatch_table[sentinel] == -1 drops them.
    {
        const auto padding_cfg_gen = TensorAccessor(padding_cfg_args, padding_config_address);
        cb_reserve_back(cb_padding_config_id, 1);
        uint32_t pc_l1 = get_write_ptr(cb_padding_config_id);
        noc_async_read_page(0, padding_cfg_gen, pc_l1);
        noc_async_read_barrier();
        tt_l1_ptr uint32_t* pc = reinterpret_cast<tt_l1_ptr uint32_t*>(pc_l1);
        uint32_t real_count = pc[0];
        uint32_t pad_side = pc[1];
        if (pad_side == 0) {
            uint32_t rounded = ((real_count + 31u) / 32u) * 32u;
            if (rounded < token_end_idx) {
                token_end_idx = rounded;
            }
        }
    }
#endif

    // Reserve scratch space once — these CBs are not used as FIFOs. Each batch
    // overwrites the same region at offsets [0, batch_count) without push/pop.
    // DRAM reads are batched to saturate DRAM bandwidth, while the L1-to-writer-CB
    // copies below are done one page at a time — this avoids CB FIFO pointer wrapping
    cb_reserve_back(cb_indices_id, read_batch_size);
    uint32_t indices_base = get_write_ptr(cb_indices_id);
    cb_reserve_back(cb_input_id, read_batch_size);
    uint32_t input_base = get_write_ptr(cb_input_id);
    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address);
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address);
    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address);

    cb_reserve_back(cb_metadata_temp_id, 1);
    uint32_t metadata_temp_addr = get_write_ptr(cb_metadata_temp_id);

    // Prefetch first batch of DRAM reads
    uint32_t first_batch_end =
        (token_start_idx + read_batch_size < token_end_idx) ? token_start_idx + read_batch_size : token_end_idx;
    uint32_t first_batch_count = first_batch_end - token_start_idx;
    if (first_batch_count > 0) {
        for (uint32_t t = 0; t < first_batch_count; t++) {
            noc_async_read_page(token_start_idx + t, input_addr_gen, input_base + t * aligned_input_page_size);
            noc_async_read_page(token_start_idx + t, indices_addr_gen, indices_base + t * aligned_indices_page_size);
        }
        noc_async_read_barrier();
    }

    uint32_t total_batches = (token_end_idx - token_start_idx + read_batch_size - 1) / read_batch_size;
    for (uint32_t B = 0; B < total_batches; B++) {
        uint32_t batch_start = token_start_idx + B * read_batch_size;
        uint32_t batch_end =
            (batch_start + read_batch_size < token_end_idx) ? batch_start + read_batch_size : token_end_idx;
        uint32_t batch_count = batch_end - batch_start;
        bool batch_did_local_write = false;

        for (uint32_t t = 0; t < batch_count; t++) {
            uint32_t token_idx = batch_start + t;
            uint32_t token_input_addr = input_base + t * aligned_input_page_size;
            tt_l1_ptr uint16_t* indices =
                reinterpret_cast<tt_l1_ptr uint16_t*>(indices_base + t * aligned_indices_page_size);

            for (uint32_t k = 0; k < num_experts_per_tok; ++k) {
                uint32_t routed_expert = (uint32_t)indices[k];

                // Skip experts not owned by this dispatch core (low bits of the expert id
                // select the dispatch core) and experts the table maps nowhere (-1).
                if (((uint32_t)routed_expert & core_mask) != dispatch_core_idx) {
                    continue;
                }

                auto expert_chip_og = expert_dispatch_table[routed_expert];
                if (expert_chip_og == -1) {
                    continue;
                }
                auto expert_chip = device_begin_idx + expert_chip_og * device_stride;
                // Allocate this token's destination DRAM page from the expert's counter.
                auto& offset = offsets[routed_expert];
                if (offset >= max_dispatch_buffer_token_size) {
                    // Token would overflow the dispatch buffer - skip to prevent
                    // out-of-bounds DRAM writes that corrupt memory and cause hangs.
                    offset++;
                    continue;
                }
                auto page_idx = offset;

                // Local: expert lives on this device — write payload + metadata straight to DRAM.
                if (expert_chip == linearized_mesh_coord) {
                    {
                        // Per-token metadata (3 × int32): [src chip, global token idx, top-k slot].
                        // Staged in scratch, then written to the payload's DRAM page.
                        volatile tt_l1_ptr int32_t* metadata =
                            reinterpret_cast<volatile tt_l1_ptr int32_t*>(metadata_temp_addr);
                        metadata[0] = linearized_mesh_coord;
                        metadata[1] = token_idx;
                        metadata[2] = k;

                        noc_async_write_page(page_idx, output_addr_gen, token_input_addr);
                        noc_async_write_page(page_idx, metadata_addr_gen, metadata_temp_addr);
                        noc_async_writes_flushed();
                        batch_did_local_write = true;
                    }
                } else {
                    // Cross-device: stage route_info, payload and metadata into the writer's CBs;
                    // the writer (RISCV_0) forwards them over the fabric.  route/distance tell the
                    // fabric writer where to send.
                    if constexpr (is_1d_topology<topology>()) {
                        uint32_t route = get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);
                        uint32_t distance =
                            manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);

                        // route_info layout: [0]=route, [1]=distance, [2]=page_idx, [3]=expert_chip.
                        // route + distance are consumed by the 1D writer; under FABRIC_2D the writer
                        // recomputes the EDM direction from route_info[3] and ignores slots [0..1].
                        // All four slots are written unconditionally
                        cb_reserve_back(cb_route_info_id, 1);
                        volatile tt_l1_ptr uint32_t* route_info =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route_info_id));
                        route_info[0] = route;
                        route_info[1] = distance;
                        route_info[2] = page_idx;
                        route_info[3] = expert_chip;
                        cb_push_back(cb_route_info_id, 1);

                        cb_reserve_back(cb_payload_for_writer_id, 1);
                        uint32_t payload_dst = get_write_ptr(cb_payload_for_writer_id);
                        noc_async_read(get_noc_addr(token_input_addr), payload_dst, aligned_input_page_size);
                        noc_async_read_barrier();
                        cb_push_back(cb_payload_for_writer_id, 1);

                        cb_reserve_back(cb_metadata_for_writer_id, 1);
                        volatile tt_l1_ptr int32_t* meta_dst =
                            reinterpret_cast<volatile tt_l1_ptr int32_t*>(get_write_ptr(cb_metadata_for_writer_id));
                        meta_dst[0] = linearized_mesh_coord;
                        meta_dst[1] = token_idx;
                        meta_dst[2] = k;
                        cb_push_back(cb_metadata_for_writer_id, 1);
                    }
                }

                offset++;
            }
        }

        // Issue next batch DRAM reads BEFORE write barrier to overlap read/write NOC channels
        uint32_t next_batch_start = batch_start + read_batch_size;
        bool has_next_batch = (next_batch_start < token_end_idx);
        if (has_next_batch) {
            uint32_t next_batch_end = (next_batch_start + read_batch_size < token_end_idx)
                                          ? next_batch_start + read_batch_size
                                          : token_end_idx;
            uint32_t next_batch_count = next_batch_end - next_batch_start;
            for (uint32_t t = 0; t < next_batch_count; t++) {
                noc_async_read_page(next_batch_start + t, input_addr_gen, input_base + t * aligned_input_page_size);
                noc_async_read_page(
                    next_batch_start + t, indices_addr_gen, indices_base + t * aligned_indices_page_size);
            }
        }

        if (batch_did_local_write) {
            noc_async_write_barrier();
        }

        if (has_next_batch) {
            noc_async_read_barrier();
        }
    }

    // Teardown: all batches done — push a ROUTE_INFO_SENTINEL route_info entry so the writer
    // (RISCV_0) knows no more tokens are coming and stops forwarding / exits.
    cb_reserve_back(cb_route_info_id, 1);
    volatile tt_l1_ptr uint32_t* sentinel_route_info =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route_info_id));
    sentinel_route_info[0] = ROUTE_INFO_SENTINEL;
    sentinel_route_info[1] = 0;
    sentinel_route_info[2] = 0;
    sentinel_route_info[3] = 0;
    cb_push_back(cb_route_info_id, 1);
}
