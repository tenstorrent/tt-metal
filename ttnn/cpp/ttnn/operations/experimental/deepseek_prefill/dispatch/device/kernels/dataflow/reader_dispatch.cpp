// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#define ENABLE_DISPATCH_DEBUG 0

#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH DPRINT
#else
#define DPRINT_DISPATCH \
    if (0)              \
    DebugPrinter()
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile Time Args =====
    // CB IDs (indices 0-9)
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_payload_for_writer_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_metadata_for_writer_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_metadata_temp_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_dispatch_table_id = get_compile_time_arg_val(9);

    // Page counts (indices 10-16)
    constexpr uint32_t input_pages = get_compile_time_arg_val(10);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(11);
    constexpr uint32_t weights_pages = get_compile_time_arg_val(12);
    constexpr uint32_t offsets_pages = get_compile_time_arg_val(13);
    constexpr uint32_t output_pages = get_compile_time_arg_val(14);
    constexpr uint32_t metadata_pages = get_compile_time_arg_val(15);
    constexpr uint32_t dispatch_table_pages = get_compile_time_arg_val(16);

    // Page sizes (indices 17-23)
    constexpr uint32_t input_page_size = get_compile_time_arg_val(17);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t weights_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(20);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(21);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(22);
    constexpr uint32_t dispatch_table_page_size = get_compile_time_arg_val(23);

    // Operation parameters (indices 24-31)
    constexpr uint32_t num_devices = get_compile_time_arg_val(24);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(25);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(26);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(27);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(28);
    constexpr uint32_t metadata_len = get_compile_time_arg_val(29);
    constexpr uint32_t max_dispatched_tokens_per_expert = get_compile_time_arg_val(30);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(31);

    // Mesh information (indices 32-36)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(32);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(33);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(34);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(35);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(36);

    // Aligned page sizes (indices 37-43)
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(37);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(38);
    constexpr uint32_t aligned_weights_page_size = get_compile_time_arg_val(39);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(40);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(41);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(42);
    constexpr uint32_t aligned_dispatch_table_page_size = get_compile_time_arg_val(43);

    // Fabric configuration (indices 44-47)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(44);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(45);
    constexpr uint32_t num_links = get_compile_time_arg_val(46);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(47);

    // TensorAccessorArgs for all 7 tensors (starting at index 48)
    constexpr auto input_args = TensorAccessorArgs<48>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto dispatch_table_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();

    // ===== Runtime Args =====
    uint32_t rt_args = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t weights_tensor_address = get_arg_val<uint32_t>(rt_args++);
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

    DPRINT_DISPATCH << "Reader kernel: tokens=[" << token_start_idx << "," << token_end_idx << ")"
                    << " dispatch_core=" << dispatch_core_idx << "/" << num_dispatch_cores << ENDL();

    // Read offsets into local scratch
    const auto offsets_addr_gen = TensorAccessor(offsets_args, offsets_tensor_address, offsets_page_size);
    cb_reserve_back(cb_offsets_id, offsets_pages);
    uint32_t offsets_base_addr = get_write_ptr(cb_offsets_id);
    for (uint32_t i = 0; i < offsets_pages; i++) {
        noc_async_read_page(i, offsets_addr_gen, offsets_base_addr + i * aligned_offsets_page_size);
    }
    noc_async_read_barrier();
    int32_t* offsets = (int32_t*)offsets_base_addr;

    // Read dispatch table into local scratch
    const auto dispatch_table_addr_gen =
        TensorAccessor(dispatch_table_args, dispatch_table_tensor_address, dispatch_table_page_size);
    cb_reserve_back(cb_dispatch_table_id, dispatch_table_pages);
    uint32_t dispatch_table_base_addr = get_write_ptr(cb_dispatch_table_id);
    for (uint32_t i = 0; i < dispatch_table_pages; i++) {
        noc_async_read_page(
            i, dispatch_table_addr_gen, dispatch_table_base_addr + i * aligned_dispatch_table_page_size);
    }
    noc_async_read_barrier();
    int32_t* expert_dispatch_table = (int32_t*)dispatch_table_base_addr;

    // Set up batched scratch buffers
    constexpr uint32_t read_batch_size = 8;
    cb_reserve_back(cb_indices_id, read_batch_size);
    uint32_t indices_base = get_write_ptr(cb_indices_id);
    cb_reserve_back(cb_weights_id, read_batch_size);
    uint32_t weights_base = get_write_ptr(cb_weights_id);
    cb_reserve_back(cb_input_id, read_batch_size);
    uint32_t input_base = get_write_ptr(cb_input_id);

    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address, aligned_input_page_size);
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address, aligned_indices_page_size);
    const auto weights_addr_gen = TensorAccessor(weights_args, weights_tensor_address, aligned_weights_page_size);
    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address, aligned_output_page_size);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, aligned_metadata_page_size);

    // Reserve metadata temp for constructing metadata locally
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
            noc_async_read_page(token_start_idx + t, weights_addr_gen, weights_base + t * aligned_weights_page_size);
        }
        noc_async_read_barrier();
    }

    // Main batch loop
    for (uint32_t batch_start = token_start_idx; batch_start < token_end_idx; batch_start += read_batch_size) {
        uint32_t batch_end =
            (batch_start + read_batch_size < token_end_idx) ? batch_start + read_batch_size : token_end_idx;
        uint32_t batch_count = batch_end - batch_start;
        bool batch_did_local_write = false;

        for (uint32_t t = 0; t < batch_count; t++) {
            uint32_t token_idx = batch_start + t;
            uint32_t input_scratch_addr = input_base + t * aligned_input_page_size;
            int32_t* indices = (int32_t*)(indices_base + t * aligned_indices_page_size);
            uint16_t* weights = (uint16_t*)(weights_base + t * aligned_weights_page_size);

            for (uint32_t k = 0; k < num_experts_per_tok; ++k) {
                auto routed_expert = indices[k];

                if (((uint32_t)routed_expert & core_mask) != dispatch_core_idx) {
                    continue;
                }

                auto expert_chip_og = expert_dispatch_table[routed_expert];
                if (expert_chip_og == -1) {
                    continue;
                }
                auto expert_chip = device_begin_idx + expert_chip_og * device_stride;
                auto expert_index_within_chip = routed_expert % experts_per_chip;
                auto& offset = offsets[routed_expert];
                auto page_idx = expert_index_within_chip * max_dispatched_tokens_per_expert + offset;

                // Construct metadata
                volatile tt_l1_ptr int32_t* metadata =
                    reinterpret_cast<volatile tt_l1_ptr int32_t*>(metadata_temp_addr);
                metadata[0] = linearized_mesh_coord;
                metadata[1] = token_idx;
                metadata[2] = k;
                metadata[3] = routed_expert;
                metadata[4] = static_cast<int16_t>(weights[k]);

                if (expert_chip == linearized_mesh_coord) {
                    noc_async_write_page(page_idx, output_addr_gen, input_scratch_addr);
                    noc_async_write_page(page_idx, metadata_addr_gen, metadata_temp_addr);
                    noc_async_writes_flushed();
                    batch_did_local_write = true;
                } else {
                    if constexpr (is_1d_topology<topology>()) {
                        uint32_t route = get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);
                        uint32_t distance =
                            manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);

                        // Push route info to writer
                        cb_reserve_back(cb_route_info_id, 1);
                        volatile tt_l1_ptr uint32_t* route_info =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route_info_id));
                        route_info[0] = route;
                        route_info[1] = distance;
                        route_info[2] = page_idx;
                        route_info[3] = 0;
                        cb_push_back(cb_route_info_id, 1);

                        // Push payload to writer
                        cb_reserve_back(cb_payload_for_writer_id, 1);
                        uint32_t payload_dst = get_write_ptr(cb_payload_for_writer_id);
                        noc_async_read(get_noc_addr(input_scratch_addr), payload_dst, aligned_input_page_size);
                        noc_async_read_barrier();
                        cb_push_back(cb_payload_for_writer_id, 1);

                        // Push metadata to writer
                        cb_reserve_back(cb_metadata_for_writer_id, 1);
                        uint32_t metadata_dst = get_write_ptr(cb_metadata_for_writer_id);
                        noc_async_read(get_noc_addr(metadata_temp_addr), metadata_dst, aligned_metadata_page_size);
                        noc_async_read_barrier();
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
                noc_async_read_page(
                    next_batch_start + t, weights_addr_gen, weights_base + t * aligned_weights_page_size);
            }
        }

        if (batch_did_local_write) {
            noc_async_write_barrier();
        }

        if (has_next_batch) {
            noc_async_read_barrier();
        }
    }

    // Push sentinel to signal writer that all dispatches are done
    cb_reserve_back(cb_route_info_id, 1);
    volatile tt_l1_ptr uint32_t* route_info =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route_info_id));
    route_info[0] = ROUTE_INFO_SENTINEL;
    route_info[1] = 0;
    route_info[2] = 0;
    route_info[3] = 0;
    cb_push_back(cb_route_info_id, 1);
}
