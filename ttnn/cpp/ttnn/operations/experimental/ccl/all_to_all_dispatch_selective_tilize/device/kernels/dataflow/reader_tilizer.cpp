// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"

using namespace ttnn::operations::ccl::common;

// Helper to get multicast NOC address with proper coordinate ordering for NOC 0 vs NOC 1.
// NOC 0: start = (min_x, min_y), end = (max_x, max_y)
// NOC 1: start = (max_x, max_y), end = (min_x, min_y) - coordinates need to be swapped
FORCE_INLINE uint64_t get_safe_multicast_noc_addr(
    uint32_t noc_x_start,
    uint32_t noc_y_start,
    uint32_t noc_x_end,
    uint32_t noc_y_end,
    uint32_t addr,
    uint8_t noc = noc_index) {
    if (noc == 0) {
        return get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, addr, noc);
    } else {
        // For NOC 1, swap start and end coordinates
        return get_noc_multicast_addr(noc_x_end, noc_y_end, noc_x_start, noc_y_start, addr, noc);
    }
}

void print_tile_rows(
    uint32_t cb_idx,
    uint32_t tile_idx,
    bool untilize = false,
    uint16_t start_row = 0,
    uint16_t end_row = 32,
    uint8_t start_col = 0,
    uint8_t end_col = 32) {
    DPRINT << "cb_idx: " << cb_idx << " tile_idx: " << tile_idx << ENDL();
    DPRINT << "======" << ENDL();
    for (uint16_t r = start_row; r < end_row; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_idx,
                      tile_idx,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)start_col,
                          .w1 = (uint8_t)end_col,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

void zero_buffer_async(uint32_t write_addr, int bytes) {
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        uint32_t curr_bytes = std::min(bytes, MEM_ZEROS_SIZE);
        noc_async_read(zeros_noc_addr, write_addr, curr_bytes);
        write_addr += curr_bytes;
        bytes -= curr_bytes;
    }
}

void zero_buffer_barrier() { noc_async_read_barrier(); }

void kernel_main() {
    constexpr uint32_t input_tensor_cb_id = get_named_compile_time_arg_val("input_tensor_cb_id");
    constexpr uint32_t indices_tensor_cb_id = get_named_compile_time_arg_val("indices_tensor_cb_id");
    constexpr uint32_t mapping_tensor_cb_id = get_named_compile_time_arg_val("mapping_tensor_cb_id");
    constexpr uint32_t scores_tensor_cb_id = get_named_compile_time_arg_val("scores_tensor_cb_id");
    constexpr uint32_t e_d_buffer_id = get_named_compile_time_arg_val("e_d_buffer_id");

    constexpr uint32_t input_pages = get_named_compile_time_arg_val("input_pages");
    constexpr uint32_t indices_pages = get_named_compile_time_arg_val("indices_pages");
    constexpr uint32_t mapping_pages = get_named_compile_time_arg_val("mapping_pages");

    constexpr uint32_t input_page_size = get_named_compile_time_arg_val("input_page_size");
    constexpr uint32_t indices_page_size = get_named_compile_time_arg_val("indices_page_size");
    constexpr uint32_t mapping_page_size = get_named_compile_time_arg_val("mapping_page_size");
    constexpr uint32_t metadata_page_size = get_named_compile_time_arg_val("metadata_page_size");
    constexpr uint32_t output_page_size = get_named_compile_time_arg_val("output_page_size");

    constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    constexpr uint32_t tokens_per_device = get_named_compile_time_arg_val("tokens_per_device");

    constexpr uint32_t mesh_rows = get_named_compile_time_arg_val("mesh_rows");
    constexpr uint32_t mesh_cols = get_named_compile_time_arg_val("mesh_cols");

    constexpr uint32_t aligned_indices_page_size = get_named_compile_time_arg_val("aligned_indices_page_size");
    constexpr uint32_t aligned_mapping_page_size = get_named_compile_time_arg_val("aligned_mapping_page_size");
    constexpr uint32_t aligned_metadata_page_size = get_named_compile_time_arg_val("aligned_metadata_page_size");

    constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
    constexpr uint32_t cluster_axis = get_named_compile_time_arg_val("cluster_axis");
    constexpr uint32_t max_indices_pages_per_packet = get_named_compile_time_arg_val("max_indices_pages_per_packet");

    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t l1_alignment = get_named_compile_time_arg_val("l1_alignment");

    // Multicast coordinates for signaling sender cores that E-D buffer is ready
    constexpr uint32_t sender_mcast_start_x = get_named_compile_time_arg_val("sender_mcast_start_x");
    constexpr uint32_t sender_mcast_start_y = get_named_compile_time_arg_val("sender_mcast_start_y");
    constexpr uint32_t sender_mcast_end_x = get_named_compile_time_arg_val("sender_mcast_end_x");
    constexpr uint32_t sender_mcast_end_y = get_named_compile_time_arg_val("sender_mcast_end_y");
    constexpr uint32_t num_sender_cores = get_named_compile_time_arg_val("num_sender_cores");
    constexpr uint32_t ed_buffer_ready_semaphore_id = get_named_compile_time_arg_val("ed_buffer_ready_semaphore_id");

    constexpr ReplicateGroup axis = ReplicateGroup(cluster_axis);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t dispatch_index =
        axis == ReplicateGroup::COLS ? linearized_mesh_coord / mesh_cols : linearized_mesh_coord % mesh_cols;

    uint32_t ed_addr = get_read_ptr(e_d_buffer_id);
    zero_buffer_async(ed_addr, experts * dispatch_devices * l1_alignment * sizeof(uint32_t));
    size_t rt_args_idx = 0;
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);         // 0
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);       // 1
    uint32_t output_scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);  // 2
    bool is_drain_tilizer_core = (bool)get_arg_val<uint32_t>(rt_args_idx++);       // 3
    zero_buffer_barrier();

    // Signal all sender cores that E-D buffer has been zeroed
    // Set local semaphore to 1, then multicast write to all sender cores
    if (is_drain_tilizer_core) {
        uint32_t ed_ready_sem_addr = get_semaphore(ed_buffer_ready_semaphore_id);
        volatile tt_l1_ptr uint32_t* ed_ready_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ed_ready_sem_addr);
        noc_semaphore_set(ed_ready_sem_ptr, 1);

        // Use safe multicast address that handles NOC 0 vs NOC 1 coordinate ordering
        uint64_t sender_mcast_noc_addr = get_safe_multicast_noc_addr(
            sender_mcast_start_x, sender_mcast_start_y, sender_mcast_end_x, sender_mcast_end_y, ed_ready_sem_addr);

        // Multicast write the semaphore value (1) to all sender cores
        // Sender cores wait for value 1 to know E-D buffer is ready
        noc_semaphore_set_multicast(ed_ready_sem_addr, sender_mcast_noc_addr, num_sender_cores);
        noc_async_write_barrier();
    }

    constexpr auto output_args = TensorAccessorArgs<0>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto output_scores_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();

    const auto output_tensor_addr_gen = TensorAccessor(output_args, output_tensor_address, output_page_size);
    const auto metadata_tensor_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, metadata_page_size);
    // scores and indices tensors use same page size
    const auto output_scores_tensor_addr_gen =
        TensorAccessor(output_scores_args, output_scores_tensor_address, indices_page_size);
}
