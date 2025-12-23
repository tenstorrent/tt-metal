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

template <typename DataType>
FORCE_INLINE DataType* tile_row_offset(DataType* indices_address, uint32_t row) {
    constexpr uint32_t num_face_width = 2;
    constexpr uint32_t num_face_height = 2;
    constexpr uint32_t FaceWidth = 16;
    constexpr uint32_t FaceHeight = 16;
    constexpr uint32_t TileHeight = 32;
    constexpr uint32_t TileWidth = 32;
    uint32_t offset = 0;
    uint32_t local_row = row;
    if (row >= FaceHeight) {
        offset += num_face_width * FaceHeight * FaceWidth;  // if it was generic, multiply by row/FaceHeight
        local_row -= FaceHeight;
    }
    offset += local_row * FaceWidth;
    return (DataType*)(indices_address + offset);
}

template <typename DataType>
FORCE_INLINE DataType* tile_col_offset(DataType* indices_address, uint32_t col) {
    constexpr uint32_t FaceWidth = 16;
    constexpr uint32_t FaceHeight = 16;
    uint32_t offset = 0;
    uint32_t local_col = col;
    if (col >= FaceWidth) {
        offset += FaceHeight * FaceWidth;  // if it was generic, multiply by col/FaceWidth
        local_col -= FaceWidth;
    }
    offset += local_col;
    return (DataType*)(indices_address + offset);
}

// DEBUG: Print E-D table contents
template <uint32_t ExpertsPerDevice, uint32_t DispatchDevices, uint32_t EntriesPerL1Alignment>
void print_ed_table(volatile tt_l1_ptr uint32_t* ed_table, uint32_t device_id, const char* label) {
    // Invalidate L1 cache to see updates from remote writes
    invalidate_l1_cache();

    DPRINT << "=== E-D Table: " << label << " (device " << device_id << ") ===" << ENDL();
    for (uint32_t local_expert = 0; local_expert < ExpertsPerDevice; local_expert++) {
        DPRINT << "Expert " << local_expert << ": [";
        for (uint32_t src_dev = 0; src_dev < DispatchDevices; src_dev++) {
            // Each entry is l1_alignment bytes apart, access first uint32 of each entry
            uint32_t entry_idx = (local_expert * DispatchDevices + src_dev) * EntriesPerL1Alignment;
            uint32_t val = ed_table[entry_idx];
            DPRINT << val;
            if (src_dev < DispatchDevices - 1) {
                DPRINT << ", ";
            }
        }
        DPRINT << "]" << ENDL();
    }
    DPRINT << "---" << ENDL();
}

template <uint32_t experts_per_device, uint32_t dispatch_devices, uint32_t entries_per_l1_alignment>
void poll_ed_table_loop(volatile tt_l1_ptr uint32_t* ed_table, uint32_t linearized_mesh_coord) {
    uint32_t poll_iteration = 0;
    while (true) {
        DPRINT << "Poll #" << poll_iteration << ENDL();
        print_ed_table<experts_per_device, dispatch_devices, entries_per_l1_alignment>(
            ed_table, linearized_mesh_coord, "polling");

        poll_iteration++;
        // Small delay between polls to avoid flooding DPRINT
        for (volatile uint32_t delay = 0; delay < 100000; delay++) {
        }
    }
}

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

    constexpr uint32_t selected_experts_k = get_named_compile_time_arg_val("selected_experts_k");

    constexpr uint32_t experts_per_device = (experts + num_devices - 1) / num_devices;

    constexpr ReplicateGroup axis = ReplicateGroup(cluster_axis);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t dispatch_index =
        axis == ReplicateGroup::COLS ? linearized_mesh_coord / mesh_cols : linearized_mesh_coord % mesh_cols;

    uint32_t ed_addr = get_read_ptr(e_d_buffer_id);
    uint32_t rt_args_idx = 0;
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);           // 0
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);         // 1
    uint32_t output_scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);    // 2
    uint32_t indices_sent_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);  // 3
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);          // 4
    bool is_drain_tilizer_core = (bool)get_arg_val<uint32_t>(rt_args_idx++);         // 5

    constexpr auto output_args = TensorAccessorArgs<0>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto output_scores_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<output_scores_args.next_compile_time_args_offset()>();

    const auto output_tensor_addr_gen = TensorAccessor(output_args, output_tensor_address, output_page_size);
    const auto metadata_tensor_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, metadata_page_size);
    // scores and indices tensors use same page size
    const auto output_scores_tensor_addr_gen =
        TensorAccessor(output_scores_args, output_scores_tensor_address, indices_page_size);
    const auto mapping_tensor_addr_gen = TensorAccessor(mapping_args, mapping_tensor_address, mapping_page_size);

    // read in the mapping tensor
    for (uint32_t mapping_page = 0; mapping_page < mapping_pages; mapping_page++) {
        cb_reserve_back(mapping_tensor_cb_id, 1);
        noc_async_read_page(mapping_page, mapping_tensor_addr_gen, get_write_ptr(mapping_tensor_cb_id));
        noc_async_read_barrier();
        cb_push_back(mapping_tensor_cb_id, 1);
    }

    uint16_t* devices_for_experts = (uint16_t*)get_read_ptr(mapping_tensor_cb_id);

    // POST-PROCESSING STAGE:
    // Read in metadata into indices cb and update the ground truth E-D table offset into the second half of the E-D
    // buffer. We iterate through all source devices' metadata, counting how many tokens will arrive at each
    // local expert from each source device.
    constexpr uint32_t tile_height = 32;
    uint32_t base_ed_table_offset = ed_addr + experts_per_device * dispatch_devices * l1_alignment;
    volatile tt_l1_ptr uint32_t* ed_table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_ed_table_offset);

    for (uint32_t device_id = 0; device_id < dispatch_devices; device_id++) {
        uint32_t tokens_processed = 0;
        uint32_t indices_page = 0;

        while (tokens_processed < tokens_per_device) {
            cb_wait_front(indices_tensor_cb_id, 1);
            uint16_t* indices_ptr = (uint16_t*)get_read_ptr(indices_tensor_cb_id);

            // Process tokens within this tile, but don't exceed tokens_per_device
            uint32_t tile_row = 0;
            while (tile_row < tile_height && tokens_processed < tokens_per_device) {
                uint16_t* token_ptr = tile_row_offset(indices_ptr, tile_row);
                for (uint32_t k = 0; k < selected_experts_k; k++) {
                    uint16_t expert_chosen = tile_col_offset(token_ptr, k)[0];
                    uint16_t d = devices_for_experts[expert_chosen];
                    if (d == dispatch_index) {
                        uint32_t local_expert = expert_chosen % experts_per_device;
                        ed_table[local_expert * dispatch_devices + device_id]++;
                    }
                }
                tile_row++;
                tokens_processed++;
            }

            cb_pop_front(indices_tensor_cb_id, 1);
            indices_page++;
        }
    }
    constexpr uint32_t entries_per_l1_alignment = l1_alignment / sizeof(uint32_t);
    print_ed_table<experts_per_device, dispatch_devices, entries_per_l1_alignment>(
        ed_table, linearized_mesh_coord, "ground truth E-D table");
}
