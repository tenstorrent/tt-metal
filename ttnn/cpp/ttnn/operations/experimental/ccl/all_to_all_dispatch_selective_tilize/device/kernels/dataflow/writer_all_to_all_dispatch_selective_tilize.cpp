// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"

namespace detail {

inline void dispatch_input_local_device_flushed(
    uint32_t input_token_read_addr, uint64_t output_token_write_addr, uint32_t output_page_size) {
    noc_async_write(input_token_read_addr, output_token_write_addr, output_page_size);
    noc_async_writes_flushed();
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

}  // namespace detail

using namespace ttnn::operations::ccl::common;

void kernel_main() {
    constexpr uint32_t input_tensor_cb_id = get_named_compile_time_arg_val("input_tensor_cb_id");
    constexpr uint32_t indices_tensor_cb_id = get_named_compile_time_arg_val("indices_tensor_cb_id");
    constexpr uint32_t mapping_tensor_cb_id = get_named_compile_time_arg_val("mapping_tensor_cb_id");
    constexpr uint32_t packet_header_cb_id = get_named_compile_time_arg_val("packet_header_cb_id");
    constexpr uint32_t send_preparation_buffer_cb_id = get_named_compile_time_arg_val("send_preparation_buffer_id");

    constexpr uint32_t input_pages = get_named_compile_time_arg_val("input_pages");
    constexpr uint32_t indices_pages = get_named_compile_time_arg_val("indices_pages");
    constexpr uint32_t mapping_pages = get_named_compile_time_arg_val("mapping_pages");
    constexpr uint32_t output_pages = get_named_compile_time_arg_val("output_pages");
    constexpr uint32_t metadata_pages = get_named_compile_time_arg_val("metadata_pages");

    constexpr uint32_t input_page_size = get_named_compile_time_arg_val("input_page_size");
    constexpr uint32_t indices_page_size = get_named_compile_time_arg_val("indices_page_size");
    constexpr uint32_t mapping_page_size = get_named_compile_time_arg_val("mapping_page_size");
    constexpr uint32_t output_page_size = get_named_compile_time_arg_val("output_page_size");
    constexpr uint32_t metadata_page_size = get_named_compile_time_arg_val("metadata_page_size");

    constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    constexpr uint32_t hidden_size = get_named_compile_time_arg_val("hidden_size");
    constexpr uint32_t batch_size = get_named_compile_time_arg_val("batch_size");
    constexpr uint32_t selected_experts_k = get_named_compile_time_arg_val("selected_experts_k");
    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t tokens_per_device = get_named_compile_time_arg_val("tokens_per_device");

    constexpr uint32_t num_links = get_named_compile_time_arg_val("num_links");
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_named_compile_time_arg_val("topology");

    constexpr uint32_t src_mesh_id = get_named_compile_time_arg_val("src_mesh_id");
    constexpr uint32_t src_chip_id = get_named_compile_time_arg_val("src_chip_id");
    constexpr uint32_t mesh_rows = get_named_compile_time_arg_val("mesh_rows");
    constexpr uint32_t mesh_cols = get_named_compile_time_arg_val("mesh_cols");
    constexpr uint32_t aligned_input_page_size = get_named_compile_time_arg_val("aligned_input_page_size");
    constexpr uint32_t aligned_indices_page_size = get_named_compile_time_arg_val("aligned_indices_page_size");
    constexpr uint32_t aligned_mapping_page_size = get_named_compile_time_arg_val("aligned_mapping_page_size");
    constexpr uint32_t aligned_output_page_size = get_named_compile_time_arg_val("aligned_output_page_size");
    constexpr uint32_t aligned_metadata_page_size = get_named_compile_time_arg_val("aligned_metadata_page_size");

    constexpr uint32_t fabric_max_packet_size = get_named_compile_time_arg_val("fabric_max_packet_size");
    constexpr uint32_t alignment = get_named_compile_time_arg_val("l1_alignment");
    constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
    constexpr uint32_t cluster_axis = get_named_compile_time_arg_val("cluster_axis");

    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto input_scores_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();
    constexpr auto output_scores_args = TensorAccessorArgs<input_scores_args.next_compile_time_args_offset()>();

    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);          // 0
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);        // 1
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);        // 2
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);         // 3
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);       // 4
    uint32_t global_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);      // 5
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);        // 6
    uint32_t input_scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 7
    uint32_t output_scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);  // 8
    uint32_t subtoken_offset = get_arg_val<uint32_t>(rt_args_idx++);               // 9
    uint32_t subtoken_size = get_arg_val<uint32_t>(rt_args_idx++);                 // 10
    uint32_t indices_start = get_arg_val<uint32_t>(rt_args_idx++);                 // 11
    uint32_t indices_end = get_arg_val<uint32_t>(rt_args_idx++);                   // 12

    constexpr uint32_t num_directions = 4;
    constexpr std::array<bool, num_directions> directions = DIRECTIONS;

    std::array<tt::tt_fabric::WorkerToFabricEdmSender, num_directions> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_args_idx);

    uint32_t send_preparation_buffer_address = get_write_ptr(send_preparation_buffer_cb_id);
    detail::zero_buffer_async(send_preparation_buffer_address, tokens_per_device * num_devices * sizeof(uint8_t));

    constexpr ReplicateGroup axis = ReplicateGroup(cluster_axis);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t col = linearized_mesh_coord % mesh_cols;

    constexpr uint32_t dispatch_index = axis == ReplicateGroup::COLS ? row : col;
    // Based on cluster axis, we only need to dispatch to the devices that are along the axis
    // If ReplicateGroup is COLs/AXIS is 1, then we dispatch alonw the ROW, and vice versa
    // For ReplicateGroup COLs/AXIS is 1, the device_begin_idx is the start of the row, and the device_end_idx is the
    // end of the row For ReplicateGroup ROWs/AXIS is 0, the device_begin_idx is the start of the column, and the
    // device_end_idx is the end of the column
    constexpr uint32_t device_begin_idx = axis == ReplicateGroup::COLS ? col : row * mesh_cols;
    constexpr uint32_t device_end_idx =
        (axis == ReplicateGroup::COLS)
            ? (col + mesh_rows * mesh_cols)   // last is col+(mesh_rows-1)*mesh_cols; add one stride
            : (row * mesh_cols + mesh_cols);  // last is row*mesh_cols+(mesh_cols-1); add one
    constexpr uint32_t device_stride = axis == ReplicateGroup::COLS ? mesh_cols : 1;

    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address, output_page_size);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, metadata_page_size);
    // Scores tensors use same page size as indices tensor
    const auto input_scores_addr_gen =
        TensorAccessor(input_scores_args, input_scores_tensor_address, indices_page_size);
    const auto output_scores_addr_gen =
        TensorAccessor(output_scores_args, output_scores_tensor_address, indices_page_size);

    uint32_t packet_header_buffer_address = get_read_ptr(packet_header_cb_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* metadata_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));

    uint32_t base_indices_addr = get_read_ptr(indices_tensor_cb_id);

    detail::zero_buffer_barrier();
    open_direction_connections_barrier(directions, fabric_connections);

    // Send initialization semaphore to configured targets for synchronization
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
    send_init_semaphore_to_configured_targets<
        linearized_mesh_coord,
        topology,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        axis,
        num_devices>(fabric_connections, metadata_packet_header, nullptr, nullptr, init_noc_semaphore_addr);

    // Wait for all devices to complete initialization synchronization
    bool needs_barrier = false;
    noc_semaphore_wait((uint32_t*)init_semaphore_address, dispatch_devices - 1);
    noc_semaphore_set((uint32_t*)init_semaphore_address, 0);

    close_direction_connections(directions, fabric_connections);
}
