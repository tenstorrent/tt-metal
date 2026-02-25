// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Prefill dispatch writer kernel

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

void kernel_main() {
    // ===== Compile Time Args =====
    // CB IDs (indices 0-4)
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_metadata_temp_id = get_compile_time_arg_val(4);

    // Page counts (indices 5-11)
    constexpr uint32_t input_pages = get_compile_time_arg_val(5);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(6);
    constexpr uint32_t weights_pages = get_compile_time_arg_val(7);
    constexpr uint32_t offsets_pages = get_compile_time_arg_val(8);
    constexpr uint32_t output_pages = get_compile_time_arg_val(9);
    constexpr uint32_t metadata_pages = get_compile_time_arg_val(10);
    constexpr uint32_t experts_counter_pages = get_compile_time_arg_val(11);

    // Page sizes (indices 12-18)
    constexpr uint32_t input_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t weights_page_size = get_compile_time_arg_val(14);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(16);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(17);
    constexpr uint32_t experts_counter_page_size = get_compile_time_arg_val(18);

    // Operation parameters (indices 19-26)
    constexpr uint32_t num_devices = get_compile_time_arg_val(19);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(20);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(21);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(22);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(23);
    constexpr uint32_t metadata_len = get_compile_time_arg_val(24);
    constexpr uint32_t max_dispatched_tokens_per_expert = get_compile_time_arg_val(25);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(26);

    // Mesh information (indices 27-31)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(27);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(28);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(29);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(30);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(31);

    // Aligned page sizes (indices 32-38)
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(32);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(33);
    constexpr uint32_t aligned_weights_page_size = get_compile_time_arg_val(34);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(36);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(37);
    constexpr uint32_t aligned_experts_counter_page_size = get_compile_time_arg_val(38);

    // Fabric configuration (indices 39-40)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(39);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(40);

    // TensorAccessorArgs for all 7 tensors (starting at index 41)
    constexpr auto input_args = TensorAccessorArgs<41>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto experts_counter_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();

    // ===== Runtime Args =====
    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t weights_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t offsets_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t experts_counter_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t cross_device_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_args_idx++);

    // Fabric connection args follow (appended by append_fabric_connection_rt_args)
    // These will be read by fabric API calls

    // Print key compile time args for debugging (using DPRINT_DATA0 - writer runs on RISCV_0)
    DPRINT << "linearized_mesh_coord=" << linearized_mesh_coord << " src_mesh_id=" << src_mesh_id
           << " src_chip_id=" << src_chip_id << " mesh_rows=" << mesh_rows << " mesh_cols=" << mesh_cols << ENDL();

    DPRINT << "Writer kernel: CBs=" << cb_input_id << "," << cb_indices_id << "," << cb_weights_id << ","
           << cb_offsets_id << " tokens=[" << token_start_idx << "," << token_end_idx << ")"
           << " hidden_size=" << hidden_size << " experts_per_chip=" << experts_per_chip << ENDL();

    // ====
    // wait for offsets to be ready
    cb_wait_front(cb_offsets_id, offsets_pages);
    int32_t* offsets = (int32_t*)(get_read_ptr(cb_offsets_id));

    for (uint32_t o = 0; o < n_routed_experts; ++o) {
        DPRINT << "Offset for expert " << o << " is " << offsets[o] << ENDL();
    }

    DPRINT << "aligned_metadata_page_size=" << aligned_metadata_page_size
           << " metadata_page_size=" << metadata_page_size << ENDL();

    // ====
    // process tokens/indices/weights one by one as they arrive to CB
    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address, aligned_output_page_size);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, aligned_metadata_page_size);

    DPRINT << "metadata_tensor_address=" << HEX() << metadata_tensor_address << DEC()
           << " aligned_metadata_page_size=" << aligned_metadata_page_size << ENDL();

    // Reserve CB for metadata buffer (using L1 memory accessible by NOC)
    cb_reserve_back(cb_metadata_temp_id, 1);

    for (uint32_t token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
        DPRINT << "Processing token_idx: " << token_idx << ENDL();

        cb_wait_front(cb_indices_id, 1);
        cb_wait_front(cb_weights_id, 1);
        cb_wait_front(cb_input_id, 1);

        uint32_t input_token_read_addr = get_read_ptr(cb_input_id);
        int32_t* indices = (int32_t*)(get_read_ptr(cb_indices_id));
        uint16_t* weights = (uint16_t*)(get_read_ptr(cb_weights_id));  // this is really a bfloat16 data
        for (uint32_t k = 0; k < num_experts_per_tok; ++k) {
            auto routed_expert = indices[k];
            auto expert_chip = routed_expert / experts_per_chip;
            auto expert_index_within_chip = routed_expert % experts_per_chip;

            DPRINT << "  Expert [" << k << "]=" << routed_expert << " (chip=" << expert_chip << ")" << ENDL();

            auto& offset = offsets[routed_expert];

            // Calculate byte offsets from page indices
            // aligned_page_size is in elements, need to multiply by element size to get bytes
            auto page_idx = expert_index_within_chip * max_dispatched_tokens_per_expert + offset;
            auto output_token_write_addr =
                output_addr_gen.get_noc_addr(0) + page_idx * aligned_output_page_size * 2;  // bfloat16 = 2 bytes
            auto metadata_write_addr =
                metadata_addr_gen.get_noc_addr(0) + page_idx * aligned_metadata_page_size * 4;  // int32 = 4 bytes

            if (expert_chip == linearized_mesh_coord) {
                DPRINT << "    Expert [" << k << "]=" << routed_expert << " is local to this chip." << ENDL();
                // For local dispatch, we can directly write to the output buffer without going through the fabric
                noc_async_write_page(page_idx, output_addr_gen, input_token_read_addr);
                // noc_async_write(
                //     input_token_read_addr,
                //     output_addr_gen.get_noc_addr(page_idx),     // Correct: let AddrGen compute address
                //     aligned_output_page_size * 2);               // Correct: use OUTPUT page size in bytes
                noc_async_writes_flushed();  // is it formally needed?

                uint32_t metadata_cb_addr = get_write_ptr(cb_metadata_temp_id);
                volatile tt_l1_ptr int32_t* metadata = reinterpret_cast<volatile tt_l1_ptr int32_t*>(metadata_cb_addr);

                // Set actual metadata values
                metadata[0] = linearized_mesh_coord;
                metadata[1] = token_idx;
                metadata[2] = k;
                metadata[3] = routed_expert;
                metadata[4] = weights[k];

                // Write metadata from CB to output tensor
                noc_async_write_page(page_idx, metadata_addr_gen, metadata_cb_addr);
                noc_async_writes_flushed();

            } else {
                DPRINT << "    Expert [" << k << "]=" << routed_expert << " is remote. Fabric not implemented."
                       << ENDL();
                // if the expert lives on a remote device, we dispatch the input token to it
                // if axis is specified then we only send to the devices that are along the axis
                // if axis is not specified then we send to all devices
                // if constexpr (is_1d_topology<topology>()) {
                //     fabric_send_chip_unicast_noc_unicast_1d<
                //         linearized_mesh_coord,
                //         topology,
                //         mesh_rows,
                //         mesh_cols,
                //         fabric_max_packet_size>(
                //         output_addr_gen,
                //         fabric_connections,
                //         unicast_packet_header,
                //         d,
                //         input_token_read_addr,
                //         global_token,
                //         (int)output_page_size,
                //         alignment);
                // } else {
                //     fabric_send_chip_unicast_noc_unicast<src_chip_id, mesh_rows, mesh_cols, fabric_max_packet_size>(
                //         output_addr_gen,
                //         fabric_connections,
                //         unicast_packet_header,
                //         dest_chip_ids[d],
                //         dest_mesh_ids[d],
                //         input_token_read_addr,
                //         global_token,
                //         (int)output_page_size,
                //         alignment);
                // }
            }

            // offset++;
            offsets[routed_expert] += 1;
        }
        noc_async_write_barrier();  // not needed if there were no local dispatches

        cb_pop_front(cb_indices_id, 1);
        cb_pop_front(cb_weights_id, 1);
        cb_pop_front(cb_input_id, 1);
    }

    // Release CB for next use
    cb_push_back(cb_metadata_temp_id, 1);

    // TODO: Implement writer kernel logic
    // - Read from input CBs (data prepared by reader)
    // - Send dispatched tokens via fabric to other devices
    // - Write metadata and experts counter to output tensors
    // - Coordinate with other devices using semaphores
}
