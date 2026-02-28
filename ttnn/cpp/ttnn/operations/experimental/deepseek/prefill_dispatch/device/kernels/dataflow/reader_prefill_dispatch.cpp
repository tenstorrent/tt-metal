// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Prefill dispatch reader kernel

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

// Debug print control - set to 0 to disable dispatch debug prints, 1 to enable
#define ENABLE_DISPATCH_DEBUG 0

#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH DPRINT
#else
#define DPRINT_DISPATCH \
    if (0)              \
    DebugPrinter()
#endif

void kernel_main() {
    // ===== Compile Time Args =====
    // CB IDs (indices 0-5)
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_metadata_temp_id = get_compile_time_arg_val(4);  // Added but not used by reader
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(5);  // Added but not used by reader

    // Page counts (indices 6-12)
    constexpr uint32_t input_pages = get_compile_time_arg_val(6);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(7);
    constexpr uint32_t weights_pages = get_compile_time_arg_val(8);
    constexpr uint32_t offsets_pages = get_compile_time_arg_val(9);
    constexpr uint32_t output_pages = get_compile_time_arg_val(10);
    constexpr uint32_t metadata_pages = get_compile_time_arg_val(11);
    constexpr uint32_t experts_counter_pages = get_compile_time_arg_val(12);

    // Page sizes (indices 13-19)
    constexpr uint32_t input_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(14);
    constexpr uint32_t weights_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(16);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(17);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t experts_counter_page_size = get_compile_time_arg_val(19);

    // Operation parameters (indices 20-27)
    constexpr uint32_t num_devices = get_compile_time_arg_val(20);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(21);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(22);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(23);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(24);
    constexpr uint32_t metadata_len = get_compile_time_arg_val(25);
    constexpr uint32_t max_dispatched_tokens_per_expert = get_compile_time_arg_val(26);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(27);

    // Mesh information (indices 28-32)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(28);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(29);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(30);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(31);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(32);

    // Aligned page sizes (indices 33-39)
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(33);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(34);
    constexpr uint32_t aligned_weights_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(36);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(37);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(38);
    constexpr uint32_t aligned_experts_counter_page_size = get_compile_time_arg_val(39);

    // Fabric configuration (indices 40-43)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(40);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(41);
    constexpr uint32_t num_links = get_compile_time_arg_val(42);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(43);

    // TensorAccessorArgs for all 7 tensors (starting at index 44)
    constexpr auto input_args = TensorAccessorArgs<44>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto experts_counter_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();

    // ===== Runtime Args =====
    uint32_t rt_args = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t weights_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t offsets_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t experts_counter_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t cross_device_semaphore_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_args++);

    // Print key compile time args for debugging (using DPRINT_DATA1 - reader runs on RISCV_1)
    DPRINT_DISPATCH << "Reader kernel: CBs=" << cb_input_id << "," << cb_indices_id << "," << cb_weights_id << ","
                    << cb_offsets_id << " tokens=[" << token_start_idx << "," << token_end_idx << ")"
                    << " hidden_size=" << hidden_size << " experts_per_chip=" << experts_per_chip
                    << " token_start_idx=" << token_start_idx << " token_end_idx=" << token_end_idx << ENDL();

    // =====
    // input (chips/fractured ==1, seq_len_per_chip, hidden_size)
    // indices (chips/fractured ==1, seq_len_per_chip, k) - topk indices
    // weights (chips/fractured ==1, seq_len_per_chip, k) - topk weights
    // dispatch buffer (chips/fractured ==1, experts_per_chip, max_dispatched_tokens_per_expert, hidden_size)
    // dispatch metadata buffer (chips/fractured ==1, experts_per_chip, max_dispatched_tokens_per_expert, metadata_len)
    // offsets (chips/fractured==1, n_routed_experts) - TODO: optimization dim-1 should be only experts this chip can
    // dispatch to; experts counter (chips/fractured ==1, experts_per_chip)
    // =====

    // =====
    // read offsets
    DPRINT_DISPATCH << "Fetching offset tensor offsets_pages=" << offsets_pages
                    << " offset_page_size=" << offsets_page_size << ENDL();
    const auto offsets_addr_gen = TensorAccessor(offsets_args, offsets_tensor_address, offsets_page_size);
    for (uint32_t i = 0; i < offsets_pages; i++) {
        DPRINT_DISPATCH << "Fetching offsets tensor index: " << i << ENDL();
        cb_reserve_back(cb_offsets_id, 1);

        uint32_t l1_write_addr = get_write_ptr(cb_offsets_id);
        noc_async_read_page(i, offsets_addr_gen, l1_write_addr);
    }
    noc_async_read_barrier();
    cb_push_back(cb_offsets_id, offsets_pages);

    // =====
    // read input, indices and weights and push to writer core CB
    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address, aligned_input_page_size);
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address, aligned_indices_page_size);
    const auto weights_addr_gen = TensorAccessor(weights_args, weights_tensor_address, aligned_weights_page_size);
    for (uint32_t token = token_start_idx; token < token_end_idx; token++) {
        DPRINT_DISPATCH << "Fetching token index: " << token << ENDL();
        cb_reserve_back(cb_indices_id, 1);
        cb_reserve_back(cb_weights_id, 1);
        cb_reserve_back(cb_input_id, 1);

        uint32_t l1_write_addr = get_write_ptr(cb_indices_id);
        noc_async_read_page(token, indices_addr_gen, l1_write_addr);
        l1_write_addr = get_write_ptr(cb_weights_id);
        noc_async_read_page(token, weights_addr_gen, l1_write_addr);
        l1_write_addr = get_write_ptr(cb_input_id);
        noc_async_read_page(token, input_addr_gen, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_indices_id, 1);
        cb_push_back(cb_weights_id, 1);
        cb_push_back(cb_input_id, 1);
        // Fetch input, weights and indices and push it to the writer core
    }
}
