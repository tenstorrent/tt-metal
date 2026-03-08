// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Prefill combine reader kernel
// This kernel reads expert outputs and metadata, then routes them back to original token positions

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// Debug print control - set to 0 to disable combine debug prints, 1 to enable
#define ENABLE_COMBINE_DEBUG 0
#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE DPRINT
#else
#define DPRINT_COMBINE \
    if (0)             \
    DebugPrinter()
#endif

// Zero a DRAM/L1 page by writing zeros from L1's MEM_ZEROS_BASE (unicast)
void zero_page_async(uint64_t noc_addr, uint32_t page_size) {
    uint32_t bytes = page_size;
    while (bytes > 0) {
        uint32_t curr_bytes = std::min(bytes, (uint32_t)MEM_ZEROS_SIZE);
        noc_async_write(MEM_ZEROS_BASE, noc_addr, curr_bytes);
        noc_addr += curr_bytes;
        bytes -= curr_bytes;
    }
}

// L1 multicast zero-init is enabled when all required defines are present
#if defined(IS_L1_OUTPUT) && IS_L1_OUTPUT && defined(L1_BANK_NOC_X_START) && defined(NUM_L1_BANKS)
#define USE_L1_MULTICAST_ZERO 1
#else
#define USE_L1_MULTICAST_ZERO 0
#endif

void kernel_main() {
    // ===== Compile Time Args =====
    // CB IDs (indices 0-3)
    constexpr uint32_t cb_dispatched_buffer_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dispatched_metadata_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_experts_tok_counter_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output_id = get_compile_time_arg_val(3);

    // Page counts (indices 4-7)
    constexpr uint32_t dispatched_buffer_pages = get_compile_time_arg_val(4);
    constexpr uint32_t dispatched_metadata_pages = get_compile_time_arg_val(5);
    constexpr uint32_t experts_tok_counter_pages = get_compile_time_arg_val(6);
    constexpr uint32_t output_pages = get_compile_time_arg_val(7);

    // Page sizes (indices 8-11)
    constexpr uint32_t dispatched_buffer_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t dispatched_metadata_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t experts_tok_counter_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(11);

    // Operation parameters (indices 12-16)
    constexpr uint32_t num_chips = get_compile_time_arg_val(12);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(13);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(14);
    constexpr uint32_t seq_len_per_chip = get_compile_time_arg_val(15);
    constexpr uint32_t max_dispatched_tokens_per_expert = get_compile_time_arg_val(16);

    // Hidden dimension (index 17)
    constexpr uint32_t hidden_size = get_compile_time_arg_val(17);

    // Aligned page sizes (indices 18-21)
    constexpr uint32_t aligned_dispatched_buffer_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t aligned_dispatched_metadata_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t aligned_experts_tok_counter_page_size = get_compile_time_arg_val(20);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(21);

    // TensorAccessorArgs for all 4 tensors (starting at index 31)
    constexpr auto dispatched_buffer_args = TensorAccessorArgs<31>();
    constexpr auto dispatched_metadata_args =
        TensorAccessorArgs<dispatched_buffer_args.next_compile_time_args_offset()>();
    constexpr auto experts_tok_counter_args =
        TensorAccessorArgs<dispatched_metadata_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<experts_tok_counter_args.next_compile_time_args_offset()>();

    // ===== Runtime Args =====
    uint32_t rt_args = 0;
    uint32_t dispatched_buffer_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t dispatched_metadata_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t experts_tok_counter_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args++);

    // Print key compile time args for debugging (RISCV_1)
    DPRINT_COMBINE << "Combine Reader: CBs=" << cb_dispatched_buffer_id << "," << cb_dispatched_metadata_id << ","
                   << cb_experts_tok_counter_id << "," << cb_output_id << " num_chips=" << num_chips
                   << " experts_per_chip=" << experts_per_chip << " num_experts_per_tok=" << num_experts_per_tok
                   << " seq_len_per_chip=" << seq_len_per_chip
                   << " max_dispatched_tokens_per_expert=" << max_dispatched_tokens_per_expert
                   << " hidden_size=" << hidden_size << ENDL();

    // Zero-initialize output buffer before reading inputs
    // This overlaps with fabric initialization in the writer kernel
    // Each chip initializes its own output tensor to ensure predictable values
    const auto output_addr_gen = TensorAccessor(output_args, output_addr, aligned_output_page_size);
#if INIT_ZEROS
#if USE_L1_MULTICAST_ZERO
    // L1 interleaved: use multicast to zero all bank cores at once
    // This is faster than per-page unicast since one multicast covers all cores
    constexpr uint32_t per_bank_bytes = OUTPUT_BYTES_PER_BANK;
    DPRINT_COMBINE << "Zero-init L1 (multicast): per_bank_bytes=" << per_bank_bytes << " num_banks=" << NUM_L1_BANKS
                   << ENDL();
    // Inline multicast loop - zeros same L1 offset on ALL bank cores simultaneously
    for (uint32_t offset = 0; offset < per_bank_bytes; offset += MEM_ZEROS_SIZE) {
        uint32_t chunk_size = std::min((uint32_t)MEM_ZEROS_SIZE, per_bank_bytes - offset);
        uint64_t mcast_addr = get_noc_multicast_addr(
            L1_BANK_NOC_X_START, L1_BANK_NOC_Y_START, L1_BANK_NOC_X_END, L1_BANK_NOC_Y_END, output_addr + offset);
        noc_async_write_multicast(MEM_ZEROS_BASE, mcast_addr, chunk_size, NUM_L1_BANKS - 1);
    }
    noc_async_write_barrier();
#else
    // DRAM interleaved or L1 without multicast defines: use per-page unicast
    DPRINT_COMBINE << "Zero-init (unicast): pages=" << output_pages << " page_size=" << aligned_output_page_size
                   << ENDL();
    for (uint32_t page = 0; page < output_pages; page++) {
        uint64_t page_noc_addr = get_noc_addr(page, output_addr_gen);
        zero_page_async(page_noc_addr, aligned_output_page_size);
    }
    noc_async_write_barrier();
#endif
    DPRINT_COMBINE << "Zero-init done" << ENDL();
#endif

    // ====
    // read experts token counter (chips/fractured ==1, experts_per_chip)
    const auto experts_tok_counter_addr_gen =
        TensorAccessor(experts_tok_counter_args, experts_tok_counter_addr, aligned_experts_tok_counter_page_size);
    for (uint32_t i = 0; i < experts_tok_counter_pages; i++) {
        DPRINT_COMBINE << "Fetching experts token counter; page=" << i << ENDL();
        cb_reserve_back(cb_experts_tok_counter_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_experts_tok_counter_id);
        noc_async_read_page(i, experts_tok_counter_addr_gen, l1_write_addr);
    }
    noc_async_read_barrier();
    cb_push_back(cb_experts_tok_counter_id, experts_tok_counter_pages);

    // read dispatched buffers and metadata
    // dispatch buffer (chips/fractured ==1, experts_per_chip, max_dispatched_tokens_per_expert, hidden_size)
    // dispatch metadata buffer (chips/fractured ==1, experts_per_chip, max_dispatched_tokens_per_expert, metadata_len)

    const auto dispatched_buffer_addr_gen =
        TensorAccessor(dispatched_buffer_args, dispatched_buffer_addr, aligned_dispatched_buffer_page_size);
    const auto dispatched_metadata_addr_gen =
        TensorAccessor(dispatched_metadata_args, dispatched_metadata_addr, aligned_dispatched_metadata_page_size);

    constexpr auto expert_stride = max_dispatched_tokens_per_expert;

    // Set up packet headers from CB (cb_packet_header_id from compile-time args)
    uint32_t experts_tok_counter_l1_addr = get_read_ptr(cb_experts_tok_counter_id);
    volatile tt_l1_ptr uint32_t* experts_tok_counter_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(experts_tok_counter_l1_addr);

    for (uint32_t local_expert = 0; local_expert < experts_per_chip; local_expert++) {
        DPRINT_COMBINE << "Local expert=" << local_expert << ENDL();
        uint32_t start_page = local_expert * expert_stride;
        uint32_t expert_tokens = experts_tok_counter_l1[local_expert];
        uint32_t end_page = start_page + expert_tokens;

        DPRINT_COMBINE << "  Tokens for expert: " << expert_tokens << " (pages [" << start_page << "," << end_page
                       << "))" << ENDL();

        for (uint32_t token = start_page; token < end_page; token++) {
            // DPRINT_COMBINE << "    Fetching token index/page: " << token << ENDL();
            cb_reserve_back(cb_dispatched_buffer_id, 1);
            cb_reserve_back(cb_dispatched_metadata_id, 1);

            uint32_t l1_buffer_write_addr = get_write_ptr(cb_dispatched_buffer_id);
            noc_async_read_page(token, dispatched_buffer_addr_gen, l1_buffer_write_addr);
            uint32_t l1_metadata_write_addr = get_write_ptr(cb_dispatched_metadata_id);
            noc_async_read_page(token, dispatched_metadata_addr_gen, l1_metadata_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_dispatched_buffer_id, 1);
            cb_push_back(cb_dispatched_metadata_id, 1);
        }
    }
}
