// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"

constexpr uint32_t cb_combine_input_id = tt::CBIndex::c_0;
constexpr uint32_t cb_weights_id = tt::CBIndex::c_1;
constexpr uint32_t cb_dispatch_table_id = tt::CBIndex::c_2;
constexpr uint32_t cb_indices_id = tt::CBIndex::c_3;
constexpr uint32_t cb_output_id = tt::CBIndex::c_16;
constexpr uint32_t cb_rowmajor_id = tt::CBIndex::c_17;

constexpr uint32_t num_experts = get_compile_time_arg_val(0);
constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(1);
// dispatch_table_num_pages is only meaningful when use_dispatch_table_skip
// is true; it carries zero from the program factory otherwise.
constexpr uint32_t dispatch_table_num_pages = get_compile_time_arg_val(2);
constexpr bool use_dispatch_table_skip = get_compile_time_arg_val(3) != 0;
// Per-token scaled fp8 combine rows (fused DQ2): c_0 holds e4m3 pseudo-tiles and c_4 the slot's fp32 row scale
// (word 0). Each local slot is unpacked to 32-bit DEST and scaled by weight * row_scale with two SFPU scalar
// multiplies; everything stays in compute.
constexpr bool fp8_in = get_compile_time_arg_val(4) != 0;
constexpr uint32_t cb_row_scale_id = tt::CBIndex::c_4;

// Like read_tile_value but reads a uint16 element (zero-extended to uint32_t).
// Indices arrive from DRAM as uint16 and stay uint16 in the CB; this avoids
// an in-place uint16→int32 expansion in the writer kernel and removes an
// architecture-dependent slot-size constraint on max top-k.
ALWI uint32_t read_tile_value_uint16(uint32_t cb_id, uint32_t tile_index, uint32_t element_offset) {
#ifndef ARCH_QUASAR
    uint32_t value = 0;
    UNPACK({
        uint32_t operand_id = get_operand_id(cb_id);
        uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr;
        uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
        uint32_t byte_address = (base_address + offset_address) << 4;
        value = (uint32_t)reinterpret_cast<volatile uint16_t*>(byte_address)[element_offset];
        mailbox_write(ckernel::ThreadId::MathThreadId, value);
        mailbox_write(ckernel::ThreadId::PackThreadId, value);
    })
    MATH(value = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK(value = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    return value;
#else
    ASSERT(false && "read_tile_value_uint16 is not implemented for ARCH_QUASAR");
    return 0;
#endif
}

void kernel_main() {
    constexpr uint32_t TOKENS_PER_CHUNK = 32;
    uint32_t token_start_idx = get_arg_val<uint32_t>(0);
    uint32_t num_chunks = get_arg_val<uint32_t>(1);
    constexpr uint32_t total_token_tiles = TOKENS_PER_CHUNK * emb_dim_cb_tiles;

    CircularBuffer cb_combine_input(cb_combine_input_id);
    CircularBuffer cb_weights(cb_weights_id);
    CircularBuffer cb_dispatch_table(cb_dispatch_table_id);
    CircularBuffer cb_indices(cb_indices_id);
    CircularBuffer cb_output(cb_output_id);
    CircularBuffer cb_rowmajor(cb_rowmajor_id);
    CircularBuffer cb_row_scale(cb_row_scale_id);

    if constexpr (use_dispatch_table_skip) {
        // Wait for writer to finish pre-loading dispatch table (loaded once for all chunks)
        cb_dispatch_table.wait_front(dispatch_table_num_pages);
    }

    compute_kernel_hw_startup(cb_combine_input_id, cb_weights_id, cb_output_id);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        if constexpr (use_dispatch_table_skip) {
            // Wait for writer to load this chunk's indices (one chunk at a time)
            cb_indices.wait_front(TOKENS_PER_CHUNK);
        }

        cb_rowmajor.reserve_back(total_token_tiles);

        // Process one expert at a time: both input (c_0) and weight (c_1) are streamed
        // one expert at a time by reader and writer respectively.
        for (uint32_t i = 0; i < TOKENS_PER_CHUNK; ++i) {
            if constexpr (!fp8_in) {
                mul_bcast_scalar_init(cb_combine_input_id, cb_weights_id);
            }

            // first_active tracks whether we've picked the accumulator-initializing
            // expert yet: the DeepSeek path looks for a locally-mapped expert via
            // the dispatch table; the GPT-OSS path looks for a non-zero routing
            // weight. A single pass skips inactive experts; if none qualified,
            // the last expert is forced through to initialise the accumulator.
            bool first_active = true;
            for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                cb_combine_input.wait_front(emb_dim_cb_tiles);
                cb_weights.wait_front(1);
                if constexpr (fp8_in) {
                    cb_row_scale.wait_front(1);
                }

                bool skip_expert = false;
                bool must_zero_init = false;
                // Forced last slot of a token with no local expert: its combine row was never written.
                bool zero_fill = false;
                if constexpr (use_dispatch_table_skip) {
                    uint32_t expert_id = read_tile_value_uint16(cb_indices_id, i, expert_idx);
                    // Check dispatch table: -1 (0xFFFFFFFF) means non-local
                    uint32_t chip_id = read_tile_value(cb_dispatch_table_id, 0, expert_id);
                    bool is_local = (chip_id != 0xFFFFFFFF);

                    // On the last expert, if none were local, we must process it to
                    // initialize the accumulator. Writer guarantees the weight is zero
                    // for this case, so multiply produces zeros.
                    bool is_last = (expert_idx == num_experts - 1);
                    must_zero_init = is_last && first_active;
                    skip_expert = !is_local && !must_zero_init;
                    zero_fill = must_zero_init && !is_local;
                } else {
                    // Read weight value — if zero, skip (but always process at least one
                    // expert so the accumulator gets initialized with valid data)
                    uint32_t weight_val = read_tile_value(cb_weights_id, 0, 0);
                    skip_expert = (weight_val == 0) && !first_active;
                }

                if (skip_expert) {
                    cb_combine_input.pop_front(emb_dim_cb_tiles);
                    cb_weights.pop_front(1);
                    if constexpr (fp8_in) {
                        cb_row_scale.pop_front(1);
                    }
                    continue;
                }

                if (!first_active) {
                    pack_reconfig_l1_acc(1);  // accumulate
                } else {
                    pack_reconfig_l1_acc(0);  // overwrite
                    first_active = false;
                }

                if constexpr (fp8_in) {
                    // Weight (bf16 element 0 -> fp32 bits) and row scale (fp32), mailboxed to all threads.
                    const uint32_t w_bits = (read_tile_value(cb_weights_id, 0, 0) & 0xFFFFu) << 16;
                    const uint32_t s_bits = read_tile_value(cb_row_scale_id, 0, 0);
                    tile_regs_acquire();
                    if (zero_fill) {
                        // No local expert: the writer zeroed the weight tile; never read the unwritten slot.
                        reconfig_data_format_srca(cb_weights_id);
                        copy_init(cb_weights_id);
                        for (uint32_t j = 0; j < emb_dim_cb_tiles; j++) {
                            copy_tile(cb_weights_id, 0, j);
                        }
                    } else {
                        reconfig_data_format_srca(cb_combine_input_id);
                        copy_init(cb_combine_input_id);
                        for (uint32_t j = 0; j < emb_dim_cb_tiles; j++) {
                            copy_tile(cb_combine_input_id, j, j);
                        }
                        binop_with_scalar_tile_init();
                        for (uint32_t j = 0; j < emb_dim_cb_tiles; j++) {
                            mul_unary_tile(j, w_bits);
                            mul_unary_tile(j, s_bits);
                        }
                    }
                } else {
                    tile_regs_acquire();
                    for (uint32_t j = 0; j < emb_dim_cb_tiles; j++) {
                        mul_tiles_bcast<BroadcastType::SCALAR>(cb_combine_input_id, cb_weights_id, j, 0, j);
                    }
                }

                tile_regs_commit();
                tile_regs_wait();

                for (uint32_t j = 0; j < emb_dim_cb_tiles; j++) {
                    pack_tile<true>(j, cb_rowmajor_id, i * emb_dim_cb_tiles + j);
                }

                tile_regs_release();

                cb_combine_input.pop_front(emb_dim_cb_tiles);
                cb_weights.pop_front(1);
                if constexpr (fp8_in) {
                    cb_row_scale.pop_front(1);
                }
            }
            pack_reconfig_l1_acc(0);
        }

        if constexpr (use_dispatch_table_skip) {
            // Release this chunk's indices so writer can load the next chunk's
            cb_indices.pop_front(TOKENS_PER_CHUNK);
        }

        cb_rowmajor.push_back(total_token_tiles);

        compute_kernel_lib::tilize<total_token_tiles, cb_rowmajor_id, cb_output_id>(1);
    }
}
