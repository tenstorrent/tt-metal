// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the 1x32 / 1x64 / 1x128 scaled-fp8 post_combine_reduce (fused DQ2, sparse SDPA style).
//
// Combine rows are [H e4m3 | H/32 fp32 scales]. Per 32-token block the reader stacks one combine row per token
// as 32 row-major rows of this core's column part (part_wt tiles wide) in c_0; compute tilizes them so tile
// row t is token t and each tile row is exactly one 1x32 scale block. Only local slots carry data, so the
// rows are compacted: stage s holds every token's s-th local slot, and a token with fewer local slots gets
// a zero scale for that row (its c_0 row keeps finite stale fp8, so it contributes 0).
//
// c_0 holds a whole block, one stage per top-k slot, so every read of the block is in flight behind one barrier:
// one stage at a time left the op bound by DRAM latency. The reader always fills and the compute always frees
// all of c_0, unused stages included, so each block starts at its base.
//
// Per stage the reader also builds the multiply operand, column 0 only (face-aware): c_4 holds one bf16 tile per
// scale block of the part, column 0 row t = the slot's routing weight times that block's scale. The scale is a power
// of two, so the product is the weight with the scale's exponent added to its own: exact, and no multiply. Per block
// it hands compute the stage count through the c_7 mailbox.

#include <cstdint>

#ifndef TILES_PER_SCALE
#define TILES_PER_SCALE 1
#endif
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include <tt-metalium/constants.hpp>

constexpr uint32_t cb_rm_id = tt::CBIndex::c_0;
constexpr uint32_t cb_dispatch_table_id = tt::CBIndex::c_2;
constexpr uint32_t cb_indices_id = tt::CBIndex::c_3;
constexpr uint32_t cb_scale_id = tt::CBIndex::c_4;
constexpr uint32_t cb_stage_count_id = tt::CBIndex::c_7;
constexpr uint32_t cb_scratch_id = tt::CBIndex::c_8;

constexpr uint32_t num_experts = get_compile_time_arg_val(0);  // top-k slots per token
constexpr uint32_t part_wt = get_compile_time_arg_val(1);      // tiles (1x32 scale blocks) per part row
constexpr uint32_t dispatch_table_num_pages = get_compile_time_arg_val(2);
constexpr uint32_t dispatch_table_page_size = get_compile_time_arg_val(3);
constexpr uint32_t dispatch_table_aligned_page_size = get_compile_time_arg_val(4);
constexpr uint32_t indices_page_size = get_compile_time_arg_val(5);
constexpr uint32_t indices_aligned_page_size = get_compile_time_arg_val(6);
constexpr uint32_t scale_slot_bytes = get_compile_time_arg_val(7);  // scratch stride of one token's scale window
constexpr uint32_t rm_zero_bytes = get_compile_time_arg_val(8);     // c_0 total size, zeroed once
constexpr auto combine_accessor_args = TensorAccessorArgs<9>();
constexpr auto weight_accessor_args = TensorAccessorArgs<combine_accessor_args.next_compile_time_args_offset()>();
constexpr auto dispatch_table_accessor_args =
    TensorAccessorArgs<weight_accessor_args.next_compile_time_args_offset()>();
constexpr auto indices_accessor_args =
    TensorAccessorArgs<dispatch_table_accessor_args.next_compile_time_args_offset()>();

constexpr uint32_t TOKENS_PER_CHUNK = 32;
constexpr uint32_t part_row_bytes = part_wt * tt::constants::TILE_WIDTH;  // e4m3: one byte per element
constexpr uint32_t part_scales = part_wt / TILES_PER_SCALE;
constexpr uint32_t stage_bytes = TOKENS_PER_CHUNK * part_row_bytes;

// bf16 weight * fp32 power-of-two scale, as bf16: the scale's unbiased exponent added to the weight's. A zero weight
// stays zero; routing weights and scales are far from the ends of the exponent range.
FORCE_INLINE uint16_t weight_times_pow2(uint16_t weight, uint32_t scale_bits) {
    if ((weight & 0x7F80u) == 0) {
        return 0;
    }
    return static_cast<uint16_t>(weight + ((static_cast<int32_t>((scale_bits >> 23) & 0xFFu) - 127) << 7));
}

// Offset, in tile elements, of column zero for a logical row in the four-face tile layout.
FORCE_INLINE uint32_t tile_col0_offset(uint32_t row) {
    const uint32_t face_row = row % tt::constants::FACE_HEIGHT;
    const uint32_t face_row_group = row / tt::constants::FACE_HEIGHT;
    constexpr uint32_t face_columns = tt::constants::TILE_WIDTH / tt::constants::FACE_WIDTH;
    return face_row_group * face_columns * tt::constants::FACE_HW + face_row * tt::constants::FACE_WIDTH;
}

void kernel_main() {
    const uint32_t combine_addr = get_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_arg_val<uint32_t>(1);
    const uint32_t dispatch_table_addr = get_arg_val<uint32_t>(2);
    const uint32_t indices_addr = get_arg_val<uint32_t>(3);
    uint32_t token_start_idx = get_arg_val<uint32_t>(4);
    const uint32_t num_chunks = get_arg_val<uint32_t>(5);
    const uint32_t part_offset = get_arg_val<uint32_t>(6);        // part's e4m3 byte offset in a row
    const uint32_t scale_read_offset = get_arg_val<uint32_t>(7);  // 64-B aligned start of the scale window
    const uint32_t scale_skip_words = get_arg_val<uint32_t>(8);   // part's first scale within the window
    const uint32_t scale_read_bytes = get_arg_val<uint32_t>(9);   // window size (64-B multiple, within the page)

    Noc noc;
    CircularBuffer cb_rm(cb_rm_id);
    CircularBuffer cb_dispatch_table(cb_dispatch_table_id);
    CircularBuffer cb_indices(cb_indices_id);
    CircularBuffer cb_scale(cb_scale_id);
    CircularBuffer cb_stage_count(cb_stage_count_id);
    CircularBuffer cb_scratch(cb_scratch_id);

    const auto combine_addrg = TensorAccessor(combine_accessor_args, combine_addr);
    const auto weight_addrg = TensorAccessor(weight_accessor_args, weight_addr);
    const auto dispatch_table_addrg =
        TensorAccessor(dispatch_table_accessor_args, dispatch_table_addr, dispatch_table_page_size);
    const auto indices_addrg = TensorAccessor(indices_accessor_args, indices_addr, indices_page_size);
    const uint32_t weight_read_bytes = weight_addrg.get_aligned_page_size();

    // Zero c_0 once: a row with no slot at some stage keeps whatever an earlier stage left there, which is then
    // always finite e4m3 (x * 0 = 0); only never-written L1 could hold a NaN pattern.
    {
        const uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t rm_base = cb_rm.get_write_ptr();
        for (uint32_t off = 0; off < rm_zero_bytes; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros_noc_addr, rm_base + off, MEM_ZEROS_SIZE);
        }
    }

    // Dispatch table (expert -> chip, -1 = not local), read once.
    cb_dispatch_table.reserve_back(1);
    const uint32_t dispatch_table_l1 = cb_dispatch_table.get_write_ptr();
    for (uint32_t i = 0; i < dispatch_table_num_pages; i++) {
        noc.async_read(
            dispatch_table_addrg,
            cb_dispatch_table,
            dispatch_table_aligned_page_size,
            {.page_id = i},
            {.offset_bytes = i * dispatch_table_aligned_page_size});
    }
    noc.async_read_barrier();
    const volatile tt_l1_ptr int32_t* dispatch_table = reinterpret_cast<volatile tt_l1_ptr int32_t*>(dispatch_table_l1);

    // Reader-private scratch: [32 tokens][num_experts] weight pages, then [num_experts stages][32 tokens] scale
    // windows.
    cb_scratch.reserve_back(1);
    const uint32_t weight_scratch = cb_scratch.get_write_ptr();
    const uint32_t scale_scratch = weight_scratch + TOKENS_PER_CHUNK * num_experts * weight_read_bytes;

    cb_indices.reserve_back(1);
    const uint32_t indices_l1 = cb_indices.get_write_ptr();

    uint8_t local_slots[TOKENS_PER_CHUNK][num_experts];
    uint32_t num_local[TOKENS_PER_CHUNK];

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // --- this block's routing: indices -> per-token local slot lists ---
        for (uint32_t t = 0; t < TOKENS_PER_CHUNK; t++) {
            noc.async_read(
                indices_addrg,
                cb_indices,
                indices_aligned_page_size,
                {.page_id = token_start_idx + t},
                {.offset_bytes = t * indices_aligned_page_size});
        }
        noc.async_read_barrier();
        uint32_t num_stages = 1;  // at least one stage, so a block with no local slot still writes zeros
        for (uint32_t t = 0; t < TOKENS_PER_CHUNK; t++) {
            const volatile tt_l1_ptr uint16_t* token_indices =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(indices_l1 + t * indices_aligned_page_size);
            uint32_t n = 0;
            for (uint32_t k = 0; k < num_experts; k++) {
                if (dispatch_table[token_indices[k]] != -1) {
                    local_slots[t][n++] = k;
                }
            }
            num_local[t] = n;
            num_stages = n > num_stages ? n : num_stages;
        }

        // Weights of the local slots (element 0 of each aligned page), all in flight.
        for (uint32_t t = 0; t < TOKENS_PER_CHUNK; t++) {
            for (uint32_t s = 0; s < num_local[t]; s++) {
                const uint32_t k = local_slots[t][s];
                noc.async_read(
                    weight_addrg,
                    use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_scratch),
                    weight_read_bytes,
                    {.page_id = (token_start_idx + t) * num_experts + k},
                    {.offset_bytes = (t * num_experts + k) * weight_read_bytes});
            }
        }

        cb_stage_count.reserve_back(1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_stage_count.get_write_ptr())[0] = num_stages;
        cb_stage_count.push_back(1);

        // --- every stage's part columns and scales: stage s is each token's s-th local slot ---
        cb_rm.reserve_back(num_experts * part_wt);
        for (uint32_t s = 0; s < num_stages; s++) {
            for (uint32_t t = 0; t < TOKENS_PER_CHUNK; t++) {
                if (s < num_local[t]) {
                    const uint32_t page = (token_start_idx + t) * num_experts + local_slots[t][s];
                    noc.async_read(
                        combine_addrg,
                        cb_rm,
                        part_row_bytes,
                        {.page_id = page, .offset_bytes = part_offset},
                        {.offset_bytes = s * stage_bytes + t * part_row_bytes});
                    noc.async_read(
                        combine_addrg,
                        use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_scratch),
                        scale_read_bytes,
                        {.page_id = page, .offset_bytes = scale_read_offset},
                        {.offset_bytes = TOKENS_PER_CHUNK * num_experts * weight_read_bytes +
                                         (s * TOKENS_PER_CHUNK + t) * scale_slot_bytes});
                }
            }
        }
        noc.async_read_barrier();

        for (uint32_t s = 0; s < num_stages; s++) {
            cb_rm.push_back(part_wt);

            cb_scale.reserve_back(part_scales);
            volatile tt_l1_ptr uint16_t* scale_tiles =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_scale.get_write_ptr());
            // Plain (cached) loads after one invalidate, a token's scales gathered before their strided stores (the
            // volatile load-shift-store per element serialized the staging).
            invalidate_l1_cache();
            for (uint32_t t = 0; t < TOKENS_PER_CHUNK; t++) {
                volatile tt_l1_ptr uint16_t* dst = scale_tiles + tile_col0_offset(t);
                if (s < num_local[t]) {
                    const uint32_t k = local_slots[t][s];
                    // bf16 weight = the page's first element.
                    const uint16_t weight = reinterpret_cast<const uint16_t*>(
                        weight_scratch + (t * num_experts + k) * weight_read_bytes)[0];
                    const uint32_t* row_scales = reinterpret_cast<const uint32_t*>(
                                                     scale_scratch + (s * TOKENS_PER_CHUNK + t) * scale_slot_bytes) +
                                                 scale_skip_words;
                    uint32_t v[part_scales];
#pragma GCC unroll 32
                    for (uint32_t j = 0; j < part_scales; j++) {
                        v[j] = row_scales[j];
                    }
#pragma GCC unroll 32
                    for (uint32_t j = 0; j < part_scales; j++) {
                        dst[j * tt::constants::TILE_HW] = weight_times_pow2(weight, v[j]);
                    }
                } else {
                    for (uint32_t j = 0; j < part_scales; j++) {
                        dst[j * tt::constants::TILE_HW] = 0;
                    }
                }
            }
            cb_scale.push_back(part_scales);
        }
        cb_rm.push_back((num_experts - num_stages) * part_wt);
        token_start_idx += TOKENS_PER_CHUNK;
    }
}
