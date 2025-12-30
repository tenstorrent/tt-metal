// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t input_page_size = get_compile_time_arg_val(0);
constexpr uint32_t output_page_size = get_compile_time_arg_val(1);
constexpr uint32_t N = get_compile_time_arg_val(2);
#if IS_READER
constexpr uint32_t C = get_compile_time_arg_val(3);
constexpr uint32_t H = get_compile_time_arg_val(4);
#else
constexpr uint32_t C = get_compile_time_arg_val(4);
constexpr uint32_t H = get_compile_time_arg_val(3);
#endif
constexpr uint32_t W = get_compile_time_arg_val(5);
#if IS_ROW_MAJOR
#define TILE_W W
#define TILE_H 1
#else
constexpr uint32_t TILE_W = 32;
constexpr uint32_t TILE_H = 32;
constexpr uint32_t face_shape = 16;
constexpr uint32_t face_offset = face_shape * face_shape * 2;
constexpr uint32_t face_width = face_shape * 2;
constexpr uint32_t full_tile = TILE_H * TILE_W * 2;
#endif

#if IS_READER
#define LOCAL_ACCESSOR output_accessor
#define REMOTE_ACCESSOR input_accessor
#if IS_ROW_MAJOR
#define NOC_SEND(SRC, DST, SIZE) noc_async_read(SRC, static_cast<uint32_t>(DST), SIZE)
#else
#define NOC_SEND(SRC, DST, SIZE) noc_async_read_one_packet(SRC, static_cast<uint32_t>(DST), SIZE)
#endif
#else
#define LOCAL_ACCESSOR input_accessor
#define REMOTE_ACCESSOR output_accessor
#if IS_ROW_MAJOR
#define NOC_SEND(SRC, DST, SIZE) noc_async_write(static_cast<uint32_t>(DST), SRC, SIZE)
#else
#define NOC_SEND(SRC, DST, SIZE) noc_async_write_one_packet(static_cast<uint32_t>(DST), SRC, SIZE)
#endif
#endif

#define SIMPLE
// #define COMPILE_TIME
// #define COMPILE_TIME_SORTED
void kernel_main() {
    size_t arg_idx = 0;
    uint32_t input_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t core_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_pages_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t end_pages_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t working_on_padding = get_arg_val<uint32_t>(arg_idx++);
    uint32_t padding_core_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t padding_start_pages_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t padding_end_pages_id = get_arg_val<uint32_t>(arg_idx++);
    constexpr auto input_args = TensorAccessorArgs<6>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t Wt = (W + TILE_W - 1) / TILE_W;
    constexpr uint32_t Ht = (H + TILE_H - 1) / TILE_H;
    constexpr uint32_t Ct = (C + TILE_H - 1) / TILE_H;
    auto input_accessor = TensorAccessor(input_args, input_address, input_page_size);
    auto output_accessor = TensorAccessor(output_args, output_address, output_page_size);
#ifdef SIMPLE
    auto shard_pages = LOCAL_ACCESSOR.shard_pages(core_id);
    uint32_t page_idx = 0;
    for (const auto& page : shard_pages) {
        if (page_idx >= start_pages_id && page_idx < end_pages_id) {
            uint64_t noc_addr = page.noc_addr();
            uint32_t page_id = page.page_id();
            uint32_t n = page_id / (H * Ct * Wt);
            uint32_t h = page_id / (Ct * Wt) % H;
            uint32_t c = page_id / Wt % Ct;
            uint32_t w = page_id % Wt;
#if IS_ROW_MAJOR
            uint32_t transposed_tile_id = n * C * Ht * Wt + c * Ht * Wt + h * Wt + w;
            uint64_t src_noc_addr = REMOTE_ACCESSOR.get_noc_addr(transposed_tile_id);
            NOC_SEND(src_noc_addr, noc_addr, input_page_size);
#else
            uint32_t h_tile = h / TILE_H;
            uint32_t h_row = h % TILE_H;
            c *= TILE_H;

            uint32_t transpose_offset = h_row * face_width + (h_row / face_shape) * face_offset;
            uint32_t transposed_tile_id_proto = n * C * Ht * Wt + c * Ht * Wt + h_tile * Wt + w;
            for (uint32_t c_row = 0; c_row < std::min(C - c, TILE_H); ++c_row) {
                uint32_t current_offset = c_row * face_width + (c_row / face_shape) * face_offset;
                uint32_t transposed_tile_id = transposed_tile_id_proto + c_row * Ht * Wt;
                uint64_t src_noc_addr = REMOTE_ACCESSOR.get_noc_addr(transposed_tile_id, transpose_offset);

                NOC_SEND(src_noc_addr, noc_addr + current_offset, face_width);
                NOC_SEND(src_noc_addr + face_offset, noc_addr + current_offset + face_offset, face_width);
            }
#endif
        }
        page_idx++;
    }
// #elif defined COMPILE_TIME
//     auto shard_pages = LOCAL_ACCESSOR.shard_pages(core_id);
//     uint32_t page_idx = 0;
//     for (const auto& page : shard_pages) {
//         if (page_idx >= start_pages_id && page_idx < end_pages_id) {
//             uint64_t noc_addr = page.noc_addr();
// #if IS_ROW_MAJOR
//             uint32_t transposed_tile_id = get_arg_val<uint32_t>(arg_idx++);
//             uint64_t src_noc_addr = REMOTE_ACCESSOR.get_noc_addr(transposed_tile_id);
//             NOC_SEND(src_noc_addr, noc_addr, input_page_size);
// #else
//             uint32_t page_id = page.page_id();
//             uint32_t c = (page_id / Wt % Ct) * TILE_H;

//             uint32_t transpose_offset = get_arg_val<uint32_t>(arg_idx++);
//             uint32_t transposed_tile_id_proto = get_arg_val<uint32_t>(arg_idx++);
//             for (uint32_t c_row = 0; c_row < std::min(C - c, TILE_H); ++c_row) {
//                 uint32_t current_offset = c_row * face_width + (c_row / face_shape) * face_offset;
//                 uint32_t transposed_tile_id = transposed_tile_id_proto + c_row * Ht * Wt;
//                 uint64_t src_noc_addr = REMOTE_ACCESSOR.get_noc_addr(transposed_tile_id, transpose_offset);

//                 NOC_SEND(src_noc_addr, noc_addr + current_offset, face_width);
//                 NOC_SEND(src_noc_addr + face_offset, noc_addr + current_offset + face_offset, face_width);
//             }
// #endif
//         }
//         page_idx++;
//     }
#elif defined COMPILE_TIME_SORTED
    // arg_idx += compile_list_size;
    uint32_t transfer_cores = get_arg_val<uint32_t>(arg_idx++);
    for (uint32_t core_idx = 0; core_idx < transfer_cores; ++core_idx) {
        uint32_t num_reads = get_arg_val<uint32_t>(arg_idx++);
        for (uint32_t i = 0; i < num_reads; ++i) {
            uint32_t page_id = get_arg_val<uint32_t>(arg_idx++);
            uint32_t transposed_tile_id = get_arg_val<uint32_t>(arg_idx++);
#if IS_ROW_MAJOR
            uint64_t src_noc_addr = REMOTE_ACCESSOR.get_noc_addr(transposed_tile_id);
            uint64_t dst_noc_addr = LOCAL_ACCESSOR.get_noc_addr(page_id);
            NOC_SEND(src_noc_addr, dst_noc_addr, input_page_size);
#else
            uint32_t current_offset = get_arg_val<uint32_t>(arg_idx++);
            uint32_t transposed_offset = get_arg_val<uint32_t>(arg_idx++);
            uint64_t src_noc_addr0 = REMOTE_ACCESSOR.get_noc_addr(transposed_tile_id, transposed_offset);
            uint64_t src_noc_addr1 = src_noc_addr0 + face_offset;
            uint64_t dst_noc_addr0 = LOCAL_ACCESSOR.get_noc_addr(page_id, current_offset);
            uint64_t dst_noc_addr1 = dst_noc_addr0 + face_offset;
            for (uint32_t w = 0; w < Wt * full_tile; w += full_tile) {
                NOC_SEND(src_noc_addr0 + w, dst_noc_addr0 + w, face_width);
                NOC_SEND(src_noc_addr1 + w, dst_noc_addr1 + w, face_width);
            }
#endif
        }
    }
#else
#error "UNSUPPORTED"
#endif

#if !IS_ROW_MAJOR

    auto out_shard_pages = output_address.shard_pages(padding_core_id);
    uint32_t out_page_idx = 0;
    for (const auto& page : out_shard_pages) {
        if (out_page_idx >= padding_start_pages_id && out_page_idx < padding_end_pages_id) {
            uint64_t noc_addr = page.noc_addr();
            uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
            uint32_t page_id = page.page_id();
            uint32_t c = page_id / Wt % Ct;
            c *= TILE_H;
            uint32_t c_row = C - c;
            uint32_t full_half_tile = (c_row / face_shape);
            if (c_row < TILE_H) {
                uint32_t current_offset = c_row * face_width + full_half_tile * face_offset;
                noc_async_read(
                    zeros_noc_addr,
                    static_cast<uint32_t>(noc_addr + current_offset),
                    (face_shape - c_row % face_shape) * face_width);
                noc_async_read(
                    zeros_noc_addr,
                    static_cast<uint32_t>(noc_addr + current_offset + face_offset),
                    (face_shape - c_row % face_shape) * face_width);
            }
            if (!full_half_tile) {
                noc_async_read(zeros_noc_addr, static_cast<uint32_t>(noc_addr + face_offset * 2), face_offset * 2);
            }
#endif
        }
        out_page_idx++;
    }
#endif

#if IS_READER
    noc_async_read_barrier();
#else
noc_async_write_barrier();
#endif
}
