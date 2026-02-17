// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/embedding/device/kernels/dataflow/embeddings_common.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "ttnn/kernel/kernel_utils.hpp"
#include "embeddings_reader_kernel_args.hpp"

//  output[idx][:] = weights[input[idx]][:];

FORCE_INLINE uint32_t
logical_to_tile_storage_index(uint32_t logical_idx, uint32_t tile_width, uint32_t face_height, uint32_t face_width) {
    uint32_t row = logical_idx / tile_width;
    uint32_t col = logical_idx % tile_width;
    uint32_t faces_per_row = tile_width / face_width;
    uint32_t face_row = row / face_height;
    uint32_t face_col = col / face_width;
    uint32_t face_id = face_row * faces_per_row + face_col;
    uint32_t sub_row = row % face_height;
    uint32_t sub_col = col % face_width;
    uint32_t face_hw = face_height * face_width;
    return face_id * face_hw + sub_row * face_width + sub_col;
}

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel;

    auto args = make_runtime_struct_from_args<EmbeddingsReaderKernelArgs>();
    constexpr auto c_args = make_compile_time_struct_from_args<CompileTimeEmbeddingsReaderKernelArgs>();

    constexpr auto input_args = TensorAccessorArgs<amount_of_fields(c_args)>();
    constexpr auto weights_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    const auto input = TensorAccessor(input_args, args.input_buffer_src_addr, c_args.input_page_size);
    const auto weights = TensorAccessor(weights_args, args.weight_buffer_src_addr, c_args.weight_page_size);

    cb_reserve_back(c_args.input_cb_index, 1);
    uint32_t index_cb_addr = get_write_ptr(c_args.input_cb_index);
    volatile tt_l1_ptr input_token_t* index_cb_ptr = reinterpret_cast<volatile tt_l1_ptr input_token_t*>(index_cb_addr);

    for (uint32_t input_page_id = args.input_page_id; input_page_id < args.input_page_id + args.num_of_pages;
         input_page_id++) {
        auto input_pages = input.pages(input_page_id, input_page_id + args.num_of_pages);
        auto input_page_iter = input_pages.begin();

        noc_async_read(input_page_iter->noc_addr(), index_cb_addr, c_args.input_page_size);
        noc_async_read_barrier();

        for (uint32_t index = 0; index < c_args.elems_per_page; ++index) {
            uint32_t storage_index = index;
            if (c_args.input_is_tile_layout) {
                storage_index =
                    logical_to_tile_storage_index(index, c_args.tile_width, c_args.face_height, c_args.face_width);
            }
            input_token_t weights_flatten_idx = index_cb_ptr[storage_index];

            cb_reserve_back(c_args.output_cb_index, 1);
            uint32_t output_cb_addr = get_write_ptr(c_args.output_cb_index);

            uint64_t weight_noc_addr = get_token_noc_addr(weights_flatten_idx, weights);
            noc_async_read<c_args.weight_page_size>(weight_noc_addr, output_cb_addr, c_args.weight_page_size);
            noc_async_read_barrier();

            cb_push_back(c_args.output_cb_index, 1);
        }
    }
}
