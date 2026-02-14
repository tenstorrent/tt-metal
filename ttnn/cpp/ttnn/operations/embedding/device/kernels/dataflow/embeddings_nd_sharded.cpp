// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/embedding/device/kernels/dataflow/embeddings_common.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "ttnn/kernel/kernel_utils.hpp"
#include "embeddings_reader_kernel_args.hpp"

// kernel is implementing the following logic:
//     output[idx][:] = weights[input[idx]][:];

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel;

    auto args = make_runtime_struct_from_args<EmbeddingsReaderKernelArgs>();
    constexpr auto c_args = make_compile_time_struct_from_args<CompileTimeEmbeddingsReaderKernelArgs>();

    constexpr auto input_args = TensorAccessorArgs<amount_of_fields(c_args)>();
    constexpr auto weights_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();

    const auto input = TensorAccessor(input_args, args.input_buffer_src_addr, c_args.input_page_size);
    const auto weights = TensorAccessor(weights_args, args.weight_buffer_src_addr, c_args.weight_stick_size);
    const auto output = TensorAccessor(output_args, args.output_buffer_src_addr, c_args.weight_stick_size);

    cb_reserve_back(c_args.cb_id_index, 1);
    uint32_t index_cb_addr = get_write_ptr(c_args.cb_id_index);
    volatile tt_l1_ptr input_token_t* index_cb_ptr = reinterpret_cast<volatile tt_l1_ptr input_token_t*>(index_cb_addr);

    uint32_t output_cb_addr = get_write_ptr(c_args.output_cb_index);
    volatile tt_l1_ptr input_token_t* output_cb_ptr =
        reinterpret_cast<volatile tt_l1_ptr input_token_t*>(output_cb_addr);

    for (uint32_t input_page_id = args.input_page_id; input_page_id < args.input_page_id + args.num_of_pages;
         input_page_id++) {
        auto input_pages = input.pages(input_page_id, input_page_id + args.num_of_pages);
        auto input_page_iter = input_pages.begin();

        noc_async_read(input_page_iter->noc_addr(), index_cb_addr, c_args.input_page_size);
        noc_async_read_barrier();

        auto flat_input_idx = input_page_id * c_args.elems_per_page;
        auto output_page_id = flat_input_idx;
        auto output_pages = output.pages(output_page_id, output_page_id + c_args.elems_per_page);
        auto output_page_iter = output_pages.begin();

        for (uint32_t index = 0; index < c_args.elems_per_page; ++index, ++output_page_iter) {
            input_token_t weights_flatten_idx = index_cb_ptr[index];

            uint64_t weight_noc_addr = get_token_noc_addr(weights_flatten_idx, weights);
            noc_async_read<c_args.weight_stick_size>(weight_noc_addr, output_cb_addr, c_args.weight_stick_size);
            noc_async_read_barrier();

            noc_async_write<c_args.weight_stick_size>(
                output_cb_addr, output_page_iter->noc_addr(), c_args.weight_stick_size);
            noc_async_write_barrier();
        }
    }
}
