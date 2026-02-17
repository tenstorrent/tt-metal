// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "ttnn/kernel/kernel_utils.hpp"
#include "embeddings_reader_kernel_args.hpp"

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel;

    auto args = make_runtime_struct_from_args<EmbeddingsWriterKernelArgs>();
    constexpr auto c_args = make_compile_time_struct_from_args<CompileTimeEmbeddingsWriterKernelArgs>();

    constexpr auto output_args = TensorAccessorArgs<amount_of_fields(c_args)>();

    const auto output = TensorAccessor(output_args, args.output_buffer_src_addr, c_args.weight_page_size);

    for (uint32_t input_page_id = args.input_page_id; input_page_id < args.input_page_id + args.num_of_pages;
         input_page_id++) {
        auto flat_input_idx = input_page_id * c_args.elems_per_page;
        auto output_page_id = flat_input_idx;
        auto output_pages = output.pages(output_page_id, output_page_id + c_args.elems_per_page);
        auto output_page_iter = output_pages.begin();

        for (uint32_t index = 0; index < c_args.elems_per_page; ++index, ++output_page_iter) {
            cb_wait_front(c_args.output_cb_index, 1);
            uint32_t output_cb_addr = get_read_ptr(c_args.output_cb_index);

            noc_async_write<c_args.weight_page_size>(
                output_cb_addr, output_page_iter->noc_addr(), c_args.weight_page_size);
            noc_async_write_barrier();

            cb_pop_front(c_args.output_cb_index, 1);
        }
    }
}
