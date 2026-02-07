// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/embedding/device/kernels/dataflow/embeddings_common.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "ttnn/kernel/kernel_utils.hpp"
#include "embeddings_reader_kernel_args.hpp"

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel;
    auto args = make_runtime_struct_from_args<EmbeddingsReaderKernelArgs>();
    constexpr auto c_args = make_compile_time_struct_from_args<CompileTimeEmbeddingsReaderKernelArgs>();

    constexpr auto input_args = TensorAccessorArgs<amount_of_fields(c_args)>();
    constexpr auto weights_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();

    // TensorAccessor page size must be aligned to tensor buffer alignment
    // and must be modulo div by input_buf_alignment
    const auto input = TensorAccessor(input_args, args.input_buffer_src_addr, c_args.input_page_size);
    const auto weights = TensorAccessor(weights_args, args.weight_buffer_src_addr, c_args.weight_stick_size);
    const auto output = TensorAccessor(output_args, args.output_buffer_src_addr, c_args.weight_stick_size);

    // No element_size() function
    // constexpr uint32_t input_block_size_bytes = input.dspec().shard_shape()[-1] * input.dspec().element_size();

    DPRINT << "start_shard_id: " << args.start_shard_id << ENDL();
    DPRINT << "num_shards: " << args.num_shards << ENDL();
    DPRINT << "next_shard_offset: " << args.next_shard_offset << ENDL();

    DPRINT << "input.dspec().shard_shape()[0]: " << input.dspec().shard_shape()[0] << ", "
           << input.dspec().shard_shape()[1] << ENDL();
    DPRINT << "input.dspec().shard_volume(): " << input.dspec().shard_volume() << ENDL();
    DPRINT << "elems_per_page: " << c_args.elems_per_page << ENDL();
    DPRINT << "input_buf_alignment: " << c_args.input_buf_alignment << ENDL();

    cb_reserve_back(c_args.cb_id_index, 1);
    uint32_t index_cb_addr = get_write_ptr(c_args.cb_id_index);
    using index_t = uint32_t;
    volatile tt_l1_ptr index_t* index_cb_ptr = reinterpret_cast<volatile tt_l1_ptr index_t*>(index_cb_addr);

    for (uint32_t shard_id = args.start_shard_id, shard_idx = 0; shard_idx < args.num_shards;
         ++shard_idx, shard_id += args.next_shard_offset) {
        DPRINT << "shard_id: " << shard_id << ENDL();
        //
        auto input_shard_pages = input.shard_pages(shard_id);
        auto output_shard_pages = output.shard_pages(shard_id);

        auto output_page_iter = output_shard_pages.begin();
        for (auto input_page_iter = input_shard_pages.begin(); input_page_iter != input_shard_pages.end();
             ++input_page_iter) {
            DPRINT << "input_page_iter page_id: " << input_page_iter->page_id() << ENDL();
            DPRINT << "input_page_iter noc_addr: " << input_page_iter->noc_addr() << ENDL();

            // NoC address must be aligned to tensor buffer alignment, input_page_size
            noc_async_read(input_page_iter->noc_addr(), index_cb_addr, c_args.input_page_size);
            noc_async_read_barrier();

            for (uint32_t index = 0; index < c_args.elems_per_page; ++index, ++output_page_iter) {
                input_token_t token = index_cb_ptr[index];

                DPRINT << "index: " << index << ENDL();
                DPRINT << "token: " << token << ENDL();

                uint64_t weight_noc_addr = get_token_noc_addr(token, weights);
                noc_async_read<c_args.weight_stick_size>(
                    weight_noc_addr, output_page_iter->noc_addr(), c_args.weight_stick_size);
                noc_async_read_barrier();
            }
        }
    }

    DPRINT << "ndsharding embedding DONE: " << ENDL();
}

//    uint32_t index_shard_size_bytes = input.dspec().shard_volume() * input_page_size;
//     for (uint32_t shard_id = start_shard_id, shard_idx = 0; shard_idx < num_shards;
//         ++shard_idx, shard_id += next_shard_offset) {
//        DPRINT << "shard_id: " << shard_id << ENDL();
//        //
//        uint64_t input_noc_addr = input.get_shard_noc_addr(shard_id);
//        uint64_t output_noc_addr = output.get_shard_noc_addr(shard_id);

//        for (auto in_ofs = 0u; in_ofs < index_shard_size_bytes; in_ofs += input_page_size) {
//            DPRINT << "in_ofs: " << in_ofs << ENDL();

//            noc_async_read(input_noc_addr + in_ofs, index_cb_addr, input_page_size);  // no update to index_cb_ptr
//            noc_async_read_barrier();

//            for (uint32_t index = 0; index < elems_per_page; ++index, output_noc_addr += weight_stick_size) {
//                DPRINT << "index: " << index << ENDL();
//                input_token_t token = index_cb_ptr[index];
//                DPRINT << "token: " << token << ENDL();
//                uint64_t weight_noc_addr = get_token_noc_addr(token, weights);
//                noc_async_read<weight_stick_size>(weight_noc_addr, output_noc_addr, weight_stick_size);
//                noc_async_read_barrier();
//            }
//        }
//    }
