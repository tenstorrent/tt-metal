// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isin_common.hpp"

#include "api/dataflow/dataflow_api.h"

#include <algorithm>
#include <numeric>

namespace {

/*
    This function compares two subchunks of data - one chunk is one stick, and one subchunk
    is the maximal length of a stick that can fit as much L1 memory as possible - this length
    is shared betwen rows of `elements`, `test_elements` and `output` tensors.
*/
template <typename elements_number_type>
FORCE_INLINE void isin_subchunks(
    uint32_t elements_l1_read_addr,
    uint32_t test_elements_l1_read_addr,
    uint32_t output_l1_write_addr,
    uint32_t elements_subchunk_size,
    uint32_t test_elements_subchunk_size,
    bool invert) {
    volatile tt_l1_ptr elements_number_type* elements_subchunk_ptr =
        reinterpret_cast<volatile tt_l1_ptr elements_number_type*>(elements_l1_read_addr);
    volatile tt_l1_ptr elements_number_type* test_elements_subchunk_ptr =
        reinterpret_cast<volatile tt_l1_ptr elements_number_type*>(test_elements_l1_read_addr);
    volatile tt_l1_ptr elements_number_type* output_subchunk_ptr =
        reinterpret_cast<volatile tt_l1_ptr elements_number_type*>(output_l1_write_addr);
    for (uint32_t elements_index = 0; elements_index < elements_subchunk_size; ++elements_index) {
        for (uint32_t test_elements_index = 0; test_elements_index < test_elements_subchunk_size;
             ++test_elements_index) {
            if (elements_subchunk_ptr[elements_index] == test_elements_subchunk_ptr[test_elements_index]) {
                output_subchunk_ptr[elements_index] = invert ? 0x00000000 : 0xFFFFFFFF;
                break;
            }
        }
    }
}

void zero_buffer(uint32_t write_addr, int bytes) {
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        uint32_t curr_bytes = std::min(bytes, MEM_ZEROS_SIZE);
        noc_async_read(zeros_noc_addr, write_addr, curr_bytes);
        write_addr += curr_bytes;
        bytes -= curr_bytes;
    }
    noc_async_read_barrier();
}

/*
    When a core begins processing of a given chunk, the output chunk is fully erased with zeroes
    (or filled with ones, if the invert flag is set) - this makes sure that the output chunk is properly filled
    since the algorithm marks only the elements from `elements` tensor that are encountered within
    `test_elements` tensor, so the output chunk get later updated only with the reverted values and
    then retuened to DRAM
*/
template <typename elements_number_type>
FORCE_INLINE void prefill_output(uint32_t output_l1_write_addr, uint32_t output_subchunk_size, bool invert) {
    if (invert) {
        volatile tt_l1_ptr elements_number_type* output_chunk_begin_ptr =
            reinterpret_cast<volatile tt_l1_ptr elements_number_type*>(output_l1_write_addr);
        for (uint32_t i = 0; i < output_subchunk_size; ++i) {
            output_chunk_begin_ptr[i] = 0xFFFFFFFF;
        }
    } else {
        zero_buffer(output_l1_write_addr, output_subchunk_size * sizeof(uint32_t));
        //         void zero_buffer(uint32_t write_addr, int bytes) {
        //     uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
        //     while (bytes > 0) {
        //         uint32_t curr_bytes = std::min(bytes, MEM_ZEROS_SIZE);
        //         noc_async_read(zeros_noc_addr, write_addr, curr_bytes);
        //         write_addr += curr_bytes;
        //         bytes -= curr_bytes;
        //     }
        //     noc_async_read_barrier();
        // }
    }
}

}  // namespace

/*
    this kernel works with row-major data that has been flattened to a 1D tensor to save on redundant padding
    the input tensors to the kernel (`elements` and `test_elements`) (and also output) are flattened to become one stick
   only this means that the whole physical volume of input (and output) may not easily fit in L1 because DRAM calls are
   costly, it is assumed (in the least optimized case) that it is feasible to subchunk data to have as long subchunks of
   the only stick the input tensors consist of as possible
*/
void kernel_main() {
    constexpr auto ctas = get_ctas();

    const uint32_t elements_buffer_address = get_arg_val<uint32_t>(0);
    const uint32_t test_elements_buffer_address = get_arg_val<uint32_t>(1);
    const uint32_t subchunks_per_core = get_arg_val<uint32_t>(2);
    const uint32_t subchunks_offset = get_arg_val<uint32_t>(3);

    using elements_number_type = std_type_t<get_dataformat(ctas.elements_cb)>;

    constexpr uint32_t elements_element_size = ctas.elements_tensor_datum_size;
    constexpr uint32_t test_elements_element_size = ctas.elements_tensor_datum_size;
    const auto elements_addr_gtor = TensorAccessor{
        ctas.elements_accessor_args, elements_buffer_address, ctas.elements_size * elements_element_size};
    const auto test_elements_addr_gtor = TensorAccessor{
        ctas.test_elements_accessor_args,
        test_elements_buffer_address,
        ctas.test_elements_size * test_elements_element_size};

    /*
        for every subchunk (part of a stick) of the elements tensor - to which an analogous output chunk
        is related to - is fully processed by the core
        there is the fact that tensors are flattened before fed to kernels, so they consist only of one stick
        each core is provided a subset of subchunks of the element tensor to be processed only by that given core -
        - this means that each core needs to go through the whole test_elements tensor (all subchunks) to
        correctly work out the output values (yes/no) for each of according input elements - subject to optimizationś
    */
    for (uint32_t elements_subchunk_id = subchunks_offset,
                  elements_offset = subchunks_offset * ctas.single_fetch_subchunk_size;
         elements_subchunk_id < subchunks_offset + subchunks_per_core;
         ++elements_subchunk_id, elements_offset += ctas.single_fetch_subchunk_size) {
        // either the maximal defined single fetch size or the remainder, which is less than that number
        const uint32_t elements_subchunk_size =
            std::min(ctas.elements_size - elements_offset, ctas.single_fetch_subchunk_size);
        load_to_cb(
            ctas.elements_cb, elements_addr_gtor, elements_offset, elements_subchunk_size, elements_element_size);
        cb_wait_front(ctas.elements_cb, ONE_PAGE);
        // prepare output mask for writing
        cb_reserve_back(ctas.output_cb, ONE_PAGE);
        const uint32_t elements_l1_read_addr = get_read_ptr(ctas.elements_cb);
        const uint32_t output_l1_write_addr = get_write_ptr(ctas.output_cb);
        prefill_output<elements_number_type>(output_l1_write_addr, elements_subchunk_size, ctas.invert);

        // for every subchunk of the test_elements stick
        for (uint32_t test_elements_subchunk_id = 0, test_elements_offset = 0;
             test_elements_offset < ctas.test_elements_size;
             ++test_elements_subchunk_id, test_elements_offset += ctas.single_fetch_subchunk_size) {
            // same as for elements_subchunk_size
            const uint32_t test_elements_subchunk_size =
                std::min(ctas.test_elements_size - test_elements_offset, ctas.single_fetch_subchunk_size);
            load_to_cb(
                ctas.test_elements_cb,
                test_elements_addr_gtor,
                test_elements_offset,
                test_elements_subchunk_size,
                test_elements_element_size);
            cb_wait_front(ctas.test_elements_cb, ONE_PAGE);
            const uint32_t test_elements_l1_read_addr = get_read_ptr(ctas.test_elements_cb);

            // exhaustively perform isin on a given elements' subchunks (one elements' subchunk vs all test_elements'
            // subchunks)
            isin_subchunks<elements_number_type>(
                elements_l1_read_addr,
                test_elements_l1_read_addr,
                output_l1_write_addr,
                elements_subchunk_size,
                test_elements_subchunk_size,
                ctas.invert);

            cb_pop_front(ctas.test_elements_cb, ONE_PAGE);
        }

        // push the output subchunk once it's been checked against test_elements
        cb_push_back(ctas.output_cb, ONE_PAGE);
        cb_pop_front(ctas.elements_cb, ONE_PAGE);
    }
}
