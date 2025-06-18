// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "dprint_pages.h"

#include "../common.hpp"

inline void print_uint16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << *ptr << " ";
        }
        DPRINT << ENDL();
    }
}

namespace detail{

template <uint32_t DeviceIdx, uint32_t NumMappingPages, uint32_t MappingPageSizeBytes, bool MappingIsDram>
void get_device_expert_indices(
    const InterleavedAddrGen<MappingIsDram>& mapping_addrgen,
    const uint32_t mapping_l1_buffer_addr,
    const uint32_t mapping_page_size,
    volatile tt_l1_ptr uint16_t* output_ptr) {
    auto mapping_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mapping_l1_buffer_addr);

    for (uint32_t expert_idx = 0; expert_idx < NumMappingPages; ++expert_idx) {
        const uint64_t map_page_noc_addr = get_noc_addr(expert_idx, mapping_addrgen);
        noc_async_read(map_page_noc_addr, mapping_l1_buffer_addr,MappingPageSizeBytes);
        noc_async_read_barrier();

        if (mapping_ptr[DeviceIdx] == 1u) {
            *(output_ptr++)=expert_idx;

            // if(DeviceIdx==1){
            //                 DPRINT<<"MAPPING PAGE: "<<expert_idx<<"\n";
            //                 print_uint16_pages(mapping_l1_buffer_addr,mapping_page_size/2,1 );
            //             }
        }
    }
}
}// namespace detail

void kernel_main() {
    constexpr uint32_t mapping_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t local_experts_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t metadata_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t data_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(4);
    constexpr uint32_t batch_size = get_compile_time_arg_val(5);
    constexpr uint32_t num_mapping_pages= get_compile_time_arg_val(6);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(7);
    constexpr uint32_t data_size_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t selected_experts_k = get_compile_time_arg_val(9);
    constexpr uint32_t mapping_page_size_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t metadata_page_size_bytes = get_compile_time_arg_val(11);
    constexpr bool input_is_dram = get_compile_time_arg_val(12);
    constexpr bool mapping_is_dram = get_compile_time_arg_val(13);
    constexpr bool metadata_is_dram = get_compile_time_arg_val(14);

    // if (src_chip_id!=0)return;

    constexpr bool aligned_16 = (data_size_bytes % ALIGN_REQ_16 == 0);

    const auto mapping_tensor_addr = get_arg_val<uint32_t>(0);
    const auto metadata_tensor_addr = get_arg_val<uint32_t>(1);
    const auto input_tensor_addr = get_arg_val<uint32_t>(2);

    InterleavedAddrGen<metadata_is_dram> metadata_addrgen{
        .bank_base_address = metadata_tensor_addr, .page_size = metadata_page_size_bytes};

    InterleavedAddrGen<mapping_is_dram>
        mapping_addrgen{.bank_base_address = mapping_tensor_addr, .page_size = mapping_page_size_bytes};
    InterleavedAddrGen<input_is_dram> data_addrgen{
        .bank_base_address = input_tensor_addr, .page_size = data_size_bytes};

    // this gets sent to writer
    cb_reserve_back(local_experts_cb_id,1);
    auto local_experts_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(local_experts_cb_id));

    // temp buffer just used here
    cb_reserve_back(mapping_cb_id,1);
    uint32_t mapping_buffer_addr = get_write_ptr(mapping_cb_id);
    cb_push_back(mapping_cb_id, 1);

    detail::get_device_expert_indices<src_chip_id, num_mapping_pages, mapping_page_size_bytes, mapping_is_dram>(
        mapping_addrgen, mapping_buffer_addr, mapping_page_size_bytes, local_experts_ptr);
    cb_push_back(local_experts_cb_id,1);

    //     DPRINT<< " local experts for DEVICE: "<<src_chip_id;
    //     for(uint32_t i =0; i< num_local_experts;++i)DPRINT<<" "<< local_experts_ptr[i]<<" ";
    //     DPRINT<<"\n";

    uint32_t input_data_page_idx = 0;
    for(uint32_t b=0;b<batch_size;++b){
        cb_reserve_back(metadata_cb_id,1);
        const uint32_t metadata_l1_addr = get_read_ptr(metadata_cb_id);
        const uint64_t metadata_noc_addr = get_noc_addr(b, metadata_addrgen);
        noc_async_read(metadata_noc_addr, metadata_l1_addr, metadata_page_size_bytes);
        noc_async_read_barrier();
        //         DPRINT<<"READER METADATA: "<<b<<"\n";
        //         print_uint16_pages(metadata_l1_addr,metadata_page_size_bytes/2,1);

        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(metadata_l1_addr);

        for (uint32_t e = 0; e < num_local_experts; ++e) {
            const auto & expert_idx = local_experts_ptr[e];

            // DPRINT<<"READER 2.5 "<< e << " "<< expert_idx<< "\n";
            if (detail::find_if<uint16_t, selected_experts_k, false>(metadata_ptr, expert_idx)) {
                // in the future we can encapsulate this with logic that manages coalescing or splitting pages,
                // taking inspiration from send-receive. This would also require a dedicated receiver kernel.
                // DPRINT<<"READER FOUND b: "<<b<<" e: "<< e<< " expert: " <<expert_idx<< "\n";

                cb_reserve_back(data_cb_id,1);
                const uint32_t data_l1_addr=get_write_ptr(data_cb_id);
                const uint64_t data_noc_addr = get_noc_addr(input_data_page_idx, data_addrgen);
                noc_async_read(data_noc_addr,data_l1_addr,data_size_bytes);
                noc_async_read_barrier();
                // DPRINT<<"READER 4 "<< e<< "\n";

                cb_push_back(data_cb_id, 1);
            }
            ++input_data_page_idx;
        }
        // DPRINT<<"READER 5 "<< b<< "\n";
        cb_push_back(metadata_cb_id,1);
    }
    DPRINT << "READER DONE \n";
}
