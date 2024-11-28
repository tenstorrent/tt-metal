// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


/*
Function reads from RM and writes to RM

Assumptions:

Compile arguments
0. src0_is_dram: 1 if source is dram else 0
1. read_size_is_pow2: 1 if read size is power of 2 else 0
2. log_base_2_of_page_size: log base 2 of page size
3. write_size_is_pow2: 1 if write size is power of 2 else 0
4. log_base_2_of_page_size: log base 2 of page size
5. needs_read_allignment: 1 if read needs allignment else 0
//Needed if BRAM and page size is not multiple of 64 bytes

Runtime arguments
0. src_addr: source address
1. dst_addr: destination address
2. source_page_size_bytes: source page size in bytes
3. dest_page_size_bytes: destination page size in bytes
4. source_read_size_bytes: source read size in bytes
5. read_start_page: read start page
6. read_end_page: read end page
7. write_start_page: write start page
*/
#include <stdint.h>
#include "dataflow_api.h"
#include <stdio.h>
#include <cstring>
#include "debug/dprint.h"  // required in all kernels using DPRINT

#define MASK_64      0xFFFFFFFFFFFFFFC0
#define OFFSET_64    0x000000000000003F
#define MASK_16      0xFFFFFFFFFFFFFFF0
#define OFFSET_16    0x000000000000000F


template <bool guaranteed_16B_alligned, bool read_async>
FORCE_INLINE
void tt_memmove (
    const uint32_t dst_l1_addr,
    const uint64_t src_l1_addr,
    const uint32_t bytes)
{
    //Uses noc_async_read when possible to copy the data over
    if constexpr (guaranteed_16B_alligned)
    {
        noc_async_read(get_noc_addr(src_l1_addr),dst_l1_addr, bytes);
        noc_async_read_barrier();
    }
    else
    {
        if ((dst_l1_addr&OFFSET_16) == (src_l1_addr&OFFSET_16))
        {
            noc_async_read(get_noc_addr(src_l1_addr),dst_l1_addr, bytes);
            noc_async_read_barrier();
        }
        else
        {
            memmove((void *)(dst_l1_addr), (void *)(src_l1_addr), (size_t) (bytes));
        }
    }
}


void kernel_main() {
    //We are guranteed to be in 2D going to 2D

    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr                 = get_arg_val<uint32_t>(1);
    const uint32_t source_page_size_bytes   = get_arg_val<uint32_t>(2);
    const uint32_t dest_page_size_bytes     = get_arg_val<uint32_t>(3);
    //If DDR this is source_page_size_bytes + 64 (rounded up to next 64B), if L1 this is source_page_size_bytes + 16 (rounded up to next 16B)
    const uint32_t source_read_size_bytes   = get_arg_val<uint32_t>(4);
    const uint32_t read_start_page          = get_arg_val<uint32_t>(5);
    const uint32_t read_end_page            = get_arg_val<uint32_t>(6);
    const uint32_t write_start_page         = get_arg_val<uint32_t>(7);
    //cb_id_in0 is a circular buffer with 1 source_page_size_bytes page if no alignment needed
    //source_read_size_bytes otherwise
    const uint32_t cb_id_in0                = get_arg_val<uint32_t>(8);
    //cb_id_in1 is a circular buffer with 1 dest_page_size_bytes+16 (rounded up to next 64B) page
    const uint32_t cb_id_in1                = get_arg_val<uint32_t>(9);


    constexpr bool tensor_is_dram                   = get_compile_time_arg_val(0) == 1;
    #define src_aligned_to_64                       get_compile_time_arg_val<uint32_t>(1) == 1
    #define src_aligned_to_16                       get_compile_time_arg_val<uint32_t>(2) == 1
    #define dst_aligned_to_16                       get_compile_time_arg_val<uint32_t>(3) == 1


    const InterleavedAddrGen<tensor_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = source_page_size_bytes
    };

    const InterleavedAddrGen<tensor_is_dram> d = {
        .bank_base_address = dst_addr,
        .page_size = dest_page_size_bytes
    };


    uint32_t read_offset = 0;
    uint32_t write_page = write_start_page;
    uint32_t readable = 0;
    uint32_t transaction = 0;
    uint32_t writable = dest_page_size_bytes;
    //cb_id_in0 is a CB source_read_size_bytes page size, 1 page
    //cb_id_in1 is a CB dest_page_size_bytes + allignment_to_64 page size, 1 page
    cb_reserve_back(cb_id_in0, 1);
    cb_reserve_back(cb_id_in1, 1);
    const uint32_t source_buffer = get_write_ptr(cb_id_in0);
    const uint32_t dest_buffer = get_write_ptr(cb_id_in1);

    uint64_t dst_noc_addr = get_noc_addr(write_page, d);
#if (dst_aligned_to_16)
    uint32_t write_offset = 0;
#else
    uint32_t write_offset = dst_noc_addr&OFFSET_16;
    uint32_t begin_write_offset = write_offset;
#endif
    for (uint32_t i = read_start_page; i <= read_end_page; i++) {
        //Read from source
        uint64_t src_noc_addr = s.get_noc_addr(i,0);

#if (src_aligned_to_64 || ((!tensor_is_dram) && src_aligned_to_16))
        //Aligned to 64 bytes or 16 bytes but L1
        noc_async_read(src_noc_addr, source_buffer, source_page_size_bytes);
        read_offset = 0;
#elif (tensor_is_dram)
        //DDR but not alligned to 64 (potentially also not alligned to 16)
        noc_async_read(src_noc_addr&MASK_64, source_buffer, source_read_size_bytes);
        read_offset = src_noc_addr&OFFSET_64;
#else
        //L1 but not alligned to 16
        noc_async_read(src_noc_addr&MASK_16, source_buffer, source_read_size_bytes);
        read_offset = src_noc_addr&OFFSET_16;
#endif
        readable = source_page_size_bytes;
        noc_async_read_barrier();

        //Write to dest
        while (readable > 0)
        {
            noc_async_write_barrier();
            if (readable < writable)
            {
                tt_memmove<false,true>(dest_buffer+write_offset, source_buffer + read_offset, readable);
                writable = writable -readable;
                write_offset = write_offset + readable;
                readable = 0;
            }
            else if (readable == writable)
            {
                tt_memmove<false,false>(dest_buffer+write_offset, source_buffer + read_offset, readable);
#if ((dst_aligned_to_16))
                noc_async_write(dest_buffer,dst_noc_addr, dest_page_size_bytes);
#else
                noc_async_write(dest_buffer+begin_write_offset,dst_noc_addr, dest_page_size_bytes);
#endif
                writable = dest_page_size_bytes;
                readable = 0;
                if (i == read_end_page-1)
                {
                    cb_push_back(cb_id_in0, 1);
                    cb_push_back(cb_id_in1, 1);
                    return;
                }
                write_page++;
                dst_noc_addr = get_noc_addr(write_page, d);
#if ((dst_aligned_to_16))
                write_offset=0;
#else
                write_offset = dst_noc_addr&OFFSET_16;
                begin_write_offset = write_offset;
#endif
            }
            else
            {
                //writable < readable

                tt_memmove<false,false>(dest_buffer+write_offset, source_buffer + read_offset, writable);
#if ((dst_aligned_to_16))
                noc_async_write(dest_buffer,dst_noc_addr, dest_page_size_bytes);
#else
                noc_async_write(dest_buffer+begin_write_offset,dst_noc_addr, dest_page_size_bytes);
#endif
                readable = readable - writable;
                read_offset = read_offset + writable;
                write_page++;
                dst_noc_addr = get_noc_addr(write_page, d);
#if ((dst_aligned_to_16))
                write_offset=0;
#else
                write_offset = dst_noc_addr&OFFSET_16;
                begin_write_offset = write_offset;
#endif
                writable = dest_page_size_bytes;
            }
        }
    }
    cb_push_back(cb_id_in0, 1);
    cb_push_back(cb_id_in1, 1);
    return;
}
