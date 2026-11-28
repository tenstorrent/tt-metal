// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// host_move_list mover: every transfer is a runtime arg computed on the host. Layout of the args:
//   [n_reads, n_zc_writes, n_copies, n_img_writes]
//   read / write : bank mode  {bank | len << 8, bank_local_off (slot * page + off), l1_off}
//                  page mode  {page, off | len << 16, l1_off}
//   copy         : {stage_off, image_off, len}
// Reads land in staging (L1 == src mod 64); zero-copy writes source staging directly (L1 == dst mod 16);
// pieces whose phase does not match are copied into the image, then written.
// Raw LLK/NoC (bypasses the TensorAccessor walk + dataflow helpers): the point is to measure what is left
// once the host did all the address arithmetic.

#include "hml_common.hpp"

void kernel_main() {
    constexpr uint32_t kElem = get_compile_time_arg_val(0);
    constexpr bool kBank = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t cb_stage = get_compile_time_arg_val(2);
    constexpr uint32_t cb_image = get_compile_time_arg_val(3);
    constexpr uint32_t kPageBytes = get_compile_time_arg_val(4);
    constexpr auto in_args = TensorAccessorArgs<5>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(0));
    const uint32_t n_reads = a[0];
    const uint32_t n_zc = a[1];
    const uint32_t n_copies = a[2];
    const uint32_t n_imgw = a[3];
    if (n_reads == 0) {
        return;
    }
    a += 4;
    const uint8_t noc = noc_index;
    const uint32_t stage = (get_write_ptr(cb_stage) + 63) & ~63u;
    const uint32_t image = (get_write_ptr(cb_image) + 63) & ~63u;
    const auto in = TensorAccessor(in_args, in_addr, kPageBytes);
    const auto out = TensorAccessor(out_args, out_addr, kPageBytes);

    auto dram_in = [&](uint32_t w0, uint32_t w1, uint32_t& len) -> uint64_t {
        if constexpr (kBank) {
            len = w0 >> 8;
            return get_noc_addr_from_bank_id<true>(w0 & 0xFF, in_addr + w1, noc);
        } else {
            len = w1 >> 16;
            return in.get_noc_addr(w0, w1 & 0xFFFF, noc);
        }
    };
    auto dram_out = [&](uint32_t w0, uint32_t w1, uint32_t& len) -> uint64_t {
        if constexpr (kBank) {
            len = w0 >> 8;
            return get_noc_addr_from_bank_id<true>(w0 & 0xFF, out_addr + w1, noc);
        } else {
            len = w1 >> 16;
            return out.get_noc_addr(w0, w1 & 0xFFFF, noc);
        }
    };

    {
        HML_ZONE("hml.reads");
        for (uint32_t i = 0; i < n_reads; ++i, a += 3) {
            uint32_t len;
            const uint64_t src = dram_in(a[0], a[1], len);
            noc_async_read(src, stage + a[2], len, noc);
        }
    }
    {
        HML_ZONE("hml.await");
        noc_async_read_barrier(noc);
    }
    {
        HML_ZONE("hml.writes");
        for (uint32_t i = 0; i < n_zc; ++i, a += 3) {
            uint32_t len;
            const uint64_t dst = dram_out(a[0], a[1], len);
            noc_async_write(stage + a[2], dst, len, noc);
        }
        if (n_copies > 0) {
            invalidate_l1_cache();
            uint32_t last = image;
            for (uint32_t i = 0; i < n_copies; ++i, a += 3) {
                hml::copy_run<kElem>(stage + a[0], image + a[1], a[2]);
                last = image + a[1] + a[2] - 1;
            }
            (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(last & ~3u));
            for (uint32_t i = 0; i < n_imgw; ++i, a += 3) {
                uint32_t len;
                const uint64_t dst = dram_out(a[0], a[1], len);
                noc_async_write(image + a[2], dst, len, noc);
            }
        }
    }
    {
        HML_ZONE("hml.drain");
        noc_async_write_barrier(noc);
    }
}
