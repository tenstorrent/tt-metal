// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated W stream of wq_b_linear (NCRISC, NoC0): the op's 65-core bank-coalesced split, one read per k row
// (core_n_tiles pages adjacent in one bank), depth-1 trid pipeline, whole-slice resident cb_w_stream.
// USE_ACCESSOR=1: reads go through TensorAccessor (the bank's preferred NoC0 endpoint).
// USE_ACCESSOR=0: raw NoC address of the same bank offset on one of the bank's three NOC2AXI ports
//   (TensorAccessor cannot target a non-preferred endpoint). rotate 1: read r uses slot a of its bank;
//   2: slots a / b alternate by r; 3: slot (a + r) % 3. Slot s = the bank's s-th translated port.
// VERIFY=1: write the landed slice back to the same pages of `out` for a bit-exact host check.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

constexpr uint32_t kMaxBanks = 8;

void kernel_main() {
    constexpr uint32_t cb_w_stream = get_compile_time_arg_val(0);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t use_accessor = get_compile_time_arg_val(4);
    constexpr uint32_t verify = get_compile_time_arg_val(5);
    constexpr uint32_t num_banks = get_compile_time_arg_val(6);
    constexpr auto w_args = TensorAccessorArgs<7>();
    constexpr auto o_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t trid = 1;

    uint32_t a = 0;
    const uint32_t w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_start = get_arg_val<uint32_t>(a++);
    const uint32_t core_n_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t n_stride = get_arg_val<uint32_t>(a++);
    const uint32_t slot_bits = get_arg_val<uint32_t>(a++);
    const uint32_t rotate = get_arg_val<uint32_t>(a++);
    const uint32_t slot_bits_b = get_arg_val<uint32_t>(a++);
    // Blackhole translated DRAM coords: a bank's three ports are consecutive y from kDramTranslatedY0, so the
    // bank's block starts at the preferred NoC0 endpoint's y rounded down to a multiple of 3 past y0.
    constexpr uint32_t kDramTranslatedY0 = 12;
    uint32_t port_xy[kMaxBanks][3];
    for (uint32_t b = 0; b < num_banks; ++b) {
        const uint32_t packed = dram_bank_to_noc_xy[0][b];
        const uint32_t x = packed & NOC_NODE_ID_MASK;
        const uint32_t y = (packed >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        const uint32_t y0 = y - (y - kDramTranslatedY0) % 3;
        for (uint32_t s = 0; s < 3; ++s) {
            port_xy[b][s] = x | ((y0 + s) << 16);
        }
    }

    const uint32_t w_page_bytes = get_tile_size(cb_w_stream);
    const auto w = TensorAccessor(w_args, w_addr, w_page_bytes);
    const uint32_t run_bytes = core_n_tiles * w_page_bytes;

    Noc noc;
    CircularBuffer cb_w(cb_w_stream);
    const uint32_t block_tiles = k_block_tiles * core_n_tiles;
    cb_w.reserve_back(num_k_blocks * block_tiles);

    const uint32_t read_tiles = n_stride > 1 ? 1 : core_n_tiles;
    const uint32_t read_bytes = n_stride > 1 ? run_bytes : w_page_bytes;
    uint32_t row_page = n_start;
    uint32_t r = 0;
    auto issue_block = [&](uint32_t offset) {
        for (uint32_t kk = 0; kk < k_block_tiles; ++kk) {
            for (uint32_t n = 0; n < read_tiles; ++n) {
                const uint32_t page = row_page + n;
                if constexpr (use_accessor) {
                    noc.async_read(w, cb_w, read_bytes, {.page_id = page}, {.offset_bytes = offset});
                } else {
                    const uint32_t bank = page % num_banks;
                    const uint32_t q = page / num_banks;
                    const uint32_t slot_a = (slot_bits >> (2 * bank)) & 3;
                    uint32_t slot = slot_a;
                    if (rotate == 3) {
                        slot = (slot_a + r) % 3;
                    } else if (rotate == 2 && (r & 1)) {
                        slot = (slot_bits_b >> (2 * bank)) & 3;
                    }
                    const uint32_t xy = port_xy[bank][slot];
                    const uint64_t src =
                        get_noc_addr(xy & 0xFFFF, xy >> 16, w_addr + q * w_page_bytes + bank_to_dram_offset[bank]);
                    noc.async_read(
                        PrecomposedUnicastEndpoint{}, cb_w, read_bytes, {.noc_addr = src}, {.offset_bytes = offset});
                }
                offset += read_bytes;
            }
            row_page += tensor_n_tiles;
            ++r;
        }
    };

    dataflow_kernel_lib::set_read_trid(noc, trid);
    issue_block(0);
    for (uint32_t b = 0; b < num_k_blocks; ++b) {
        dataflow_kernel_lib::async_read_barrier_with_trid(noc, trid);
        if (b + 1 < num_k_blocks) {
            issue_block(block_tiles * w_page_bytes);
        }
        cb_w.push_back(block_tiles);
    }
    dataflow_kernel_lib::set_read_trid(noc, 0);

    if constexpr (verify) {
        const auto o = TensorAccessor(o_args, out_addr, w_page_bytes);
        cb_w.wait_front(num_k_blocks * block_tiles);
        uint32_t offset = 0;
        uint32_t row_page = n_start;
        for (uint32_t k = 0; k < num_k_blocks * k_block_tiles; ++k) {
            for (uint32_t n = 0; n < core_n_tiles; ++n) {
                noc.async_write(cb_w, o, w_page_bytes, {.offset_bytes = offset}, {.page_id = row_page + n * n_stride});
                offset += w_page_bytes;
            }
            row_page += tensor_n_tiles;
        }
        noc.async_write_barrier();
        cb_w.pop_front(num_k_blocks * block_tiles);
    }
}
