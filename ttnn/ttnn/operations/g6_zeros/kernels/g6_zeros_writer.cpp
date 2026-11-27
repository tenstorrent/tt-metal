// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_zeros writer, regime bank_run_write (op_design.md, Blocking Model).
//
// The output's pages are numbered bank-major: position q enumerates bank 0's slots, then bank 1's,
// and so on. This core owns positions [q_begin, q_begin + q_count). Slots k, k+1, ... of one bank are
// contiguous bytes, so every bank the range touches is one run: one contiguous DRAM range, written in
// packets that all read the same zeroed L1 source.
//
// Raw NoC writes instead of a kernel_lib helper: no helper writes a multi-page DRAM range from a
// constant L1 source. Noc::async_write_zeros overload 2 stays inside one page
// (noc_zero_dram.inl:24), so a run would become run_slots single-page writes. A page-range overload
// of it would make write_zero_run one helper call.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

namespace {

struct BankRun {
    uint64_t noc_addr;
    uint32_t bytes;
    uint32_t slots;
};

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_zero_src = get_compile_time_arg_val(0);
    constexpr uint32_t zero_src_bytes = get_compile_time_arg_val(1);
    constexpr auto output_args = TensorAccessorArgs<2>();

    constexpr uint32_t aligned_page_bytes = output_args.get_aligned_page_size();
    constexpr uint32_t num_banks = NUM_DRAM_BANKS;
    // A one-packet write must fit one burst and must not read past the zeroed bytes.
    constexpr uint32_t packet_bytes = zero_src_bytes < NOC_MAX_BURST_SIZE ? zero_src_bytes : NOC_MAX_BURST_SIZE;
    // BH DRAM writes need 16 B granularity; every tile page is a multiple of 64 B.
    static_assert(
        zero_src_bytes % 64 == 0 && aligned_page_bytes % 64 == 0, "zero source and pages must be 64 B multiples");

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tensor_pages = get_common_arg_val<uint32_t>(1);
    const uint32_t q_begin = get_arg_val<uint32_t>(0);
    const uint32_t q_count = get_arg_val<uint32_t>(1);

    if (q_count == 0) {
        return;
    }

    Noc noc;
    CircularBuffer zero_src(cb_zero_src);
    {
        DeviceZoneScopedN("make_zero_source");
        // A fresh CB has read_ptr == write_ptr; the source is never pushed, so it stays this kernel's own.
        noc.async_write_zeros(zero_src, zero_src_bytes);
    }

    const auto output = TensorAccessor(output_args, output_addr);
    const uint32_t src_addr = zero_src.get_write_ptr();

    // Banks [0, full_banks) hold slots_per_bank slots; the rest hold one fewer.
    const uint32_t slots_per_bank = (tensor_pages + num_banks - 1) / num_banks;
    const uint32_t full_banks = tensor_pages - num_banks * (slots_per_bank - 1);
    const uint32_t full_span = full_banks * slots_per_bank;
    const uint32_t q_end = q_begin + q_count;

    auto run_at = [&](uint32_t q) -> BankRun {
        uint32_t bank, slot, bank_slots;
        if (q < full_span) {
            bank = q / slots_per_bank;
            slot = q - bank * slots_per_bank;
            bank_slots = slots_per_bank;
        } else {
            // Only reached when slots_per_bank >= 2: with one slot per bank, full_span == tensor_pages.
            bank_slots = slots_per_bank - 1;
            const uint32_t bank_offset = (q - full_span) / bank_slots;
            slot = (q - full_span) - bank_offset * bank_slots;
            bank = full_banks + bank_offset;
        }
        const uint32_t run_slots = (q_end - q) < (bank_slots - slot) ? (q_end - q) : (bank_slots - slot);
        return BankRun{output.get_noc_addr(slot * num_banks + bank), run_slots * aligned_page_bytes, run_slots};
    };

    // The first run's address math overlaps the zero fill.
    uint32_t q = q_begin;
    BankRun run = run_at(q);

    {
        DeviceZoneScopedN("wait_zero_source");
        noc.write_zeros_l1_barrier();
    }

    while (true) {
        {
            DeviceZoneScopedN("write_zero_run");
            const uint32_t full_packets = run.bytes / packet_bytes;
            const uint32_t tail_bytes = run.bytes - full_packets * packet_bytes;
            // DRAM bank offsets stay below 4 GiB, so the low word carries the in-bank address.
            uint32_t dst_addr = static_cast<uint32_t>(run.noc_addr);
            if (full_packets) {
                noc_async_write_one_packet_set_state(run.noc_addr, packet_bytes);
                for (uint32_t p = 0; p < full_packets; ++p) {
                    noc_async_write_one_packet_with_state(src_addr, dst_addr);
                    dst_addr += packet_bytes;
                }
            }
            if (tail_bytes) {
                noc_async_write_one_packet_set_state(run.noc_addr + full_packets * packet_bytes, tail_bytes);
                noc_async_write_one_packet_with_state(src_addr, dst_addr);
            }
        }
        q += run.slots;
        if (q >= q_end) {
            break;
        }
        run = run_at(q);
    }

    {
        DeviceZoneScopedN("drain_writes");
        noc_async_write_barrier();
    }
}
