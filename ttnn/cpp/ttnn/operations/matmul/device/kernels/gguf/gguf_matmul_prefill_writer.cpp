// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Input and output mover of the GGUF prefill matmul (gguf_matmul_prefill_program_factory.cpp). The cores of a band
// (whole grid rows) share the band's M rows. The input goes to them in blocks of ur rows x kb K tiles (per M chunk, per
// K block, per ur rows) through a ring of `slots` blocks in c_1, at the same address on every core. Block b comes from
// sender b % senders (two: their writers use different NoCs, so two multicasts can be in flight). A sender reads each
// of its blocks from DRAM into its own slot once the slot is free, waits until every core of the band has counted
// itself ready for the block in the slot's ready semaphore on that sender (a count per slot, reset by the sender before
// it multicasts into the slot, so that early counts of fast cores never stand in for a slow core), multicasts the
// block and then its count (b / senders + 1) through the same command buffer, so the count lands after the data.
// A core counts itself ready for block j as soon as block j - slots has been consumed (an idle core: received). The
// output (c_16, subblocks of sh x sw tiles in compute order) is written to DRAM whenever this kernel would wait. K
// split: band 1 sends its fp32 sums per unit of rows (c_17) into its partner's c_18 in band 0, a ring of two units:
// the partner grants each slot once its compute has freed it (counted in the sender's partial semaphore) and counts
// the units that landed in its own.
// Compile args: K tiles, N tiles, mc, kb, K blocks, M chunks, slots (<= 4), band cores, four slot ready semaphores, two
// block count semaphores, a source semaphore, senders (1 or 2), ur, the partial semaphore, then the input's and the
// output's TensorAccessorArgs. Runtime args: input address, output address, band rectangle (NoC x / y start, x / y
// end, in this writer's NoC order), first M tile of the band, first N tile, N tiles (0: no output), sh, sw, role (0, or
// 1 + the sender's index), the two senders' NoC x, y, first K tile of the band, K split role (0: none, 1: send sums,
// 2: add the partner's), the partner's NoC x, y, the plan's N tiles per core (a unit of sums is ur x that).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t mc = get_compile_time_arg_val(2);
    constexpr uint32_t kb = get_compile_time_arg_val(3);
    constexpr uint32_t kblocks = get_compile_time_arg_val(4);
    constexpr uint32_t mchunks = get_compile_time_arg_val(5);
    constexpr uint32_t slots = get_compile_time_arg_val(6);
    constexpr uint32_t band_cores = get_compile_time_arg_val(7);
    // Ready counts per slot (on the slot's sender of the moment; reset by it before each multicast into the slot).
    constexpr uint32_t ready_sems[4] = {
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11)};
    constexpr uint32_t count_sem0 = get_compile_time_arg_val(12), count_sem1 = get_compile_time_arg_val(13);
    constexpr uint32_t source_sem = get_compile_time_arg_val(14);
    constexpr uint32_t senders = get_compile_time_arg_val(15);
    constexpr uint32_t ur = get_compile_time_arg_val(16);
    constexpr uint32_t partial_sem = get_compile_time_arg_val(17);
    static_assert(slots <= 4);
    constexpr auto input_args = TensorAccessorArgs<18>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048, block_tiles = ur * kb, block_bytes = block_tiles * tile_bytes;
    constexpr uint32_t units = mc / ur;  // per M chunk per K block
    constexpr uint32_t blocks = mchunks * kblocks * units;
    constexpr uint32_t cb_in0 = 1, cb_out = 16, cb_send = 17, cb_recv = 18;

    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(1));
    const uint32_t x0 = get_arg_val<uint32_t>(2), y0 = get_arg_val<uint32_t>(3);
    const uint32_t x1 = get_arg_val<uint32_t>(4), y1 = get_arg_val<uint32_t>(5);
    const uint32_t m_base = get_arg_val<uint32_t>(6);
    const uint32_t n0 = get_arg_val<uint32_t>(7), nc = get_arg_val<uint32_t>(8);
    const uint32_t sh = get_arg_val<uint32_t>(9), sw = get_arg_val<uint32_t>(10);
    const uint32_t role = get_arg_val<uint32_t>(11);  // 0: not a sender; s + 1: sender of blocks b % senders == s
    const uint32_t count_sem[2] = {count_sem0, count_sem1};
    uint32_t count_addr[2];
    volatile tt_l1_ptr uint32_t* count_ptr[2];
    for (uint32_t s = 0; s < 2; s++) {
        count_addr[s] = get_semaphore(count_sem[s]);
        count_ptr[s] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(count_addr[s]);
    }
    auto ready_noc = [&](uint32_t j) {
        const uint32_t s = j % senders;
        return get_noc_addr(
            get_arg_val<uint32_t>(12 + 2 * s), get_arg_val<uint32_t>(13 + 2 * s), get_semaphore(ready_sems[j % slots]));
    };
    const bool active = nc > 0;
    // K split in two bands: band 1 (krole 1) sends its fp32 sums per unit to the core of band 0 with the same N tiles
    // (krole 2), which adds them.
    const uint32_t k_base = get_arg_val<uint32_t>(16), krole = get_arg_val<uint32_t>(17);
    const uint64_t partner_sem =
        get_noc_addr(get_arg_val<uint32_t>(18), get_arg_val<uint32_t>(19), get_semaphore(partial_sem));
    const uint32_t partner_x = get_arg_val<uint32_t>(18), partner_y = get_arg_val<uint32_t>(19);
    volatile tt_l1_ptr uint32_t* partial_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(partial_sem));
    const uint32_t unit_tiles = ur * get_arg_val<uint32_t>(20), unit_bytes = unit_tiles * 4096;  // ur x the planned nc
    const uint32_t partial_units = active && krole ? mchunks * units : 0;
    uint32_t units_sent = 0, units_got = 0, units_granted = 0;
    constexpr uint32_t kRecvUnits = 2;  // c_18 holds two units of sums
    const uint32_t recv_base = get_write_ptr(cb_recv);

    const uint32_t in0_base = get_write_ptr(cb_in0);
    const uint32_t source_addr = get_semaphore(source_sem);
    volatile tt_l1_ptr uint32_t* source_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source_addr);

    // Output: subblocks in compute order (per M chunk, per row of subblocks, per subblock).
    const uint32_t sub = sh * sw;
    const uint32_t subs_total = active && krole != 1 ? mchunks * (mc / sh) * (nc / sw) : 0;
    uint32_t subs_done = 0, chunk = 0, row = 0, col = 0;
    auto drain = [&]() {
        if (krole == 1) {
            // A unit goes into ring slot units_sent % kRecvUnits of the partner's c_18 once the partner has granted
            // it (credits counted in this core's partial semaphore).
            while (units_sent < partial_units && *partial_ptr > units_sent &&
                   cb_pages_available_at_front(cb_send, unit_tiles)) {
                noc_async_write(
                    get_read_ptr(cb_send),
                    get_noc_addr(partner_x, partner_y, recv_base + (units_sent % kRecvUnits) * unit_bytes),
                    unit_bytes);
                noc_async_write_barrier();  // the sums have landed before they are counted
                noc_semaphore_inc(partner_sem, 1);
                cb_pop_front(cb_send, unit_tiles);
                units_sent++;
            }
        } else if (krole == 2) {
            // Grants the partner a slot once compute has freed it, then pushes each unit that has landed.
            while (units_granted < partial_units && units_granted < units_got + kRecvUnits &&
                   cb_pages_reservable_at_back(cb_recv, (units_granted - units_got + 1) * unit_tiles)) {
                noc_semaphore_inc(partner_sem, 1);
                units_granted++;
            }
            while (units_got < units_granted && *partial_ptr > units_got) {
                cb_reserve_back(cb_recv, unit_tiles);
                cb_push_back(cb_recv, unit_tiles);
                units_got++;
            }
        }
        while (subs_done < subs_total && cb_pages_available_at_front(cb_out, sub)) {
            uint32_t l1 = get_read_ptr(cb_out);
            for (uint32_t h = 0; h < sh; h++) {
                const uint32_t m = m_base + chunk * mc + row * sh + h;
                for (uint32_t w = 0; w < sw; w++) {
                    noc_async_write(l1, output.get_noc_addr(m * n_tiles + n0 + col * sw + w), tile_bytes);
                    l1 += tile_bytes;
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, sub);
            subs_done++;
            if (++col == nc / sw) {
                col = 0;
                if (++row == mc / sh) {
                    row = 0;
                    chunk++;
                }
            }
        }
    };

    // Sender: its blocks j into its own slots, j < b + slots (block j - slots, in the same slot, has arrived) and,
    // on an active core, the slot free (relative to block b's slot, reserved).
    uint32_t read_next = role ? role - 1 : blocks;
    auto read_ahead = [&](uint32_t b) {
        while (read_next < blocks && read_next < b + slots &&
               (!active || cb_pages_reservable_at_back(cb_in0, (read_next - b + 1) * block_tiles))) {
            const uint32_t j = read_next;
            read_next += senders;
            const uint32_t m_first = m_base + (j / (kblocks * units)) * mc + (j % units) * ur;
            const uint32_t k_first = k_base + (j / units) % kblocks * kb;
            uint32_t l1 = in0_base + (j % slots) * block_bytes;
            for (uint32_t h = 0; h < ur; h++) {
                for (uint32_t w = 0; w < kb; w++) {
                    noc_async_read(input.get_noc_addr((m_first + h) * k_tiles + k_first + w), l1, tile_bytes);
                    l1 += tile_bytes;
                }
            }
        }
    };

    // Counts this core ready for block j as soon as its slot is free: block j - slots has arrived (j < b + slots, b
    // the next block to receive) and, on an active core, been consumed.
    uint32_t ready_next = 0;
    auto signal_ahead = [&](uint32_t b) {
        while (ready_next < blocks && ready_next < b + slots &&
               (!active || cb_pages_reservable_at_back(cb_in0, (ready_next - b + 1) * block_tiles))) {
            noc_semaphore_inc(ready_noc(ready_next), 1);
            ready_next++;
        }
    };

    for (uint32_t b = 0; b < blocks; b++) {
        const uint32_t slot = b % slots;
        if (active) {
            while (!cb_pages_reservable_at_back(cb_in0, block_tiles)) {
                drain();
            }
            cb_reserve_back(cb_in0, block_tiles);
        }
        const uint32_t s = b % senders, n = b / senders + 1;  // block b is the n-th of sender s
        signal_ahead(b);
        read_ahead(b);
        if (role == s + 1) {
            volatile tt_l1_ptr uint32_t* ready_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sems[slot]));
            while (*ready_ptr < band_cores) {
                drain();
                signal_ahead(b);
                read_ahead(b);
            }
            *ready_ptr = 0;  // the next counts for this slot come after this block has been consumed
            noc_async_read_barrier();
            *source_ptr = n;
            if (band_cores > 1) {
                noc_async_write_multicast(
                    in0_base + slot * block_bytes,
                    get_noc_multicast_addr(x0, y0, x1, y1, in0_base + slot * block_bytes),
                    block_bytes,
                    band_cores - 1,
                    true);
                // The count through the same command buffer as the data, so that it lands after the data.
                noc_async_write_multicast(
                    source_addr, get_noc_multicast_addr(x0, y0, x1, y1, count_addr[s]), 4, band_cores - 1, false);
                noc_async_writes_flushed();  // the slot may be read into again
            }
            *count_ptr[s] = n;
        } else {
            while (*count_ptr[s] < n) {
                drain();
                signal_ahead(b);
                read_ahead(b);
            }
        }
        if (active) {
            cb_push_back(cb_in0, block_tiles);
        }
    }
    while (subs_done < subs_total || units_sent < (krole == 1 ? partial_units : 0) ||
           units_got < (krole == 2 ? partial_units : 0)) {
        drain();
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
