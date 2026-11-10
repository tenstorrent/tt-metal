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
// split: band 1 sends its fp32 sums per unit of rows (c_17) to the one or two cores of band 0 holding its N tiles, into
// their c_18, a ring of two units: each receiver grants a slot to its senders once its compute has freed it, and counts
// the parts that landed, a partial semaphore per partner either way. N passes: the blocks repeat per pass.
// Compile args: K tiles, N tiles, mc, kb, M chunks, slots (<= 4), four slot ready semaphores, two block count
// semaphores, a source semaphore, senders (1 or 2), ur, two partial semaphores, N tiles per pass, passes, then the
// input's and the output's TensorAccessorArgs. Runtime args: input address, output address, band rectangle (NoC x / y
// start, x / y end, in this writer's NoC order), first M tile of the band, first N tile, N tiles (0: no output), sh,
// sw, role (0, or 1 + the sender's index), the two senders' NoC x, y, first K tile of the band, K split role (0: none,
// 1: send sums, 2: add the partners'), the plan's N tiles per core (a unit of sums is ur x that), K blocks, band cores,
// K split partners, then six words per partner (below).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t mc = get_compile_time_arg_val(2);
    constexpr uint32_t kb = get_compile_time_arg_val(3);
    constexpr uint32_t mchunks = get_compile_time_arg_val(4);
    constexpr uint32_t slots = get_compile_time_arg_val(5);
    // Ready counts per slot (on the slot's sender of the moment; reset by it before each multicast into the slot).
    constexpr uint32_t ready_sems[4] = {
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9)};
    constexpr uint32_t count_sem0 = get_compile_time_arg_val(10), count_sem1 = get_compile_time_arg_val(11);
    constexpr uint32_t source_sem = get_compile_time_arg_val(12);
    constexpr uint32_t senders = get_compile_time_arg_val(13);
    constexpr uint32_t ur = get_compile_time_arg_val(14);
    // K split, per partner index: on band 0 the units of sums that landed from it, on band 1 the slots it granted.
    constexpr uint32_t partial_sems[2] = {get_compile_time_arg_val(15), get_compile_time_arg_val(16)};
    constexpr uint32_t np = get_compile_time_arg_val(17), passes = get_compile_time_arg_val(18);
    static_assert(slots <= 4);
    constexpr auto input_args = TensorAccessorArgs<19>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048, block_tiles = ur * kb, block_bytes = block_tiles * tile_bytes;
    constexpr uint32_t units = mc / ur;  // per M chunk per K block
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
    // Semaphores that other cores update over the NoC: read past the L1 data cache where it is on for this core (the
    // invalidation is a fence, which costs the polling loops time otherwise).
    auto l1_word = [](volatile tt_l1_ptr uint32_t* word) {
#if defined(ENABLE_L1_DATA_CACHE)
        invalidate_l1_cache();
#endif
        return *word;
    };
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
    const uint32_t k_base = get_arg_val<uint32_t>(16), krole = get_arg_val<uint32_t>(17);
    const uint32_t unit_tiles = ur * get_arg_val<uint32_t>(18), unit_bytes = unit_tiles * 4096;  // ur x the planned nc
    const uint32_t kblocks = get_arg_val<uint32_t>(19), band_cores = get_arg_val<uint32_t>(20);
    const uint32_t blocks = passes * mchunks * kblocks * units;
    // N tiles of this core in pass `pass`.
    auto pass_nc = [&](uint32_t pass) {
        return passes == 1 ? nc : nc > pass * np ? (nc - pass * np < np ? nc - pass * np : np) : 0;
    };
    // K split in two bands: band 1 (krole 1) sends its fp32 sums per unit of rows to the cores of band 0 (krole 2)
    // holding its N tiles, which add them. Per partner p: NoC x, y, this core's index among the partner's partners
    // (the partner's partial semaphore it counts in), and for a sender the tile offsets in its unit and in the
    // partner's, and the tiles.
    const uint32_t partners = get_arg_val<uint32_t>(21);
    auto partner_arg = [&](uint32_t p, uint32_t i) { return get_arg_val<uint32_t>(22 + 6 * p + i); };
    auto partner_sem = [&](uint32_t p) {
        return get_noc_addr(partner_arg(p, 0), partner_arg(p, 1), get_semaphore(partial_sems[partner_arg(p, 2) & 1]));
    };
    volatile tt_l1_ptr uint32_t* partial_ptr[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(partial_sems[0])),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(partial_sems[1]))};
    auto all_counted = [&](uint32_t n) {
        for (uint32_t p = 0; p < partners; p++) {
            if (l1_word(partial_ptr[p]) <= n) {
                return false;
            }
        }
        return true;
    };
    const uint32_t partial_units = active && krole ? mchunks * units : 0;
    uint32_t units_sent = 0, units_got = 0, units_granted = 0;
    constexpr uint32_t kRecvUnits = 2;  // c_18 holds two units of sums
    const uint32_t recv_base = get_write_ptr(cb_recv);

    const uint32_t in0_base = get_write_ptr(cb_in0);
    const uint32_t source_addr = get_semaphore(source_sem);
    volatile tt_l1_ptr uint32_t* source_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source_addr);

    // Output: subblocks in compute order (per pass, per M chunk, per row of subblocks, per subblock).
    const uint32_t sub = sh * sw;
    const uint32_t subs_total = active && krole != 1 ? mchunks * (mc / sh) * (nc / sw) : 0;
    uint32_t subs_done = 0, chunk = 0, row = 0, col = 0;  // chunk counts over all passes
    auto drain = [&]() {
        if (krole == 1) {
            // A unit goes into ring slot units_sent % kRecvUnits of each partner's c_18 once every partner has
            // granted it.
            while (units_sent < partial_units && all_counted(units_sent) &&
                   cb_pages_available_at_front(cb_send, unit_tiles)) {
                const uint32_t slot = recv_base + (units_sent % kRecvUnits) * unit_bytes;
                for (uint32_t p = 0; p < partners; p++) {
                    noc_async_write(
                        get_read_ptr(cb_send) + partner_arg(p, 3) * 4096,
                        get_noc_addr(partner_arg(p, 0), partner_arg(p, 1), slot + partner_arg(p, 4) * 4096),
                        partner_arg(p, 5) * 4096);
                }
                noc_async_write_barrier();  // the sums have landed before they are counted
                for (uint32_t p = 0; p < partners; p++) {
                    noc_semaphore_inc(partner_sem(p), 1);
                }
                cb_pop_front(cb_send, unit_tiles);
                units_sent++;
            }
        } else if (krole == 2) {
            // Grants the partners a slot once compute has freed it, then pushes each unit once all partners' parts
            // have landed.
            while (units_granted < partial_units && units_granted < units_got + kRecvUnits &&
                   cb_pages_reservable_at_back(cb_recv, (units_granted - units_got + 1) * unit_tiles)) {
                for (uint32_t p = 0; p < partners; p++) {
                    noc_semaphore_inc(partner_sem(p), 1);
                }
                units_granted++;
            }
            while (units_got < units_granted && all_counted(units_got)) {
                cb_reserve_back(cb_recv, unit_tiles);
                cb_push_back(cb_recv, unit_tiles);
                units_got++;
            }
        }
        while (subs_done < subs_total && cb_pages_available_at_front(cb_out, sub)) {
            uint32_t l1 = get_read_ptr(cb_out);
            const uint32_t pass = passes == 1 ? 0 : chunk / mchunks;
            for (uint32_t h = 0; h < sh; h++) {
                const uint32_t m = m_base + (chunk - pass * mchunks) * mc + row * sh + h;
                for (uint32_t w = 0; w < sw; w++) {
                    noc_async_write(l1, output.get_noc_addr(m * n_tiles + n0 + pass * np + col * sw + w), tile_bytes);
                    l1 += tile_bytes;
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, sub);
            subs_done++;
            if (++col == pass_nc(pass) / sw) {
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
            const uint32_t chunk_of_j = passes == 1 ? j / (kblocks * units) : j / (kblocks * units) % mchunks;
            const uint32_t m_first = m_base + chunk_of_j * mc + (j % units) * ur;
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
            while (l1_word(ready_ptr) < band_cores) {
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
            while (l1_word(count_ptr[s]) < n) {
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
