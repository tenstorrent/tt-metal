// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/persistent_counter.hpp"

constexpr uint32_t cb_eout = get_compile_time_arg_val(0);
constexpr uint32_t cb_chron = get_compile_time_arg_val(1);
constexpr uint32_t kt = get_compile_time_arg_val(2);
// Per head slot, the senders' releases of the head.
constexpr uint32_t first_ack_semaphore_id = get_compile_time_arg_val(3);
constexpr uint32_t relay_heads = get_compile_time_arg_val(4);  // the landing buffer's slots per state
constexpr auto carry_args = TensorAccessorArgs<5>();
// Slab (slab_kernel_args): the carry output is the slab, written at entry `index * slab_scale + slab_offset`; nothing
// past slab_index_limit. The reader leaves the index in cb_chron's word slab_index_word.
constexpr uint32_t slab_ct = carry_args.next_compile_time_args_offset();
constexpr bool slab = get_compile_time_arg_val(slab_ct);
constexpr uint32_t slab_index_limit = get_compile_time_arg_val(slab_ct + 1);
constexpr uint32_t slab_scale = get_compile_time_arg_val(slab_ct + 2);
constexpr uint32_t slab_offset = get_compile_time_arg_val(slab_ct + 3);
constexpr uint32_t entry_tiles = get_compile_time_arg_val(slab_ct + 4);
constexpr uint32_t slab_index_word = get_compile_time_arg_val(slab_ct + 5);
constexpr uint32_t state_tiles = kt * kt;
// The slab is SP-sharded: this rank's entry holds heads [sp_rank * stored_heads, +stored_heads).
constexpr uint32_t stored_heads = entry_tiles / state_tiles;

// A relay core hands each tile column of each of its heads' next entry states to the head's sender as compute packs
// it (per-core args: the heads, then per head its index, its count semaphore on the senders, and its link's relay and
// second carry senders' NoC x | y << 16). On the first rank it hands on and writes each head's final carry; on every
// other rank it writes each head's carry once the first rank's senders have landed it in the head's final slot. With
// the SP-sharded slab only the heads this rank holds are written, at their place in its entry, and only those land.
void kernel_main() {
    const uint32_t heads = get_arg_val<uint32_t>(0);
    const auto head_arg = [](uint32_t s, uint32_t i) { return get_arg_val<uint32_t>(1 + 5 * s + i); };
    // The landing buffer's final slots, and the persistent count of the columns landed there.
    const uint32_t final_landing = get_common_arg_val<uint32_t>(1) + relay_heads * state_tiles * get_tile_size(cb_eout);
    const uint32_t final_semaphore = get_common_arg_val<uint32_t>(2);
    const uint32_t sp_rank = get_common_arg_val<uint32_t>(3);
    // Whether this rank holds head slot s's carry, and the head's place in this rank's carry output.
    const auto owns = [&](uint32_t s) { return !slab || head_arg(s, 0) / stored_heads == sp_rank; };
    const auto stored = [&](uint32_t s) { return slab ? head_arg(s, 0) % stored_heads : head_arg(s, 0); };

    // The reader's chronology: this rank's position, and the slab index.
    cb_wait_front(cb_chron, 1);
    const auto* chron = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_chron));
    const uint32_t position = chron[0];
    const uint32_t index = slab ? chron[slab_index_word] : 0;
    const bool writes_carry = !slab || index < slab_index_limit;
    const uint32_t carry_base = writes_carry ? (index * slab_scale + slab_offset) * entry_tiles : 0;
    cb_pop_front(cb_chron, 1);

    // Hand each tile column of the heads' states compute forms in cb_eout, head after head, to the heads' sender(s) as
    // it is packed (a count on the sender's core): the relay's sender, or for the final carry the head's carrying
    // senders.
    const auto senders_of = [&](uint32_t s, bool carry) { return carry ? head_arg(s, 4) : 1u; };
    const auto hand_on = [&](bool carry) {
        for (uint32_t s = 0; s < heads; ++s) {
            const uint32_t senders = senders_of(s, carry);
            const uint32_t notify_semaphore = get_semaphore(head_arg(s, 1));
            for (uint32_t j = 0; j < kt; ++j) {
                cb_wait_front(cb_eout, s * state_tiles + (j + 1) * kt);
                for (uint32_t sender = 0; sender < 2; ++sender) {
                    if (senders >> sender & 1) {
                        const uint32_t xy = head_arg(s, 2 + sender);
                        noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, notify_semaphore), 1);
                    }
                }
            }
        }
    };
    // Each head's slot is free again once its senders have read its last column: the relay's, then for the final
    // carry each carrying sender's.
    const auto release = [&](bool carry) {
        for (uint32_t s = 0; s < heads; ++s) {
            const uint32_t senders = senders_of(s, true);
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(first_ack_semaphore_id + s)),
                1 + (carry ? (senders & 1) + (senders >> 1) : 0));
        }
        cb_wait_front(cb_eout, relay_heads * state_tiles);
    };
    // Each head's carry (stored column-major by tile at `source`, slot after slot) into this rank's row-major carry
    // output.
    const auto carry = TensorAccessor(carry_args, get_common_arg_val<uint32_t>(0));
    const uint32_t tile = get_tile_size(cb_eout);
    const auto write_carry = [&](uint32_t source) {
        for (uint32_t s = 0; s < heads && writes_carry; ++s) {
            if (!owns(s)) {
                continue;
            }
            for (uint32_t i = 0; i < state_tiles; ++i) {
                noc_async_write_page(
                    carry_base + stored(s) * state_tiles + (i % kt) * kt + i / kt,
                    carry,
                    source + (s * state_tiles + i) * tile);
            }
        }
        noc_async_write_barrier();
    };
    hand_on(false);
    release(false);
    cb_pop_front(cb_eout, relay_heads * state_tiles);

    if (position == 0) {
        // The final carries: hand their columns to the senders, then, once they have read them (the relay's release,
        // then each carrying sender's), write them out, off the DRAM the output gate still reads meanwhile.
        hand_on(true);
        release(true);
        write_carry(get_read_ptr(cb_eout));
        cb_pop_front(cb_eout, relay_heads * state_tiles);
    } else {
        // Each carry column out as it lands, in slot order (one sender carries a relay core's heads to a rank); the
        // call ends once the whole carry has arrived, which separates consecutive calls across the SP line.
        uint32_t landed = 0;
        for (uint32_t s = 0; s < heads; ++s) {
            if (!owns(s)) {
                continue;
            }
            for (uint32_t j = 0; j < kt; ++j) {
                kda_counter::wait(final_semaphore, ++landed);
                for (uint32_t r = 0; r < kt && writes_carry; ++r) {
                    noc_async_write_page(
                        carry_base + stored(s) * state_tiles + r * kt + j,
                        carry,
                        final_landing + (s * state_tiles + j * kt + r) * tile);
                }
            }
        }
        noc_async_write_barrier();
        kda_counter::consume(final_semaphore, landed);
    }
    noc_async_atomic_barrier();
}
