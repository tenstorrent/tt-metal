// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/dataflow/packed_summary.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/persistent_counter.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/dataflow/relay_chronology.hpp"

// The chronology (relay_chronology.hpp), from common arg 0.
using Chronology = kda_relay::Chronology<0, 0>;
constexpr uint32_t relay_ct = Chronology::next_ct;
constexpr uint32_t kt = get_compile_time_arg_val(relay_ct);  // key tiles; the state is kt x kt tiles
constexpr uint32_t cb_e = get_compile_time_arg_val(relay_ct + 1);
constexpr uint32_t cb_b = get_compile_time_arg_val(relay_ct + 2);
constexpr uint32_t cb_ein = get_compile_time_arg_val(relay_ct + 3);
// cb_ein's storage again, as compute's exact (UnpackToDest) view of the entry states.
constexpr uint32_t cb_ein_exact = get_compile_time_arg_val(relay_ct + 4);
// One BF16 32x32 identity tile, for compute's exact I @ B.
constexpr uint32_t cb_identity = get_compile_time_arg_val(relay_ct + 5);
// The final carry: the first rank streams each head's relayed final state into compute's second pass, with the final
// transition's E and B when the interval ends in a separated tail.
constexpr uint32_t cb_fe = get_compile_time_arg_val(relay_ct + 6);
constexpr uint32_t cb_fb = get_compile_time_arg_val(relay_ct + 7);
constexpr uint32_t cb_p = get_compile_time_arg_val(relay_ct + 8);
constexpr uint32_t cb_p_exact = get_compile_time_arg_val(relay_ct + 9);
// The correction cores' seed buffers (at the same address here), and their count of the seed columns landed there.
constexpr uint32_t cb_seed = get_compile_time_arg_val(relay_ct + 10);
constexpr uint32_t cb_tail = get_compile_time_arg_val(relay_ct + 11);
constexpr uint32_t landed_semaphore_id = get_compile_time_arg_val(relay_ct + 12);
// Per head slot, the senders' releases of the head: 1 once the relay's last column has left, then one per carry
// sender once the carry's has.
constexpr uint32_t first_ack_semaphore_id = get_compile_time_arg_val(relay_ct + 13);
constexpr uint32_t relay_heads = get_compile_time_arg_val(relay_ct + 14);  // the landing buffer's slots per state
constexpr uint32_t cores_per_head = get_compile_time_arg_val(relay_ct + 15);
constexpr uint32_t core_chunks = get_compile_time_arg_val(relay_ct + 16);
constexpr auto packed_args = TensorAccessorArgs<relay_ct + 17>();
constexpr auto init_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
constexpr auto fe_args = TensorAccessorArgs<init_args.next_compile_time_args_offset()>();
constexpr auto fb_args = TensorAccessorArgs<fe_args.next_compile_time_args_offset()>();
constexpr uint32_t state_tiles = kt * kt;
// The slab is SP-sharded: this rank's entry holds heads [sp_rank * stored_heads, +stored_heads).
constexpr uint32_t stored_heads = Chronology::entry_tiles / state_tiles;
// The common args after the chronology's.
constexpr uint32_t packed_arg = Chronology::next_common;
constexpr uint32_t init_arg = packed_arg + 1;
constexpr uint32_t fe_arg = packed_arg + 2;
constexpr uint32_t fb_arg = packed_arg + 3;
constexpr uint32_t landing_arg = packed_arg + 4;
constexpr uint32_t arrival_arg = packed_arg + 5;
constexpr uint32_t final_arg = packed_arg + 6;

// A relay core serves its heads (per-core args: their count, then per head its index and its correction cores' NoC
// x | y << 16), one after the other in its link's order. It reads each head's complement-form [E | B] rows (E = A - I)
// and entry state column by column (the initial state on the chronologically first rank, else the predecessor's landed
// columns, in the head's landing slot), then copies the landed entry state to the head's correction cores as their
// seed; on the first rank, the relayed final states P as well, which the final carry and a separated tail start from.
// With the SP-sharded slab, the first rank reads the heads it holds from its entry, and the rank holding any other head
// sends that head's entry state into its landing slot here (sender_relay_chunk_scan.cpp).
void kernel_main() {
    const uint32_t heads = get_arg_val<uint32_t>(0);
    const auto head_of = [](uint32_t s) { return get_arg_val<uint32_t>(1 + s * (1 + cores_per_head)); };
    const auto core_xy = [](uint32_t s, uint32_t d) { return get_arg_val<uint32_t>(2 + s * (1 + cores_per_head) + d); };
    const uint32_t landing = get_common_arg_val<uint32_t>(landing_arg);
    const uint32_t arrival_semaphore = get_common_arg_val<uint32_t>(arrival_arg);
    const uint32_t final_semaphore = get_common_arg_val<uint32_t>(final_arg);

    const Chronology chronology;
    const bool holder = chronology.holder();
    // The writer's chronology: this rank's position (the slab index stays in its word); compute's: whether this rank
    // forms the final carry, whether it applies the final transition, and the heads.
    auto* chron_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(Chronology::cb_chron));
    auto* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(Chronology::cb_ctl));
    chron_words[0] = chronology.position;
    cb_push_back(Chronology::cb_chron, 1);
    ctl[0] = holder;
    ctl[1] = chronology.interval.split;
    ctl[2] = heads;
    cb_push_back(Chronology::cb_ctl, 1);

    // [E | B] rows of a head: kt tile rows of 2 kt tiles.
    const auto packed = TensorAccessor(packed_args, get_common_arg_val<uint32_t>(packed_arg));
    cb_reserve_back(cb_e, heads * state_tiles);
    cb_reserve_back(cb_b, heads * state_tiles);
    const uint32_t eb_tile = get_tile_size(cb_e);
    const uint32_t e_base = get_write_ptr(cb_e);
    const uint32_t b_base = get_write_ptr(cb_b);
    const auto read_summary = [&](uint32_t s) {
        for (uint32_t r = 0; r < kt; ++r) {
            for (uint32_t c = 0; c < kt; ++c) {
                const uint32_t page = (head_of(s) * kt + r) * 2 * kt + c;
                const uint32_t offset = (s * state_tiles + r * kt + c) * eb_tile;
                noc_async_read_page(page, packed, e_base + offset);
                noc_async_read_page(page + kt, packed, b_base + offset);
            }
        }
    };
    // The BF16 identity tile.
    DataflowBuffer identity_tile(cb_identity);
    identity_tile.reserve_back(1);
    dataflow_kernel_lib::zero_tile(identity_tile);
    kda::write_identity_diagonal(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_identity)));
    cb_push_back(cb_identity, 1);

    const uint32_t state_tile = get_tile_size(cb_ein);
    const uint32_t column_bytes = kt * state_tile;
    const uint32_t state_bytes = kt * column_bytes;
    // The entry states are stored column-major by tile, so each tile column (all the next state's column needs) can
    // land, be computed and leave on its own.
    cb_reserve_back(cb_ein, heads * state_tiles);
    cb_reserve_back(cb_ein_exact, heads * state_tiles);
    if (holder) {
        // The relay starts from the first rank's entry states, head by head: the first head's hop starts as soon as
        // its own reads have landed, or a remote head's as each of its columns lands.
        const auto init = TensorAccessor(init_args, get_common_arg_val<uint32_t>(init_arg));
        // A head another rank holds: that rank's sender writes each tile column of the head's entry state into its
        // landing slot, in order, then counts it in the arrival counter (no predecessor counts there on this rank).
        const auto owner = [&](uint32_t s) { return head_of(s) / stored_heads; };
        const auto remote = [&](uint32_t s) {
            return Chronology::slab && !chronology.zero_entry && owner(s) != chronology.sp_rank;
        };
        // A relay core's heads lie about H/2 apart (relay_chunk_scan_relay_slot), so no two have one owner: a single
        // remote head's columns arrive in order, but two remote heads' columns may interleave in the one counter, so
        // then every remote column is waited for first.
        uint32_t remote_heads = 0;
        for (uint32_t s = 0; s < heads; ++s) {
            remote_heads += remote(s) ? 1 : 0;
        }
        const uint32_t remote_columns = remote_heads * kt;
        const bool interleaved = remote_heads > 1;
        uint32_t arrived = 0;
        for (uint32_t s = 0; s < heads; ++s) {
            read_summary(s);
            if (chronology.zero_entry) {
                const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
                for (uint32_t offset = 0; offset < state_bytes; offset += MEM_ZEROS_SIZE) {
                    noc_async_read(zeros, landing + s * state_bytes + offset, MEM_ZEROS_SIZE);
                }
            } else if (!remote(s)) {
                const uint32_t stored = Chronology::slab ? head_of(s) % stored_heads : head_of(s);
                for (uint32_t j = 0; j < kt; ++j) {
                    for (uint32_t r = 0; r < kt; ++r) {
                        noc_async_read_page(
                            chronology.entry_base() + stored * state_tiles + r * kt + j,
                            init,
                            landing + s * state_bytes + j * column_bytes + r * state_tile);
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_e, state_tiles);
            cb_push_back(cb_b, state_tiles);
            if (remote(s)) {
                for (uint32_t j = 0; j < kt; ++j) {
                    kda_counter::wait(arrival_semaphore, interleaved ? remote_columns : ++arrived);
                    cb_push_back(cb_ein, kt);
                    cb_push_back(cb_ein_exact, kt);
                }
            } else {
                cb_push_back(cb_ein, state_tiles);
                cb_push_back(cb_ein_exact, state_tiles);
            }
        }
        if (remote_columns != 0) {
            kda_counter::consume(arrival_semaphore, remote_columns);
        }
    } else {
        for (uint32_t s = 0; s < heads; ++s) {
            read_summary(s);
        }
        noc_async_read_barrier();
        cb_push_back(cb_e, heads * state_tiles);
        cb_push_back(cb_b, heads * state_tiles);
        // The entry states arrive after a share of the lap: the predecessor's sender writes each tile column of each
        // head's entry state into the head's landing slot, in order, then counts it.
        for (uint32_t u = 0; u < heads * kt; ++u) {
            kda_counter::wait(arrival_semaphore, u + 1);
            cb_push_back(cb_ein, kt);
            cb_push_back(cb_ein_exact, kt);
        }
        kda_counter::consume(arrival_semaphore, heads * kt);
    }

    // Each landed tile column of a K x K state (column-major by tile) goes into the seed buffer of a head's correction
    // cores [d_begin, d_end) (at the same address on each), counted there; compute starts the chunk outputs once the
    // whole seed has landed. A head's correction cores are a column of the grid (correction_core.hpp): one multicast
    // reaches them.
    const uint32_t landed = get_semaphore(landed_semaphore_id);
    const auto distribute = [&](uint32_t s, uint32_t source, uint32_t seed, uint32_t d_begin, uint32_t d_end) {
        // NoC 0 multicasts from the column's top, NoC 1 from its bottom.
        const uint32_t start = core_xy(s, noc_index == 0 ? d_begin : d_end - 1);
        const uint32_t end = core_xy(s, noc_index == 0 ? d_end - 1 : d_begin);
        const uint32_t cores = d_end - d_begin;
        const auto address = [&](uint32_t local) {
            return cores == 1 ? get_noc_addr(start & 0xFFFF, start >> 16, local)
                              : get_noc_multicast_addr(start & 0xFFFF, start >> 16, end & 0xFFFF, end >> 16, local);
        };
        for (uint32_t j = 0; j < kt; ++j) {
            if (cores == 1) {
                noc_async_write(source + j * column_bytes, address(seed + j * column_bytes), column_bytes);
            } else {
                noc_async_write_multicast(
                    source + j * column_bytes, address(seed + j * column_bytes), column_bytes, cores);
            }
            noc_async_writes_flushed();
            if (cores == 1) {
                noc_semaphore_inc(address(landed), 1);
            } else {
                noc_semaphore_inc_multicast(address(landed), 1, cores);
            }
        }
        noc_async_write_barrier();
    };
    const auto released = [](uint32_t s) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(first_ack_semaphore_id + s));
    };
    // A head's seeds leave only once its sender has read the head's last column and released it: the copies would
    // otherwise share the NoC with the sender's reads of the next state, and hold up the relay at every hop.
    const uint32_t valid = chronology.interval.valid_chunks(0, 1);
    if (valid != 0) {
        for (uint32_t s = 0; s < heads; ++s) {
            noc_semaphore_wait_min(released(s), 1);
            distribute(s, landing + s * state_bytes, get_write_ptr(cb_seed), 0, cores_per_head);
        }
    }

    if (holder) {
        // The last rank's sender writes each tile column of each head's final state into the head's final slot, then
        // counts it. With a separated tail, the heads' final transitions go in first: P lands only after the whole lap.
        // They take the entry states' landing slots and cb_b's storage, free once every head's hop has left.
        if (chronology.interval.split) {
            const auto fe = TensorAccessor(fe_args, get_common_arg_val<uint32_t>(fe_arg));
            const auto fb = TensorAccessor(fb_args, get_common_arg_val<uint32_t>(fb_arg));
            for (uint32_t s = 0; s < heads; ++s) {
                noc_semaphore_wait_min(released(s), 1);
            }
            cb_reserve_back(cb_fe, heads * state_tiles);
            cb_reserve_back(cb_fb, heads * state_tiles);
            for (uint32_t i = 0; i < heads * state_tiles; ++i) {
                const uint32_t page = head_of(i / state_tiles) * state_tiles + i % state_tiles;
                noc_async_read_page(page, fe, get_write_ptr(cb_fe) + i * state_tile);
                noc_async_read_page(page, fb, get_write_ptr(cb_fb) + i * state_tile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_fe, heads * state_tiles);
            cb_push_back(cb_fb, heads * state_tiles);
        }
        // Each head's P also seeds the tail's chunks [reset, valid), on the cores that hold them, as soon as it lands.
        const uint32_t reset = chronology.interval.reset_chunk(0, 1);
        cb_reserve_back(cb_p, heads * state_tiles);
        cb_reserve_back(cb_p_exact, heads * state_tiles);
        for (uint32_t s = 0; s < heads; ++s) {
            for (uint32_t j = 0; j < kt; ++j) {
                kda_counter::wait(final_semaphore, s * kt + j + 1);
                cb_push_back(cb_p, kt);
                cb_push_back(cb_p_exact, kt);
            }
            if (reset != 0) {
                distribute(
                    s,
                    landing + (relay_heads + s) * state_bytes,
                    get_write_ptr(cb_tail),
                    reset / core_chunks,
                    (valid + core_chunks - 1) / core_chunks);
            }
        }
        kda_counter::consume(final_semaphore, heads * kt);
    }
    noc_async_atomic_barrier();
}
