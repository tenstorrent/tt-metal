// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

#include <cstdint>

#include "all_gather_common.hpp"

using address_t = uint32_t;

// Store-and-forward reader: CB producer, no fabric. It starts at once: the init handshake gates only the writer's
// remote sends, so the first slots and the local copy overlap it. Iteration 0 reads this device's stripe from the
// input; iteration i >= 1 re-reads the stripe upstream relayed into our output in its iteration i - 1, gated on
// data_valid. Each slot carries up to packets_per_slot packets behind one read barrier (see ag::fill_slots and the
// protocol note in all_gather_common.hpp).
void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t unit_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t in_page_units = get_compile_time_arg_val(1);
    constexpr uint32_t out_page_units = get_compile_time_arg_val(2);
    constexpr uint32_t stripe_units = get_compile_time_arg_val(3);
    constexpr uint32_t num_devices = get_compile_time_arg_val(4);
    constexpr uint32_t element_units = get_compile_time_arg_val(5);
    constexpr uint32_t max_packet_bytes = get_compile_time_arg_val(6);
    constexpr bool dst_cross = get_compile_time_arg_val(7) != 0;
    constexpr bool src_cross = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t packet_cap = get_compile_time_arg_val(9);
    constexpr uint32_t packets_per_slot = get_compile_time_arg_val(10);
    constexpr uint32_t packet_header_size = get_compile_time_arg_val(11);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(12);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(13);
    constexpr auto input_tensor_args = TensorAccessorArgs<14>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    static_assert(
        packet_cap % unit_bytes == 0 && packet_cap >= unit_bytes && packet_cap <= max_packet_bytes,
        "the packet cap is a whole number of units within the payload");
    constexpr ag::SlotLayout lay = ag::slot_offsets(packets_per_slot, unit_bytes, packet_header_size, packet_cap);
    static_assert(lay.slot_bytes == cb_page_size, "host and kernel disagree on the slot layout");
    // The most units one slot carries.
    constexpr uint32_t slot_units = packets_per_slot * (packet_cap / unit_bytes);
    constexpr ag::WalkGeo geo{
        .unit_bytes = unit_bytes,
        .in_page_units = in_page_units,
        .out_page_units = out_page_units,
        .stripe_units = stripe_units,
        .num_devices = num_devices,
        .element_units = element_units,
        .max_packet_bytes = max_packet_bytes,
        .dst_cross = dst_cross,
        .src_cross = src_cross};

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t initial_stripe = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stripe_step = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_iters = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t total_units = get_arg_val<uint32_t>(arg_idx++);  // units upstream delivers (completion wait)
    const uint32_t slice_start = get_arg_val<uint32_t>(arg_idx++);  // walk elements [start, start + count)
    const uint32_t slice_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_skip = get_arg_val<uint32_t>(arg_idx++);  // last iteration: walk positions skipped, taken
    const uint32_t final_count = get_arg_val<uint32_t>(arg_idx++);
    const bool do_local_write = get_arg_val<uint32_t>(arg_idx++) != 0;
    const address_t data_valid_sem = get_arg_val<uint32_t>(arg_idx++);

    const auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    const auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);
    auto* data_valid_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_valid_sem);

    ag::Walk walk(output_tensor_accessor, geo);
    ag::InputPieces<decltype(input_tensor_accessor)> input_pieces(input_tensor_accessor, geo);
    ag::RelayPieces relay_pieces(geo);
    ag::PacketBuilder input_builder(walk, input_pieces, packet_cap);
    ag::PacketBuilder relay_builder(walk, relay_pieces, packet_cap);

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    const uint32_t iter_units = slice_count * element_units;
    {
        AG_TEST_ZONE("AG-RD-FILL");  // TEST ONLY
        for (uint32_t iter = 0; iter < num_iters; ++iter) {
            const bool last = iter + 1 == num_iters;
            const uint32_t skip = last ? final_skip : 0;
            const uint32_t take = last ? final_count : slice_count;
            const uint32_t take_units = take * element_units;
            walk.init((initial_stripe + iter * stripe_step) % num_devices, slice_start, slice_count, skip, take);
            if (iter == 0) {
                // Local data (our own input tensor)
                ag::fill_slots(noc, cb, lay, unit_bytes, input_builder, take_units, do_local_write, [](uint32_t) {});
            } else {
                // Relay: upstream sent its iteration iter - 1 in walk order and signals cumulative units, so the first
                // `skip` elements of that pass precede ours. Wait before reserving: a slot holds at most slot_units,
                // all of this iteration, so the target covers every read of the slot and is reached without holding one
                // (a ring would deadlock otherwise).
                const uint32_t relay_base = (iter - 1) * iter_units + skip * element_units;
                ag::fill_slots(noc, cb, lay, unit_bytes, relay_builder, take_units, false, [&](uint32_t consumed) {
                    noc_semaphore_wait_min(data_valid_ptr, relay_base + std::min(consumed + slot_units, take_units));
                });
            }
        }
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Completion: wait for every unit upstream sends us, then subtract only this launch's; a reset could drop the
    // next's.
    {
        AG_TEST_ZONE("AG-RD-DONEWAIT");  // TEST ONLY
        ag::wait_and_consume(data_valid_sem, total_units);
    }
    AG_TEST_ZONE("AG-RD-BARRIERS");  // TEST ONLY
    noc.async_atomic_barrier();
    // The local copy has landed in our output before the op completes.
    noc.async_write_barrier();
}
