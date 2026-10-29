// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

#include <cstdint>

#include "all_gather_common.hpp"

using address_t = uint32_t;

// Multicast reader: CB producer, no fabric. Reads this worker's slice of our input once, packet by packet, writes it
// into our own output and hands the slots to the writer, which multicasts them. Then waits for every other device's
// done credit (see the protocol note in all_gather_common.hpp).
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
    constexpr uint32_t cb0_id = get_compile_time_arg_val(11);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(12);
    constexpr auto input_tensor_args = TensorAccessorArgs<13>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    static_assert(
        packet_cap % unit_bytes == 0 && packet_cap >= unit_bytes && packet_cap <= max_packet_bytes,
        "the packet cap is a whole number of units within the payload");
    // The multicast writer keeps no fabric headers in the slot.
    constexpr ag::SlotLayout lay = ag::slot_offsets(packets_per_slot, unit_bytes, 0, packet_cap);
    static_assert(lay.slot_bytes == cb_page_size, "host and kernel disagree on the slot layout");
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
    const uint32_t slice_start = get_arg_val<uint32_t>(arg_idx++);  // walk elements [start, start + count)
    const uint32_t slice_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t device_idx = get_arg_val<uint32_t>(arg_idx++);
    const address_t done_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t done_wait_value = get_arg_val<uint32_t>(arg_idx++);  // done credits to wait for

    const auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    const auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);

    ag::Walk walk(output_tensor_accessor, geo);
    ag::InputPieces<decltype(input_tensor_accessor)> input_pieces(input_tensor_accessor, geo);
    ag::PacketBuilder builder(walk, input_pieces, packet_cap);

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    walk.init(device_idx, slice_start, slice_count, 0, slice_count);
    {
        AG_TEST_ZONE("AG-RD-FILL");  // TEST ONLY
        ag::fill_slots(
            noc, cb, lay, unit_bytes, builder, slice_count * element_units, /*local_copy=*/true, [](uint32_t) {});
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Every other device's done credit has landed (each follows that device's data); take only this launch's.
    {
        AG_TEST_ZONE("AG-RD-DONEWAIT");  // TEST ONLY
        ag::wait_and_consume(done_sem, done_wait_value);
    }
    AG_TEST_ZONE("AG-RD-BARRIERS");  // TEST ONLY
    noc.async_atomic_barrier();
    // The local copy has landed in our output before the op completes.
    noc.async_write_barrier();
}
