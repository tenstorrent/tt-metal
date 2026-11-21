// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

#include <cstdint>

#include "all_gather_common.hpp"

using address_t = uint32_t;
namespace ag = ttnn::operations::ccl::all_gather_walk;

// Multicast reader: CB producer, no fabric. Reads this worker's slice of our input once, packet by packet, writes it
// into our own output and hands the slots to the writer, which multicasts them. Then waits for every other device's
// done credit (see the protocol note in all_gather_common.hpp).
void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr ag::WalkGeo geo = ag::walk_geo_from_ct_args<0>();
    constexpr uint32_t packet_cap = ag::packet_cap_from_ct_args<0>();
    constexpr uint32_t packets_per_slot = get_compile_time_arg_val(ag::kWalkCtArgCount + 0);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(ag::kWalkCtArgCount + 1);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(ag::kWalkCtArgCount + 2);
    constexpr auto input_tensor_args = TensorAccessorArgs<ag::kWalkCtArgCount + 3>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t unit_bytes = geo.unit_bytes;
    // The multicast writer keeps no fabric headers in the slot.
    constexpr ag::SlotLayout lay = ag::checked_slot_layout<packets_per_slot, unit_bytes, 0, packet_cap, slot_bytes>();

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
    const uint32_t num_done_credits = get_arg_val<uint32_t>(arg_idx++);  // one per route from every other device

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
    ag::fill_slots(noc, cb, lay, unit_bytes, builder, /*local_copy=*/true, [](uint32_t) {});

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Every other device's done credit has landed (each follows that device's data); take only this launch's.
    ag::wait_and_consume(done_sem, num_done_credits);
    noc.async_atomic_barrier();
    // The local copy has landed in our output before the op completes.
    noc.async_write_barrier();
}
