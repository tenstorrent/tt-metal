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

// Store-and-forward reader: CB producer, no fabric. It starts at once: the init handshake gates only the writer's
// remote sends, so the first slots and the local copy overlap it. Iteration 0 reads this device's stripe from the
// input; iteration i >= 1 re-reads the stripe upstream relayed into our output in its iteration i - 1, gated on
// data_valid. Each slot carries up to packets_per_slot packets behind one read barrier (see ag::fill_slots and the
// protocol note in all_gather_common.hpp).
void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr ag::WalkGeo geo = ag::walk_geo_from_ct_args<0>();
    constexpr uint32_t packet_cap = ag::packet_cap_from_ct_args<0>();
    constexpr uint32_t packets_per_slot = get_compile_time_arg_val(ag::kWalkCtArgCount + 0);
    constexpr uint32_t packet_header_size = get_compile_time_arg_val(ag::kWalkCtArgCount + 1);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(ag::kWalkCtArgCount + 2);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(ag::kWalkCtArgCount + 3);
    constexpr auto input_tensor_args = TensorAccessorArgs<ag::kWalkCtArgCount + 4>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t unit_bytes = geo.unit_bytes;
    constexpr uint32_t element_units = geo.element_units;
    constexpr uint32_t num_devices = geo.num_devices;
    constexpr ag::SlotLayout lay =
        ag::checked_slot_layout<packets_per_slot, unit_bytes, packet_header_size, packet_cap, slot_bytes>();
    // The most units one slot carries.
    constexpr uint32_t slot_units = packets_per_slot * (packet_cap / unit_bytes);

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
    const bool local_copy = get_arg_val<uint32_t>(arg_idx++) != 0;
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
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const bool last = iter + 1 == num_iters;
        const uint32_t skip = last ? final_skip : 0;
        const uint32_t take = last ? final_count : slice_count;
        const uint32_t take_units = take * element_units;
        walk.init((initial_stripe + iter * stripe_step) % num_devices, slice_start, slice_count, skip, take);
        if (iter == 0) {
            // Local data (our own input tensor)
            ag::fill_slots(noc, cb, lay, unit_bytes, input_builder, take_units, local_copy, [](uint32_t) {});
        } else {
            // Relay: upstream sent its iteration iter - 1 in walk order and signals cumulative units, so the first
            // `skip` elements of that pass precede ours. Wait before reserving: a slot holds at most slot_units, all
            // of this iteration, so the target covers every read of the slot and is reached without holding one (a
            // ring would deadlock otherwise).
            const uint32_t relay_base = (iter - 1) * iter_units + skip * element_units;
            ag::fill_slots(noc, cb, lay, unit_bytes, relay_builder, take_units, false, [&](uint32_t consumed) {
                noc_semaphore_wait_min(data_valid_ptr, relay_base + std::min(consumed + slot_units, take_units));
            });
        }
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Completion: wait for every unit upstream sends us, then subtract only this launch's; a reset could drop the
    // next's.
    ag::wait_and_consume(data_valid_sem, total_units);
    noc.async_atomic_barrier();
    // The local copy has landed in our output before the op completes.
    noc.async_write_barrier();
}
