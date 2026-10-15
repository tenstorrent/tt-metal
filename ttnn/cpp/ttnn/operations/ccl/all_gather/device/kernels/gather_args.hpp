// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Compile-time (ct) and runtime (rt) argument indices of the four all_gather kernels. Included by the kernels
// (which read every fixed-position arg by these names) and by the program factories (which fill their arg
// vectors by these names, and patch them in override_runtime_arguments), so the two cannot drift.
// Macro-free and dependency-free, like gather_rules.hpp.
//
// Units: every *_chunks / chunk id / cpp / cps value is in chunks, i.e. already multiplied by k.
// `tensor_args` / `fabric_args` / `route_args` mark where a variable-length tail starts.

#include <cstdint>

namespace gather_args {

namespace unicast_reader {
namespace ct {
enum : uint32_t {
    in_cpp = 0,    // input chunks per input page (split_factor * k)
    chunk_size,    // content bytes per chunk (min page / k)
    out_cpp,       // output chunks per output page (concat ratio * k)
    cps,           // chunks per stripe row (stripe walk pages * k)
    num_devices,   // stripe indexing
    cb_id,         // the CB
    cb_page_size,  // CB entry bytes (chunks_per_entry * slot_bytes)
    k,             // chunks per walk page (power of 2)
    walk_stride,   // TiledWalk stride (walk pages)
    walk_xfer,     // TiledWalk xfer (chunks per column)
    slot_bytes,    // CB slot bytes
    in_runs,       // run_source_bits() of the input
    out_runs,      // run_source_bits() of the output (the relay reads it back)
    tensor_args,   // TensorAccessorArgs(input), then TensorAccessorArgs(output)
};
}  // namespace ct
namespace rt {
enum : uint32_t {
    input_addr = 0,
    output_addr,
    initial_stripe,  // this device's index
    stripe_step,     // stripe index step per iteration
    num_iters,       // iterations this direction runs
    total_chunks,    // chunks upstream delivers (completion wait)
    slice_first,     // slice start (chunk id)
    slice_chunks,    // slice length (chunks)
    final_skip,      // last-iteration seqno offset (even-ring split); the data_valid base
    final_take,      // last-iteration seqno count
    rot,             // lane rotation of full tiles (this slice and direction)
    final_pos_base,  // walk_seek(.., final_skip, ..): the last iteration's start
    final_pos_slot,
    final_pos_i,
    barrier_sem,     // L1 address
    data_valid_sem,  // L1 address
    count,
};
}  // namespace rt
}  // namespace unicast_reader

namespace unicast_writer {
namespace ct {
enum : uint32_t {
    chunk_size = 0,
    out_cpp,
    cps,
    num_devices,
    cb_id,
    cb_page_size,
    packet_size,
    data_valid_granularity,  // signal data_valid once per this many CB entries
    k,
    walk_stride,
    walk_xfer,
    slot_bytes,
    out_runs,
    tensor_args,  // TensorAccessorArgs(output)
};
}  // namespace ct
namespace rt {
enum : uint32_t {
    output_addr = 0,
    initial_stripe,
    stripe_step,
    num_iters,
    slice_first,
    slice_chunks,
    final_skip,
    final_take,
    rot,
    final_pos_base,
    final_pos_slot,
    final_pos_i,
    do_local_write,  // write local data into local output on iteration 0
    barrier_sem,     // L1 address
    data_valid_sem,  // L1 address
    barrier_noc_x,   // barrier_sem target: the neighbor's opposite-direction core
    barrier_noc_y,
    data_valid_noc_x,  // data_valid_sem target: the neighbor's mirror core
    data_valid_noc_y,
    num_granular_sends,  // leading sends whose downstream relays the stripe (signal at granularity)
    neighbor_chip_id,    // packet header 2D route
    neighbor_mesh_id,
    fabric_args,  // mux client or routing-plane connection args (only when num_iters > 0)
};
}  // namespace rt
}  // namespace unicast_writer

namespace multicast_reader {
namespace ct {
enum : uint32_t {
    in_cpp = 0,
    chunk_size,
    out_cpp,
    cps,
    num_devices,
    cb_id,
    cb_depth,  // also the in-flight transaction id count
    cb_page_size,
    packet_size,
    alt_routes,       // load_balance_across_alt_routes
    num_connections,  // forward fabric connections (E line + S rect)
    k,
    walk_stride,
    walk_xfer,
    slot_bytes,
    in_runs,
    out_runs,
    tensor_args,  // TensorAccessorArgs(input), then TensorAccessorArgs(output)
};
}  // namespace ct
namespace rt {
enum : uint32_t {
    input_addr = 0,
    output_addr,
    slice_first,
    slice_chunks,
    device_idx,
    barrier_sem,
    barrier_noc_x,
    barrier_noc_y,
    barrier_wait_value,
    rot,
    route_args,                     // McastRouteArgs::from_args: 12 words
    fabric_args = route_args + 12,  // routing-plane connection args
};
}  // namespace rt
}  // namespace multicast_reader

namespace multicast_writer {
namespace ct {
enum : uint32_t {
    chunk_size = 0,
    out_cpp,
    cps,
    num_devices,
    cb_id,
    cb_page_size,
    packet_size,
    alt_routes,
    num_connections,  // backward fabric connections (W line + N rect)
    k,
    walk_stride,
    walk_xfer,
    slot_bytes,
    out_runs,
    tensor_args,  // TensorAccessorArgs(output)
};
}  // namespace ct
namespace rt {
enum : uint32_t {
    output_addr = 0,
    slice_first,
    slice_chunks,
    device_idx,
    barrier_sem,
    barrier_noc_x,
    barrier_noc_y,
    rot,
    route_args,                     // McastRouteArgs::from_args: 12 words
    fabric_args = route_args + 12,  // routing-plane connection args
};
}  // namespace rt
}  // namespace multicast_writer

}  // namespace gather_args
