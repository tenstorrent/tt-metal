// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host helpers the all_gather factories and the factory selector share. Private to the op: not installed.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "all_gather_device_operation_types.hpp"
#include "ttnn/operations/ccl/all_gather/shared_with_host/all_gather_walk.hpp"

namespace ttnn::operations::ccl {

// The kernels' walk geometry (all_gather_walk.hpp): this device's input pages and the output pages cut into units that
// are a multiple of the NOC write alignment and of the read alignment of every source the kernels read, or nullopt
// when the pages cannot be cut so. reads_output: the kernels also read the output (the unicast relay re-reads what
// upstream wrote there), so the unit is also a multiple of the output's read alignment. Fails when the walk's output
// unit index would not fit 32 bits.
std::optional<all_gather_walk::WalkGeo> compute_walk_geometry(
    const Tensor& input_tensor,
    const tt::tt_metal::TensorSpec& output_spec,
    int32_t gather_dim_from_end,
    uint32_t num_devices,
    uint32_t packet_size,
    bool reads_output);

// The kernels' packet payload cap for this output (all_gather_walk::packet_cap).
uint32_t compute_packet_cap(const all_gather_walk::WalkGeo& geo, const Tensor& output_tensor);

// Walk elements of this device's input.
uint32_t num_walk_elements(const all_gather_walk::WalkGeo& geo, const Tensor& input_tensor);

// The most packets one CB slot holds.
inline constexpr uint32_t kMaxPacketsPerSlot = 32;

// Packets per CB slot when the walk elements are split over num_slices workers. header_bytes: the fabric packet header
// the writer keeps in the slot, 0 for none (all_gather_walk::slot_offsets). min_slots: the slots a worker iteration
// should fill at least (all_gather_walk::kNumSlots, so both work). max_packets_per_slot: at most kMaxPacketsPerSlot.
uint32_t compute_packets_per_slot(
    const all_gather_walk::WalkGeo& geo,
    uint32_t packet_cap,
    uint32_t num_slices,
    uint32_t header_bytes,
    const Tensor& input_tensor,
    uint32_t min_slots = all_gather_walk::kNumSlots,
    uint32_t max_packets_per_slot = kMaxPacketsPerSlot);

// Worker cores the op may use: the sub-device's, within sub_core_grid.
CoreRangeSet get_available_worker_cores(const AllGatherParams& args, const Tensor& input_tensor);

// Unicast cores per link: workers_per_dir workers in each direction, plus the direction's fabric mux from 2 workers
// (one worker connects straight to the fabric router).
inline constexpr uint32_t unicast_cores_per_link(uint32_t workers_per_dir) {
    return 2u * (workers_per_dir + (workers_per_dir > 1 ? 1u : 0u));
}

// Whether the unicast relay re-reads the output: only when a direction runs 2 or more iterations, i.e. a line of 3 or
// more devices or a ring of 4 or more.
inline constexpr bool unicast_relay_reads_output(bool is_ring, uint32_t num_devices) {
    return is_ring ? num_devices >= 4 : num_devices >= 3;
}

// Appends the walk geometry and the packet cap as kernel compile-time args, in the order
// all_gather_walk::walk_geo_from_ct_args and packet_cap_from_ct_args read them (kWalkCtArgCount args).
inline void append_walk_ct_args(std::vector<uint32_t>& args, const all_gather_walk::WalkGeo& geo, uint32_t packet_cap) {
    args.insert(
        args.end(),
        {geo.unit_bytes,
         geo.in_page_units,
         geo.out_page_units,
         geo.stripe_units,
         geo.num_devices,
         geo.element_units,
         geo.max_packet_bytes,
         geo.dst_cross,
         geo.src_cross,
         packet_cap});
}

// The worker's slice of num_elements walk elements split over num_slices workers: contiguous, the first
// num_elements % num_slices slices one element longer.
struct WalkSlice {
    uint32_t start, count;
};
inline WalkSlice walk_slice(uint32_t num_elements, uint32_t num_slices, uint32_t slice_idx) {
    const uint32_t per_slice = num_elements / num_slices;
    const uint32_t remainder = num_elements % num_slices;
    return {
        .start = (slice_idx * per_slice) + std::min(slice_idx, remainder),
        .count = per_slice + (slice_idx < remainder ? 1 : 0)};
}

// The CB of the worker cores: kNumSlots slots of slot_bytes. A slot is untyped bytes; the format only sizes the page.
inline tt::tt_metal::CBDescriptor slot_cb_descriptor(
    uint32_t cb_id, uint32_t slot_bytes, const CoreRangeSet& worker_core_range) {
    return tt::tt_metal::CBDescriptor{
        .total_size = all_gather_walk::kNumSlots * slot_bytes,
        .core_ranges = worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = tt::DataFormat::RawUInt8,
            .page_size = slot_bytes,
        }}},
    };
}

}  // namespace ttnn::operations::ccl
