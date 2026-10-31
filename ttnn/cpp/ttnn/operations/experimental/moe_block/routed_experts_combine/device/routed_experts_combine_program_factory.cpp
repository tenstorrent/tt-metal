// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine_program_factory.hpp"

#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "combine_program.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

tt::tt_metal::WorkloadDescriptor RoutedExpertsCombineFactory::create_workload_descriptor(
    const RoutedExpertsCombineParams& params,
    const RoutedExpertsCombineInputs& t,
    ttnn::Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    TT_FATAL(params.plan, "routed_experts_combine: no combine plan (call through the host entry point)");
    const auto& plan = *params.plan;
    auto& mesh_device = *t.out.device();
    const CombineBuffers buffers{
        .plan = t.plan.buffer(),
        .table = t.table.buffer(),
        .out = output.buffer(),
        .bar = t.bar.buffer(),
        .hcache = t.hcache.buffer(),
        .lut = t.lut ? t.lut->buffer() : nullptr};
    tt::tt_metal::WorkloadDescriptor wd;
    for (const auto& range : tensor_coords.ranges()) {
        for (const auto& coord : range) {
            const ChipCoord chip{coord[0], coord[1]};
            tt::tt_metal::ProgramDescriptor desc;
            // 1. the arena CB and the injection rings: the first CBs of the program (combine_program.hpp)
            add_combine_arena_cbs(desc, plan, chip);
            // 2. SEAM(flat): the flat routed expert's kernels, CBs and runtime args on its cells, from
            //    flat_routed_expert/device/flat_program.hpp, with its arena CB at index kCbArena (the first CB on its
            //    cores), its y writers on the L1 hand-off (the HandoffArgs below: per writer the l1_args with its
            //    index, the collector at counts_offset) and its writers zeroing their words and counting in on the
            //    gate. With dispatch_in_program, the dispatch phase (sparse_dispatch_2d's reader / sender as functions)
            //    runs first in the combine's port kernels, and the flat x relays wait for its ready count (x_ready).
            // 3. the combine
            const auto handoff = add_combine_program(desc, plan, chip, mesh_device, buffers, params.flat);
            (void)handoff;  // SEAM(flat): patched into the flat writers' runtime args
            wd.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(desc)});
        }
    }
    return wd;
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
