// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/distributed/types.hpp"

#include "select_final_carry_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Unlike the other KDA ops this factory builds legacy program descriptors: the metal2 ProgramSpec cannot yet open
// fabric connections, which the owner's multicast needs.
struct SelectFinalCarryProgramFactory {
    // One program per coordinate: each program's fabric routes depend on the device's place on the
    // sequence-parallel line. They differ only in runtime args (sp_rank, the multicast hops, the connections and
    // peers), so a range runs them as one program with per-device runtime args.
    static constexpr bool merge_per_device_programs = true;
    // At the ends of a line a device has one neighbor, so fewer fabric connections: its writer args are shorter.
    // The writer reads only the connections it is given, so the zero padding is never read.
    static constexpr bool pad_per_device_runtime_args = true;
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const SelectFinalCarryParams&,
        const SelectFinalCarryInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim
