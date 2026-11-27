// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "groupnorm_sc_N_1_HW_C_b_device_operation_types.hpp"

namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b {

// The descriptor holds its kernels in the Python order: readers (NoC pairing 0, then 1), writers ((root, 0),
// (root, 1), (member, 0), (member, 1)), computes (root, member); only non-empty groups. A kernel's index in the
// cached Program follows from these counts, which the miss records with the program.
struct KernelLayout {
    uint32_t num_readers = 0;
    uint32_t num_writers = 0;
    uint32_t num_computes = 0;
    uint32_t first_writer() const { return num_readers; }
    uint32_t first_compute() const { return num_readers + num_writers; }
};

// Cache miss: looks up the memoized plan, builds its work split and every CB, kernel and semaphore.
struct GroupnormScN1HwCbProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GroupnormScN1HwCbParams& attrs, const GroupnormScN1HwCbInputs& tensors, Tensor& output);
};

// Every value a hit writes, i.e. everything the key leaves out: the tensor addresses (reader IN / GAMMA / BETA / OUT,
// writer OUT, the shard-backed CBs) and compute EPS_BITS.
struct GroupnormScN1HwCbHitValues {
    uint32_t in_addr = 0;
    uint32_t gamma_addr = 0;
    uint32_t beta_addr = 0;
    uint32_t out_addr = 0;
    uint32_t eps_bits = 0;
    bool operator==(const GroupnormScN1HwCbHitValues&) const = default;
};

// Owns the cached workload so that a hit reads what the miss recorded with each program (its kernel layout and its
// shard-backed CBs) instead of re-solving the plan, and writes nothing when the call repeats the last completed hit.
struct GroupnormScN1HwCbMeshWorkloadFactory {
    using descriptor_factory_t = GroupnormScN1HwCbProgramFactory;  // every program comes from its create_descriptor

    // A CB backed by a shard: the input's (INPUT_RM / INPUT_TILES) or the output's.
    struct ShardCb {
        tt::tt_metal::CBHandle handle = 0;
        bool backs_input = false;
    };
    struct shared_variables_t {
        KernelLayout layout;
        std::vector<ShardCb> shard_cbs;
        // What the last completed hit wrote to this program; empty after the miss and while a hit is writing.
        std::optional<GroupnormScN1HwCbHitValues> applied;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    // Cache miss: one program per range from create_descriptor (which takes no mesh coordinate), with its record.
    static cached_mesh_workload_t create_mesh_workload(
        const GroupnormScN1HwCbParams& attrs,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const GroupnormScN1HwCbInputs& tensors,
        Tensor& output);

    // Cache hit: rewrites the common address slots and EPS_BITS of every kernel, and re-points a shard-backed CB whose
    // tensor moved, only in a program whose record differs from the call's values.
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const GroupnormScN1HwCbParams& attrs,
        const GroupnormScN1HwCbInputs& tensors,
        Tensor& output);
};

}  // namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b
