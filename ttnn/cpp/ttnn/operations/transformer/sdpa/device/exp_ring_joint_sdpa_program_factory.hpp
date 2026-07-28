// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation_types.hpp"

namespace ttnn::prim {

// Single source of truth for the DYNAMIC (hash-excluded) global-semaphore address runtime args.
//
// The per-link GlobalSemaphore addresses are excluded from the program-cache key
// (ExpRingJointSDPAParams::attribute_values omits `semaphore`), so two calls that differ only in
// which GlobalSemaphores they pass still cache-hit. That makes the addresses dynamic: the factory
// bakes them for the cache-miss build, and ExpRingJointSDPADeviceOperation::get_dynamic_runtime_args()
// re-applies them on every dispatch — otherwise a cache hit with a different semaphore set would
// silently reuse the address frozen at the first miss (the frozen-runtime-arg bug).
//
// The kernel indices and per-core arg slots below are the shared reference for BOTH the factory's
// cache-miss bake (build_exp_ring_joint_sdpa_program_descriptor) and the cache-hit patch
// (get_dynamic_runtime_args); reorder the runtime args in the factory and these constants (and thus
// the re-apply targets) must be updated in lockstep.
namespace exp_ring_joint_sdpa_dynamic {
// Kernel indices — must match the desc.kernels push order (reader, writer, writer_fabric, compute[, mux]).
inline constexpr uint32_t kReaderKernelIdx = 0;
inline constexpr uint32_t kWriterFabricKernelIdx = 2;
// Per-core reader runtime-arg slot of the first per-link semaphore address; slots
// kReaderSemaphoreArgBase .. +num_links-1 hold args.semaphore[lnk].address().
inline constexpr uint32_t kReaderSemaphoreArgBase = 26;
// Per-core fabric-writer runtime-arg slot of out_ready_sem_addr (= args.semaphore[link].address()).
inline constexpr uint32_t kWriterFabricOutReadySemArg = 25;
}  // namespace exp_ring_joint_sdpa_dynamic

struct ExpRingJointSDPAProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const ExpRingJointSDPAParams& operation_attributes,
        const ExpRingJointSDPAInputs& tensor_args,
        ExpRingJointSDPAResult& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim
