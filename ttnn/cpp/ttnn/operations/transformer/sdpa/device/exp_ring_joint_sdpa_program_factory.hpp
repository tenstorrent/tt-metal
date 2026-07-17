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

// The per-link GlobalSemaphore addresses are excluded from the program-cache key
// (ExpRingJointSDPAParams::attribute_values omits `semaphore`), so a cache hit with a different
// semaphore set is possible. The factory bakes them for the cache-miss build, and
// ExpRingJointSDPADeviceOperation::override_runtime_arguments() re-derives + re-applies them (plus all
// other rt-args and CB addresses) on every dispatch, so no frozen address can be reused.
struct ExpRingJointSDPAProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const ExpRingJointSDPAParams& operation_attributes,
        const ExpRingJointSDPAInputs& tensor_args,
        ExpRingJointSDPAResult& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim
