// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct Conv2dShardedProgramFactory {
    // Builds the workload in one call (cache miss).  The intermediate
    // conv_reader_indices tensor — which must outlive the cached program — is
    // allocated here and parked on the WorkloadDescriptor's `buffers` vector
    // (wrapped in `shared_ptr<Tensor>` so `~Tensor` cannot force-deallocate the
    // device memory while the cached program is still alive).
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const Conv2dParams& operation_attributes,
        const Conv2dInputs& tensor_args,
        Tensor& output_tensor,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // desc.kernels starts with the weights sender writer, then the receiver writer when the weights are multicast.
    static constexpr uint32_t kWriterSenderKernel = 0;
    static constexpr uint32_t kWriterReceiverKernel = 1;
    // The writers' common runtime args with epilogue_stats: the stats buffer, then the page base.
    static constexpr uint32_t kStatsPageBaseArg = 1;
};

}  // namespace ttnn::prim
