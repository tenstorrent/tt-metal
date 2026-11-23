// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "combine_fabric2d_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <ttnn/global_semaphore.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

struct CombineFabric2dProgramFactory {
    // One ProgramDescriptor per mesh coordinate: each chip sends to its own neighbor, so compile-time args
    // are coord-dependent and cannot be replicated.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const CombineFabric2dParams& operation_attributes,
        const CombineFabric2dInputs& tensor_args,
        ttnn::Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// The spec of the forwarding buffer, every stream's region back to back: one page per relayed token, the token plus
// its forwarding metadata. `token_size_bytes` is a token's bytes as the op moves it, emb_dim BFLOAT16 values. Reads
// only the device, axis, num_links and fwd_vc_pages of `args`, so the standalone op and the overlapped fork size it
// alike. A caller that passes its own buffer must make it at least this large.
tt::tt_metal::TensorSpec forwarding_buffer_spec(const CombineFabric2dParams& args, uint32_t token_size_bytes);

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
