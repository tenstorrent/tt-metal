// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "dispatch_fabric2d_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <ttnn/global_semaphore.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

struct DispatchFabric2dProgramFactory {
    using tensor_return_value_t = std::array<ttnn::Tensor, 2>;

    // One ProgramDescriptor per mesh coordinate: each chip's senders name their own downstream workers,
    // so compile-time args are coord-dependent and cannot be replicated.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const DispatchFabric2dParams& args,
        const DispatchFabric2dInputs& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// The output payload's page: one bf16 token of `emb_dim`, padded to the DRAM alignment.
uint32_t token_page_bytes(ttnn::MeshDevice* mesh, uint32_t emb_dim);

// The DRAM buffers the op keeps tokens in: the forwarding buffer, every stream's fwd_section back to back, and for a
// TILE input the staging buffer the untilizers write. `token_size_bytes` is token_page_bytes. They read only the
// device, axis, num_links, fwd_vc_pages and seq_len_per_chip of `args`. A caller that passes its own buffers must
// make them at least this large.
tt::tt_metal::TensorSpec forwarding_buffer_spec(const DispatchFabric2dParams& args, uint32_t token_size_bytes);
tt::tt_metal::TensorSpec staging_buffer_spec(const DispatchFabric2dParams& args, uint32_t token_size_bytes);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
