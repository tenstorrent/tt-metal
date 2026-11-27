// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/workload/legacy/program_descriptors.hpp>

#include "attn_res_gather_softmax_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim::qsr {

struct AttnResGatherSoftmaxSiteOffsets {
    uint32_t shift;
    uint32_t mass;
    uint32_t partial;
};

AttnResGatherSoftmaxSiteOffsets attn_res_gather_softmax_site_offsets(
    const AttnResGatherSoftmaxParams& operation_attributes, const AttnResGatherSoftmaxInputs& tensor_args);

struct AttnResGatherSoftmaxProgramFactory {
    // Per mesh coordinate rather than once for the mesh: a chip's rank on the
    // tensor-parallel axis decides which slot of the statistics tensor it fills and
    // which way each peer lies on the fabric.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AttnResGatherSoftmaxParams& operation_attributes,
        const AttnResGatherSoftmaxInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);

    // Cache-hit refresh. `site` and the caller's semaphore are left out of the program hash,
    // so beyond every buffer address this re-applies the site page offsets and the
    // semaphore address. Everything else is derived from hashed inputs and the coordinate.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const AttnResGatherSoftmaxParams& operation_attributes,
        const AttnResGatherSoftmaxInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);
};

}  // namespace ttnn::experimental::prim::qsr
