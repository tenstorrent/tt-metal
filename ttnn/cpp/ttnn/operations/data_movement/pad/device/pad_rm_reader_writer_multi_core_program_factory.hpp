// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim {

struct PadRmReaderWriterMultiCoreProgramFactory {
    // The pad-value const tensor is allocated once on cache miss inside create_program_artifacts()
    // and handed to the framework as an op-owned tensor, so it outlives the cache miss and keeps a
    // stable address for the cached Program's lifetime.  The owning MeshTensor is moved out of the
    // build Tensor with release_mesh_tensor(): holding the SOURCE Tensor would not be enough on its
    // own, because ~Tensor force-deallocates the device memory through DeviceStorage::deallocate
    // regardless of external shared_ptr<MeshBuffer> owners (see #44565).
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output);
};
}  // namespace ttnn::prim
