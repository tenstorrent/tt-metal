// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/storage.hpp"

namespace ttnn {

// Builds retained-view storage for ttnn::experimental::create_sharded_tensor_view. Its members are defined in
// storage.cpp, where the storage internals they inspect are visible.
class RetainedTensorViewFactory {
public:
    // Throws unless `source` owns its allocation or is a retained view created from such storage. A reinterpretation
    // aliases memory through a MeshBuffer that neither owns nor retains it, so it cannot be a retained view's base.
    static void validate_source(const DeviceStorage& source);

    static DeviceStorage create(const DeviceStorage& owning_storage, tt::tt_metal::MeshTensor view_mesh_tensor);
};

}  // namespace ttnn
