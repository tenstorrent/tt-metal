// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/types.hpp"

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace tt::tt_metal {

class TensorAttributes {
public:
    TensorAttributes() = default;
    TensorAttributes(HostTensor host_tensor);
    TensorAttributes(MeshTensor mesh_tensor);

    StorageType storage_type() const;

    HostTensor& host_tensor();
    const HostTensor& host_tensor() const;

    MeshTensor& mesh_tensor();
    const MeshTensor& mesh_tensor() const;

private:
    std::variant<HostTensor, MeshTensor> tensor_attributes;
};

}  // namespace tt::tt_metal
