// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_attributes.hpp"

namespace tt::tt_metal {

TensorAttributes::TensorAttributes(HostTensor host_tensor) : tensor_attributes(std::move(host_tensor)) {}

TensorAttributes::TensorAttributes(MeshTensor mesh_tensor) : tensor_attributes(std::move(mesh_tensor)) {}

StorageType TensorAttributes::storage_type() const {
    return std::visit(
        tt::stl::overloaded{
            [](const HostTensor&) { return StorageType::HOST; },
            [](const MeshTensor&) { return StorageType::DEVICE; },
        },
        tensor_attributes);
}

HostTensor& TensorAttributes::host_tensor() {
    TT_FATAL(storage_type() == StorageType::HOST, "Tensor is not on host");
    return std::get<HostTensor>(tensor_attributes);
}

const HostTensor& TensorAttributes::host_tensor() const {
    TT_FATAL(storage_type() == StorageType::HOST, "Tensor is not on host");
    return std::get<HostTensor>(tensor_attributes);
}

MeshTensor& TensorAttributes::mesh_tensor() {
    TT_FATAL(storage_type() == StorageType::DEVICE, "Tensor is not on device");
    return std::get<MeshTensor>(tensor_attributes);
}

const MeshTensor& TensorAttributes::mesh_tensor() const {
    TT_FATAL(storage_type() == StorageType::DEVICE, "Tensor is not on device");
    return std::get<MeshTensor>(tensor_attributes);
}

}  // namespace tt::tt_metal
