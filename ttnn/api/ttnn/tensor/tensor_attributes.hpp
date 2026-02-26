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
    TensorAttributes() : tensor_attributes(Deallocated{}) {}
    TensorAttributes(HostTensor host_tensor) : tensor_attributes(std::move(host_tensor)) {}
    TensorAttributes(MeshTensor mesh_tensor) : tensor_attributes(std::move(mesh_tensor)) {}

    // Deallocation related
    void deallocate() { tensor_attributes = Deallocated{}; }

    bool is_allocated() const { return std::holds_alternative<Deallocated>(tensor_attributes); }

    StorageType storage_type() const {
        return std::visit(
            tt::stl::overloaded{
                [](const Deallocated&) {
                    TT_THROW("Tensor is deallocated");
                    // Unreachable
                    return StorageType::HOST;
                },
                [](const HostTensor&) { return StorageType::HOST; },
                [](const MeshTensor&) { return StorageType::DEVICE; },
            },
            tensor_attributes);
    }

    HostTensor& host_tensor() {
        TT_FATAL(storage_type() == StorageType::HOST, "Tensor is not on host");
        return std::get<HostTensor>(tensor_attributes);
    }

    const HostTensor& host_tensor() const {
        TT_FATAL(storage_type() == StorageType::HOST, "Tensor is not on host");
        return std::get<HostTensor>(tensor_attributes);
    }

    MeshTensor& mesh_tensor() {
        TT_FATAL(storage_type() == StorageType::DEVICE, "Tensor is not on device");
        return std::get<MeshTensor>(tensor_attributes);
    }

    const MeshTensor& mesh_tensor() const {
        TT_FATAL(storage_type() == StorageType::DEVICE, "Tensor is not on device");
        return std::get<MeshTensor>(tensor_attributes);
    }

private:
    struct Deallocated {};
    std::variant<Deallocated, HostTensor, MeshTensor> tensor_attributes;
};

}  // namespace tt::tt_metal
