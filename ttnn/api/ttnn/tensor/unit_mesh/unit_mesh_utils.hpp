// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::experimental::unit_mesh {

// Aggregates tensors from unit meshes (1x1 submeshes) into a single tensor on the parent mesh.
//
// All input tensors must be allocated on unit meshes that share the same parent mesh, have identical
// TensorSpecs, and their mesh buffers must be at the same address. The number of input tensors must
// match the parent mesh size.
//
// Returns a tensor distributed across the parent mesh.
Tensor aggregate(const std::vector<tt::tt_metal::Tensor>& tensors);

// Disaggregates a tensor from a parent mesh into individual tensors on unit meshes (1x1 submeshes).
//
// The input tensor must be allocated on a mesh device that has submeshes; the number of submeshes must match the
// parent mesh size, and each submesh must be a unit mesh (1x1).
//
// Returns a vector of tensors, one per submesh, all sharing the same buffer address.
std::vector<tt::tt_metal::Tensor> disaggregate(const tt::tt_metal::Tensor& tensor);

}  // namespace tt::tt_metal::experimental::unit_mesh
