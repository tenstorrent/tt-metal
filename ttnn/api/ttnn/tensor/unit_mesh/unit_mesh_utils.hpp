// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::unit_mesh {

using ttnn::Tensor;

// Aggregates tensors from unit meshes (1x1 submeshes) into a single tensor on the parent mesh.
//
// All input tensors must be allocated on unit meshes that share the same parent mesh, have identical
// TensorSpecs, and their mesh buffers must be at the same address. The number of input tensors must
// match the parent mesh size.
//
// Returns a tensor distributed across the parent mesh.
Tensor aggregate(const std::vector<Tensor>& tensors);

// Disaggregates a tensor from a parent mesh into individual tensors on unit meshes (1x1 submeshes).
//
// The input tensor must be allocated on a mesh device that has submeshes; the number of submeshes must match the
// parent mesh size, and each submesh must be a unit mesh (1x1).
//
// Returns a vector of tensors, one per submesh, all sharing the same buffer address.
std::vector<Tensor> disaggregate(const Tensor& tensor);

}  // namespace ttnn::experimental::unit_mesh

namespace tt::tt_metal::experimental::unit_mesh {

// TODO(deprecate): temporary backward-compat aliases while call sites migrate to ttnn::experimental::unit_mesh.
using ttnn::experimental::unit_mesh::aggregate;
using ttnn::experimental::unit_mesh::disaggregate;

}  // namespace tt::tt_metal::experimental::unit_mesh
