// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::unit_mesh {

// Aggregates tensors from unit meshes (1x1 submeshes) into a single tensor on the parent mesh.
//
// All input tensors must be allocated on unit meshes that share the same parent mesh, have identical
// TensorSpecs, and their mesh buffers must be at the same address. The number of input tensors must
// match the parent mesh size.
//
// Returns a tensor distributed across the parent mesh.
ttnn::Tensor aggregate(const std::vector<ttnn::Tensor>& tensors);

// Disaggregates a tensor from a parent mesh into individual tensors on unit meshes (1x1 submeshes).
//
// The input tensor must be allocated on a mesh device that has submeshes; the number of submeshes must match the
// parent mesh size, and each submesh must be a unit mesh (1x1).
//
// Returns a vector of tensors, one per submesh, all sharing the same buffer address.
std::vector<ttnn::Tensor> disaggregate(const ttnn::Tensor& tensor);

// Non-owning view of an interleaved (DRAM / L1) device tensor's pages on `mesh_device` (the tensor's own mesh, its
// parent mesh, or one of its unit submeshes -- they share physical memory): a tensor of logical `shape` (same dtype,
// layout, memory config and page size as `tensor`) whose page 0 is page `page_offset` of `tensor`. `page_offset`
// must be a multiple of the number of banks (so page p of the view lands in the same bank as page p + page_offset
// of `tensor`). The view does not keep `tensor` alive and is not known to any allocator: the caller must keep the
// source allocated (and, across meshes, reserved) for the view's lifetime. Experimental.
ttnn::Tensor view_pages(
    const ttnn::Tensor& tensor,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint64_t page_offset,
    const ttnn::Shape& shape);

// Reserve in `target`'s allocator every region `source`'s allocator has allocated (DRAM, L1, L1_SMALL, TRACE) that
// `target` has free (see tt::tt_metal::experimental::reserve_allocator_regions). Returns bytes reserved per bank.
uint64_t reserve_allocator_regions(
    tt::tt_metal::distributed::MeshDevice* target, tt::tt_metal::distributed::MeshDevice* source);

}  // namespace ttnn::experimental::unit_mesh
