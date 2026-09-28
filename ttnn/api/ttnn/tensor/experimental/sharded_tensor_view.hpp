// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

// Experimental and subject to change: this header carries no API-stability guarantee. It is kept out of widely
// included tensor headers so that ttnn::experimental does not become visible to code that uses the unqualified
// name experimental:: for tt::tt_metal::experimental.
namespace ttnn::experimental {

/**
 * Create a sharded L1 tensor view that starts `shard_offset` bytes into each shard of `owner`.
 *
 * `owner` must be a device tensor replicated across its mesh that owns its allocation or is itself a view created by
 * this function; reinterpreted tensors are rejected. `tensor_spec` must describe L1 storage with the owner's
 * allocation mode and a subset of its worker cores, and each view shard must fit within the owner shard at an aligned
 * offset.
 *
 * The view keeps the owner's allocation alive while it exists. Explicitly deallocating the owner frees that memory and
 * invalidates every view created from it, directly or through other views. Deallocating a view invalidates the views
 * created from it and releases only that view's hold on the owner's allocation. The full ownership model is described
 * with DeviceStorage in ttnn/tensor/storage.hpp.
 */
Tensor create_sharded_tensor_view(
    const Tensor& owner, const tt::tt_metal::TensorSpec& tensor_spec, tt::tt_metal::DeviceAddr shard_offset);

}  // namespace ttnn::experimental
