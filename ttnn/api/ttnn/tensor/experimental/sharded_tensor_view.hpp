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
 * Create a sharded SRAM tensor view at a byte offset within an owner tensor.
 *
 * The result retains the owner's MeshBuffer. The view specification must use
 * the owner's allocation mode and a subset of its worker cores.
 */
Tensor create_sharded_tensor_view(
    const Tensor& owner, const tt::tt_metal::TensorSpec& tensor_spec, tt::tt_metal::DeviceAddr shard_offset);

}  // namespace ttnn::experimental
