// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exception types for unit mesh tensor operations.
// Use these instead of asserting on exception message substrings in tests.
#pragma once
#include <stdexcept>
#include <string>

namespace ttnn {

// Thrown when aggregate() is called with an empty tensor vector.
struct UnitMeshEmptyTensorVector : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Thrown when a tensor is not on a unit mesh (1x1).
struct UnitMeshNotUnitMesh : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Thrown when tensors to aggregate have mismatched TensorSpecs.
struct UnitMeshTensorSpecMismatch : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Thrown when tensors to aggregate have mismatched mesh buffer addresses.
struct UnitMeshBufferAddressMismatch : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Thrown when input tensors do not span the entire parent mesh.
struct UnitMeshTensorsNotSpanningMesh : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Thrown when the number of submeshes doesn't match the mesh size.
struct UnitMeshSubmeshCountMismatch : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn
