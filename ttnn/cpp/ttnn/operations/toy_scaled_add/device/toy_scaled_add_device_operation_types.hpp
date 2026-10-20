// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <stdexcept>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::toy_scaled_add {

// The operation's attributes: everything the public entry resolved before launch.
//
// The framework hashes, prints and serializes attributes through reflection (tt_stl/reflection.hpp), so an
// attribute can be any aggregate of fields it already knows: scalars, enums, MemoryConfig,
// DeviceComputeKernelConfig, optionals and vectors of those. Without a compute_program_hash, an operation's
// cache key is every attribute plus the specs of its tensor args. This op writes its own key only to leave
// alpha out (ToyScaledAddDeviceOperation::compute_program_hash).
struct ToyScaledAddParams {
    float alpha = 1.0f;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::BFLOAT16;
    tt::tt_metal::MemoryConfig output_memory_config;
    // Resolved for the device's architecture by the public entry, so equal configs hash equal.
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ToyScaledAddInputs {
    const Tensor& a;
    const Tensor& b;
    const std::optional<Tensor>& gamma;
    // Preallocated output; may alias `a` (in place).
    const std::optional<Tensor>& output;
};

// Refusals of the support contract, named after their Python counterparts in ttnn.operations._op_contract.
// The binding (toy_scaled_add_nanobind.cpp) raises each as that Python type, so callers and the eval harness
// see the same exception from this op as from its generic_op version. Inputs that do not fit together
// (mismatched shapes or shard specs) are TT_FATAL errors instead.
struct SupportRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct UnsupportedAxisValue : SupportRefusal {
    using SupportRefusal::SupportRefusal;
};
struct ExcludedCell : SupportRefusal {
    using SupportRefusal::SupportRefusal;
};

}  // namespace ttnn::operations::toy_scaled_add
