// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <span>

#include "agmm_registry_descriptor.hpp"
#include "ttnn/config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"

namespace ttnn::experimental::all_gather_minimal_matmul_registry {
using Mode = ttnn::MatmulRegistryMode;

struct RegistryRequestFacts {
    compact::DeviceDescriptor device{};
    compact::WorkloadDescriptor workload{};
    compact::OperationDescriptor operation{};
    compact::TensorDescriptor input{};
    compact::TensorDescriptor weight{};
    compact::OptionalTensorDescriptor bias{};
    compact::OptionalTensorDescriptor ternary_input_a{};
    compact::OptionalTensorDescriptor ternary_input_b{};
    compact::OptionalTensorDescriptor persistent_output{};
    compact::OptionalTensorDescriptor persistent_weight{};
};

struct Recipe {
    ttnn::experimental::prim::MinimalMatmulConfig config{};
    DeviceComputeKernelConfig compute_kernel_config{};
};

// Exact, allocation-free construction and lookup. In a live request the
// device grid is the available grid; in an emitted entry it is the minimum
// grid on which that recipe was certified. Unsupported device counts and the
// unsupported device counts return no recipe.
std::optional<compact::KeyDescriptor> build_registry_key(const RegistryRequestFacts& facts) noexcept;
std::span<const compact::CohortDescriptor> cohorts_for_device_count(std::uint16_t device_count) noexcept;
const compact::EntryDescriptor* lookup(const compact::KeyDescriptor& key) noexcept;
std::optional<Recipe> materialize_recipe(const compact::EntryDescriptor& descriptor) noexcept;
std::optional<Recipe> select_recipe(Mode mode, const RegistryRequestFacts& facts);

}  // namespace ttnn::experimental::all_gather_minimal_matmul_registry
