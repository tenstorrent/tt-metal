// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "toy_scaled_add_device_operation_types.hpp"
#include "toy_scaled_add_program_factory.hpp"

namespace ttnn::operations::toy_scaled_add {

// out = a + alpha * (b * gamma), gamma an optional row broadcast down the rows.
//
// The order in which the framework calls these hooks on every launch (ttnn/api/ttnn/device_operation.hpp,
// ttnn/api/ttnn/mesh_device_operation_adapter.hpp):
//
//   1. every tensor in tensor_args is checked to be an allocated device tensor;
//   2. create_output_tensors allocates the output (before any validation, so the public entry checks the
//      support contract first, see check_support);
//   3. compute_program_hash gives the cache key; the program cache is looked up;
//   4a. hit:  validate_on_program_cache_hit, then the factory's override_runtime_arguments on the cached
//             Program;
//   4b. miss: validate_on_program_cache_miss, select_program_factory, the factory's create_descriptor, a
//             Program built from the descriptor and cached;
//   5. the program is enqueued.
struct ToyScaledAddDeviceOperation {
    using operation_attributes_t = ToyScaledAddParams;
    using tensor_args_t = ToyScaledAddInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<InterleavedProgramFactory, HeightShardedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // The full contract, checked once per cached program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // A hit has the same cache key as the miss that passed the full check, so only what the key
    // leaves out is checked again: which device each tensor is on, and the preallocated output.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// The support contract of ttnn/ttnn/operations/toy_scaled_add/toy_scaled_add.py (SUPPORTED and EXCLUSIONS):
// throws UnsupportedAxisValue or ExcludedCell. The public entry calls it before launching, so a refused input
// never allocates an output.
void check_support(const ToyScaledAddParams& attrs, const ToyScaledAddInputs& t);

}  // namespace ttnn::operations::toy_scaled_add

namespace ttnn::prim {

Tensor toy_scaled_add(
    const Tensor& a,
    const Tensor& b,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& output,
    const ttnn::operations::toy_scaled_add::ToyScaledAddParams& params);

}  // namespace ttnn::prim
