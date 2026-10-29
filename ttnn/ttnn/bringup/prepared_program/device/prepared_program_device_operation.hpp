// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "../prepared_program.hpp"

namespace ttnn::operations::bringup::prepared_program {

struct operation_attributes_t {
    std::shared_ptr<const Prepared> prepared;
    std::vector<uint32_t> values;  // (value_slots option: this run's values, patched every call, not in the key)

    // (the descriptor is too large for reflection inline storage; keyed by compute_program_hash)
    static constexpr auto attribute_names = std::forward_as_tuple("hash");
    auto attribute_values() const { return std::make_tuple(prepared->hash); }
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

// (as generic_op: the output is the last io tensor)
struct tensor_args_t {
    const std::vector<Tensor>& io_tensors;
    const Tensor& output_tensor;
};

struct PreparedProgramFactory {
    // cache miss: the stored descriptor of this coordinate with this call's io tensors patched in
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // cache hit: only the recorded runtime-arg slots and CBs, onto this call's io tensors
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

struct PreparedProgramDeviceOperation {
    // (the descriptor carries every address itself: nothing is derived per core from a tensor)
    static constexpr bool supports_per_core_allocation = true;

    using operation_attributes_t = prepared_program::operation_attributes_t;
    using tensor_args_t = prepared_program::tensor_args_t;
    using spec_return_value_t = prepared_program::spec_return_value_t;
    using tensor_return_value_t = prepared_program::tensor_return_value_t;
    using program_factory_t = std::variant<PreparedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::bringup::prepared_program
