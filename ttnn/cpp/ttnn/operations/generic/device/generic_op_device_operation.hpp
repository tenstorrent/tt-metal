// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <tt_stl/reflection.hpp>  // ttsl::hash::hash_t

#include "ttnn/tensor/tensor.hpp"
#include "generic_op_program_factory.hpp"
#include "generic_op_spec_factory.hpp"
#include "generic_op_device_operation_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::generic {

struct GenericOpDeviceOperation {
    // This op never derives an address from a tensor: the descriptor path returns the caller's
    // ProgramDescriptor verbatim, so resolving per-core addresses is the caller's job
    static constexpr bool supports_per_core_allocation = true;

    using operation_attributes_t = generic::operation_attributes_t;
    using tensor_args_t = generic::tensor_args_t;
    using spec_return_value_t = generic::spec_return_value_t;
    using tensor_return_value_t = generic::tensor_return_value_t;
    using program_factory_t = std::variant<program::GenericMeshDescriptorFactory, program::GenericSpecFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Note: will either compute a program hash, or simply return user provided custom program hash
    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};  // struct GenericOpDeviceOperation

// Structural hash of a ProgramSpec: the program cache key the spec path uses. Excludes runtime
// argument values and tensor addresses, which are re-applied on every dispatch.
ttsl::hash::hash_t compute_program_spec_hash(const tt::tt_metal::experimental::ProgramSpec& spec);

}  // namespace ttnn::operations::generic

namespace ttnn::prim {
ttnn::operations::generic::tensor_return_value_t generic_op(
    const std::vector<Tensor>& io_tensors,
    const ttnn::operations::generic::operation_attributes_t& operation_attributes);
}  // namespace ttnn::prim
