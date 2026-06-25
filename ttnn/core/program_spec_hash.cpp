// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/program_spec_hash.hpp"

#include <reflect>
#include <tt_stl/reflection.hpp>

namespace ttnn::device_operation {

namespace {

// TensorParameter, with shape folded in per its declared relaxations. (Everything
// else about the parameter is identity-relevant and always hashed.)
std::size_t hash_tensor_parameter(const tt::tt_metal::experimental::TensorParameter& p) {
    const auto& spec = p.spec;
    // Data type, layout and memory config (which carries the sharding configuration) are always
    // identity-relevant, so they always go into the hash. Only the *shape* is folded in conditionally,
    // per the parameter's declared relaxations below.
    std::size_t hash = ttsl::hash::hash_objects_with_default_seed(
        p.unique_id.get(), spec.data_type(), spec.layout(), spec.memory_config());

    if (p.advanced_options.dynamic_tensor_shape) {
        // Program is agnostic to the tensor's shape (shape is an implicit runtime arg); fold no shape in.
    } else if (p.advanced_options.match_padded_shape_only) {
        // Logical shape may vary; the padded shape (and thus access pattern) is fixed.
        hash = ttsl::hash::hash_objects(hash, spec.padded_shape());
    } else {
        hash = ttsl::hash::hash_objects(hash, spec.logical_shape(), spec.padded_shape());
    }
    return hash;
}

}  // namespace

std::size_t program_spec_cache_key(const tt::tt_metal::experimental::ProgramSpec& spec) {
    // ProgramSpec currently has 7 fields; tensor_parameters is the only one needing
    // relaxation-aware handling, so it is hashed separately below and the other six
    // are hashed generically. If a field is added/removed, update this function (and
    // this count) so the new field participates in the key.
    static_assert(reflect::size<tt::tt_metal::experimental::ProgramSpec>() == 7);

    std::size_t hash = ttsl::hash::hash_objects_with_default_seed(
        spec.name,
        spec.kernels,
        spec.dataflow_buffers,
        spec.cross_node_dataflow_buffers,
        spec.semaphores,
        spec.work_units);

    for (const auto& p : spec.tensor_parameters) {
        hash = ttsl::hash::hash_objects(hash, hash_tensor_parameter(p));
    }
    return hash;
}

}  // namespace ttnn::device_operation
