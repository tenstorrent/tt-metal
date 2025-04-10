// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <functional>
#include <unordered_map>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program_cache.hpp>
#include <tt_stl/overloaded.hpp>

#include "ttnn/core.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation_concepts.hpp"

namespace ttnn::device_operation::mesh_device_operation_utils {

template <typename T>
using AdaptedCachedMeshWorkload = tt::tt_metal::program_cache::detail::AdaptedCachedMeshWorkload<T>;

// Sets runtime ID for all programs in `workload`.
inline void set_runtime_id(tt::tt_metal::distributed::MeshWorkload& workload, ttnn::MeshDevice* mesh_device) {
    for (auto& [_, program] : workload.get_programs()) {
        program.set_runtime_id(ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id());
    }
}

// Tracks all programs in `workload` and returns true if any program was hooked.
inline bool track_workload(tt::tt_metal::distributed::MeshWorkload& workload, ttnn::MeshDevice* mesh_device) {
    bool hook_program = false;
    for (auto& [_, program] : workload.get_programs()) {
        tt::tt_metal::GraphTracker::instance().track_program(&program, mesh_device);
        hook_program |= tt::tt_metal::GraphTracker::instance().hook_program(&program);
    }
    return hook_program;
}

// Applies override_runtime_arguments in a uniform way across different program factory interfaces because
// we have many different program factory interfaces we're currently supporting.
template <
    ProgramFactoryConcept ProgramFactory,
    typename OperationAttributes,
    typename TensorArgs,
    typename TensorReturnValue>
void apply_override_runtime_arguments(
    const ProgramFactory& factory,
    tt::tt_metal::Program& program,
    typename ProgramFactory::shared_variables_t& shared_vars,
    const OperationAttributes& attrs,
    const ttnn::MeshCoordinate& coord,
    const TensorArgs& tensor_args,
    TensorReturnValue& return_value) {
    if constexpr (requires {
                      typename ProgramFactory::cached_program_t;
                      factory.override_runtime_arguments(
                          std::declval<typename ProgramFactory::cached_program_t&>(), attrs, tensor_args, return_value);
                  }) {
        // Proxy references to `program` and `shared_vars` as a `CachedProgram` object.
        auto cached_program_proxy = ProgramFactory::cached_program_t::proxy(program, shared_vars);
        factory.override_runtime_arguments(cached_program_proxy, attrs, tensor_args, return_value);
    } else {
        factory.override_runtime_arguments(program, shared_vars, attrs, tensor_args, return_value);
    }
}

}  // namespace ttnn::device_operation::mesh_device_operation_utils
