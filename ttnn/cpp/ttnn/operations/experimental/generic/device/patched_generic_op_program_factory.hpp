// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "patched_generic_op_types.hpp"

#include <tt-metalium/core_coord.hpp>

#include <cstdint>
#include <vector>

namespace ttnn::operations::experimental::generic::program {

struct PatchedGenericMeshProgramFactory {
    /// Per-core runtime arg whose value is a tensor buffer address (PR 39972-style patching).
    struct PerCoreRuntimeArgSlot {
        std::uint32_t kernel_idx{};
        CoreCoord core{};
        std::uint32_t arg_idx{};
        std::uint32_t io_tensor_index{};
    };

    struct CommonRuntimeArgSlot {
        std::uint32_t kernel_idx{};
        std::uint32_t arg_idx{};
        std::uint32_t io_tensor_index{};
    };

    struct CBTensorSlot {
        std::uint32_t cb_idx{};
        std::uint32_t io_tensor_index{};
    };

    struct shared_variables_t {
        std::uint32_t num_kernel_handles{};
        std::vector<tt::tt_metal::CBHandle> cb_handles;
        std::vector<PerCoreRuntimeArgSlot> per_core_runtime_arg_slots;
        std::vector<CommonRuntimeArgSlot> common_runtime_arg_slots;
        std::vector<CBTensorSlot> cb_tensor_slots;

        /// Previous io_tensor addresses — enables skip-if-unchanged patching.
        /// Populated after the first override; empty until then.
        std::vector<std::uint32_t> prev_io_addresses;
    };

    struct mesh_shared_variables_t {
        shared_variables_t program_shared_variables;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<mesh_shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const patched_operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const patched_tensor_args_t& tensor_args,
        patched_tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_mesh_workload,
        const patched_operation_attributes_t& operation_attributes,
        const patched_tensor_args_t& tensor_args,
        patched_tensor_return_value_t& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const tt::tt_metal::ProgramDescriptor& program_descriptor, const patched_tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::experimental::generic::program
