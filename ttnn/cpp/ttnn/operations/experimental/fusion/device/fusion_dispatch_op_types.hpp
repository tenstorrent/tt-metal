// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <tt_stl/reflection.hpp>

#include <vector>

namespace ttnn::operations::generic {
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor);
}

namespace ttnn::operations::experimental::fusion {

struct fusion_dispatch_operation_attributes_t : tt::tt_metal::experimental::MeshProgramDescriptor {
    fusion_dispatch_operation_attributes_t() = default;
    fusion_dispatch_operation_attributes_t(
        const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor) :
        tt::tt_metal::experimental::MeshProgramDescriptor(mesh_program_descriptor) {}
    fusion_dispatch_operation_attributes_t(
        tt::tt_metal::experimental::MeshProgramDescriptor&& mesh_program_descriptor) :
        tt::tt_metal::experimental::MeshProgramDescriptor(std::move(mesh_program_descriptor)) {}

    static constexpr auto attribute_names = std::forward_as_tuple("mesh_coord_ranges", "program_descriptor_hashes");
    auto attribute_values() const {
        std::vector<tt::tt_metal::distributed::MeshCoordinateRange> mesh_coord_ranges;
        std::vector<ttsl::hash::hash_t> program_descriptor_hashes;
        mesh_coord_ranges.reserve(mesh_programs.size());
        program_descriptor_hashes.reserve(mesh_programs.size());
        for (const auto& [mesh_coord_range, program_descriptor] : mesh_programs) {
            mesh_coord_ranges.push_back(mesh_coord_range);
            program_descriptor_hashes.push_back(
                ttnn::operations::generic::compute_program_descriptor_hash(program_descriptor));
        }
        return std::make_tuple(std::move(mesh_coord_ranges), std::move(program_descriptor_hashes));
    }
};

using fusion_dispatch_tensor_return_value_t = Tensor;
using fusion_dispatch_spec_return_value_t = TensorSpec;

struct fusion_dispatch_tensor_args_t {
    const std::vector<Tensor>& io_tensors;
    const Tensor& output_tensor;
};

}  // namespace ttnn::operations::experimental::fusion
