// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/mesh_socket.hpp>

#include "ttnn/run_operation.hpp"

#include <optional>
#include <vector>

namespace ttnn {

using ccl::EriscDatamoverBuilder;

struct RecvAsync {
    const tt::tt_metal::distributed::MeshSocket& mesh_socket;

    RecvAsync(const tt::tt_metal::distributed::MeshSocket& mesh_socket) : mesh_socket(mesh_socket) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        // TODO: Adding mesh_socket requires reflection support including json serialization
        // attrs.emplace_back("mesh_socket", mesh_socket);
        return attrs;
    }

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

tt::tt_metal::operation::ProgramWithCallbacks recv_async_multicore(
    const Tensor& output_tensor, IDevice* target_device, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

namespace operations::experimental::ccl {

std::vector<Tensor> recv_async(const Tensor& output_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

}  // namespace operations::experimental::ccl

}  // namespace ttnn
