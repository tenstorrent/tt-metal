// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/mesh_socket.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

struct SendAsync {
    const tt::tt_metal::distributed::MeshSocket& mesh_socket;

    SendAsync(const tt::tt_metal::distributed::MeshSocket& mesh_socket) : mesh_socket(mesh_socket) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        // TODO #25011: Adding mesh_socket requires reflection support including json serialization
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

tt::tt_metal::operation::ProgramWithCallbacks send_async_multicore(
    const Tensor& input_tensor,
    tt::tt_metal::IDevice* target_device,
    const tt::tt_metal::distributed::MeshSocket& mesh_socket);

namespace operations::experimental::ccl {

std::vector<Tensor> send_async(const Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

}  // namespace operations::experimental::ccl

}  // namespace ttnn
