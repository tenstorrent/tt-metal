// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_h2d_op_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

// Returns the single MeshCoreCoord backing the H2D socket. Validation in the device
// operation guarantees that the socket has exactly one active core.
inline tt::tt_metal::distributed::MeshCoreCoord get_h2d_active_core(
    const tt::tt_metal::distributed::H2DSocket& h2d_socket) {
    const auto active_cores = h2d_socket.get_active_cores();
    TT_FATAL(
        active_cores.size() == 1,
        "recv_async_h2d: expected H2DSocket to have exactly one active core, found {}",
        active_cores.size());
    return active_cores.front();
}

}  // namespace

RecvAsyncH2DMeshWorkloadFactory::cached_mesh_workload_t RecvAsyncH2DMeshWorkloadFactory::create_mesh_workload(
    const RecvAsyncH2DParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const Tensor& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    // The H2D socket lives on exactly one mesh coordinate; only dispatch a program there,
    // and require that coordinate to be part of the output tensor's coordinate set.
    const auto active_core = get_h2d_active_core(*operation_attributes.h2d_socket);
    const auto& socket_device_coord = active_core.device_coord;

    const auto tensor_coords_flattened = tensor_coords.coords();
    TT_FATAL(
        std::find(tensor_coords_flattened.begin(), tensor_coords_flattened.end(), socket_device_coord) !=
            tensor_coords_flattened.end(),
        "recv_async_h2d: H2DSocket device coordinate is not part of the output tensor's coordinate set");

    auto cached_program = create_at(operation_attributes, socket_device_coord, tensor_args, tensor_return_value);
    workload.add_program(ttnn::MeshCoordinateRange(socket_device_coord), std::move(cached_program.program));
    shared_variables.emplace(ttnn::MeshCoordinateRange(socket_device_coord), cached_program.shared_variables);

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<RecvAsyncH2DMeshWorkloadFactory::shared_variables_t>
RecvAsyncH2DMeshWorkloadFactory::create_at(
    const RecvAsyncH2DParams& operation_attributes,
    const ttnn::MeshCoordinate& /*mesh_coordinate*/,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/) {
    const auto& h2d_socket = *operation_attributes.h2d_socket;
    const auto& output_tensor = tensor_args;

    const auto active_core = get_h2d_active_core(h2d_socket);
    const auto receiver_core_coord = active_core.core_coord;

    tt::tt_metal::Program program{};

    const uint32_t output_page_size = output_tensor.buffer()->aligned_page_size();
    const uint32_t num_pages = output_tensor.buffer()->num_pages();
    const bool pull_from_host = h2d_socket.get_h2d_mode() == tt::tt_metal::distributed::H2DMode::DEVICE_PULL;

    const auto receiver_core_range_set = CoreRangeSet({CoreRange(receiver_core_coord, receiver_core_coord)});

    const auto output_accessor_args = tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer());
    auto output_accessor_compile_time_args = output_accessor_args.get_compile_time_args();

    std::vector<uint32_t> writer_compile_args = {
        h2d_socket.get_config_buffer_address(),  // recv_socket_config_addr
        output_page_size,                        // page_size
        static_cast<uint32_t>(pull_from_host),   // pull_from_host
    };
    writer_compile_args.insert(
        writer_compile_args.end(), output_accessor_compile_time_args.begin(), output_accessor_compile_time_args.end());

    const auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async_h2d/device/kernels/"
        "h2d_receiver_writer.cpp",
        receiver_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    const std::vector<uint32_t> writer_rt_args = {
        output_tensor.buffer()->address(),  // output_base_addr
        num_pages,                          // num_pages
    };
    tt::tt_metal::SetRuntimeArgs(program, writer_kernel, receiver_core_coord, writer_rt_args);

    return {
        std::move(program),
        shared_variables_t{
            .receiver_core_coord = receiver_core_coord,
            .writer_kernel_id = writer_kernel,
        }};
}

void RecvAsyncH2DMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const RecvAsyncH2DParams& operation_attributes,
    const Tensor& tensor_args,
    [[maybe_unused]] std::vector<Tensor>& tensor_return_value) {
    const auto& output_tensor = tensor_args;
    const uint32_t num_pages = output_tensor.buffer()->num_pages();

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        auto& writer_runtime_args =
            GetRuntimeArgs(program, shared_vars.writer_kernel_id, shared_vars.receiver_core_coord);
        writer_runtime_args[0] = output_tensor.buffer()->address();
        writer_runtime_args[1] = num_pages;

        // H2D socket config buffer address is fixed for a given socket, so it does not
        // need to be patched here; it lives in compile-time args.
        (void)operation_attributes;
    }
}

}  // namespace ttnn::experimental::prim
