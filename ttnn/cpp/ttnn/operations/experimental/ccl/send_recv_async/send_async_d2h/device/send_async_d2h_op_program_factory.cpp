// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_d2h_op_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

// Returns the single MeshCoreCoord backing the D2H socket. Validation in the device
// operation guarantees that the socket has exactly one active core.
inline tt::tt_metal::distributed::MeshCoreCoord get_d2h_active_core(
    const tt::tt_metal::distributed::D2HSocket& d2h_socket) {
    const auto active_cores = d2h_socket.get_active_cores();
    TT_FATAL(
        active_cores.size() == 1,
        "send_async_d2h: expected D2HSocket to have exactly one active core, found {}",
        active_cores.size());
    return active_cores.front();
}

}  // namespace

SendAsyncD2HMeshWorkloadFactory::cached_mesh_workload_t SendAsyncD2HMeshWorkloadFactory::create_mesh_workload(
    const SendAsyncD2HParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const Tensor& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    // The D2H socket lives on exactly one mesh coordinate; only dispatch a program there,
    // and require that coordinate to be part of the input tensor's coordinate set.
    const auto active_core = get_d2h_active_core(*operation_attributes.d2h_socket);
    const auto& socket_device_coord = active_core.device_coord;

    const auto tensor_coords_flattened = tensor_coords.coords();
    TT_FATAL(
        std::find(tensor_coords_flattened.begin(), tensor_coords_flattened.end(), socket_device_coord) !=
            tensor_coords_flattened.end(),
        "send_async_d2h: D2HSocket device coordinate is not part of the input tensor's coordinate set");

    auto cached_program = create_at(operation_attributes, socket_device_coord, tensor_args, tensor_return_value);
    workload.add_program(ttnn::MeshCoordinateRange(socket_device_coord), std::move(cached_program.program));
    shared_variables.emplace(ttnn::MeshCoordinateRange(socket_device_coord), cached_program.shared_variables);

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<SendAsyncD2HMeshWorkloadFactory::shared_variables_t>
SendAsyncD2HMeshWorkloadFactory::create_at(
    const SendAsyncD2HParams& operation_attributes,
    const ttnn::MeshCoordinate& /*mesh_coordinate*/,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/) {
    const auto& d2h_socket = *operation_attributes.d2h_socket;
    const auto& input_tensor = tensor_args;

    const auto active_core = get_d2h_active_core(d2h_socket);
    const auto sender_core_coord = active_core.core_coord;

    tt::tt_metal::Program program{};

    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t num_pages = input_tensor.buffer()->num_pages();

    const auto sender_core_range_set = CoreRangeSet({CoreRange(sender_core_coord, sender_core_coord)});

    // Single-page L1 scratch CB used to stage each tensor page between the local NOC read
    // from the tensor and the PCIe NOC write to host pinned memory. We never consume from
    // the CB - we just grab its base address as a stable L1 staging region.
    const auto scratch_cb_index = tt::CBIndex::c_0;
    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig scratch_cb_config =
        tt::tt_metal::CircularBufferConfig(input_page_size, {{scratch_cb_index, data_format}})
            .set_page_size(scratch_cb_index, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_range_set, scratch_cb_config);

    const auto input_accessor_args = tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer());
    auto input_accessor_compile_time_args = input_accessor_args.get_compile_time_args();

    std::vector<uint32_t> reader_compile_args = {
        d2h_socket.get_config_buffer_address(),   // send_socket_config_addr
        input_page_size,                          // page_size
        static_cast<uint32_t>(scratch_cb_index),  // scratch_cb_id
    };
    reader_compile_args.insert(
        reader_compile_args.end(), input_accessor_compile_time_args.begin(), input_accessor_compile_time_args.end());

    const auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async_d2h/device/kernels/"
        "d2h_sender_reader.cpp",
        sender_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    const std::vector<uint32_t> reader_rt_args = {
        input_tensor.buffer()->address(),  // input_base_addr
        num_pages,                         // num_pages
    };
    tt::tt_metal::SetRuntimeArgs(program, reader_kernel, sender_core_coord, reader_rt_args);

    return {
        std::move(program),
        shared_variables_t{
            .sender_core_coord = sender_core_coord,
            .reader_kernel_id = reader_kernel,
        }};
}

void SendAsyncD2HMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const SendAsyncD2HParams& operation_attributes,
    const Tensor& tensor_args,
    [[maybe_unused]] std::vector<Tensor>& tensor_return_value) {
    const auto& input_tensor = tensor_args;
    const uint32_t num_pages = input_tensor.buffer()->num_pages();

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        auto& reader_runtime_args =
            GetRuntimeArgs(program, shared_vars.reader_kernel_id, shared_vars.sender_core_coord);
        reader_runtime_args[0] = input_tensor.buffer()->address();
        reader_runtime_args[1] = num_pages;

        // D2H socket config buffer address is fixed for a given socket, so it does not
        // need to be patched here; it lives in compile-time args.
        (void)operation_attributes;
    }
}

}  // namespace ttnn::experimental::prim
