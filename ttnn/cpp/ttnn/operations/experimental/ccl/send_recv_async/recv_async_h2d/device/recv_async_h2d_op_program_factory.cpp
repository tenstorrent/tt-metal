// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_h2d_op_device_operation.hpp"

#include <cstdint>
#include <optional>
#include <utility>
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

tt::tt_metal::ProgramDescriptor RecvAsyncH2DDeviceOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    using namespace tt::tt_metal;

    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "recv_async_h2d: the program factory requires a per-device mesh dispatch coordinate");

    const auto& h2d_socket = *operation_attributes.h2d_socket;
    const auto& output_tensor = tensor_args;

    // The H2D socket lives on exactly one mesh coordinate; only that coordinate gets a program.
    // Validation requires that coordinate to be part of the output tensor's coordinate set.
    const auto active_core = get_h2d_active_core(h2d_socket);
    if (*mesh_dispatch_coordinate != active_core.device_coord) {
        return ProgramDescriptor{};
    }
    const auto receiver_core_coord = active_core.core_coord;

    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(output_buffer != nullptr, "recv_async_h2d: output tensor buffer is null");

    const uint32_t output_page_size = output_buffer->aligned_page_size();
    const uint32_t num_pages = output_buffer->num_pages();
    const bool pull_from_host = h2d_socket.get_h2d_mode() == tt::tt_metal::distributed::H2DMode::DEVICE_PULL;

    const auto receiver_core_range_set = CoreRangeSet({CoreRange(receiver_core_coord, receiver_core_coord)});

    const auto output_accessor_args = tt::tt_metal::TensorAccessorArgs(*output_buffer);
    auto output_accessor_compile_time_args = output_accessor_args.get_compile_time_args();

    // The socket config buffer address is a compile-time arg and part of the program hash
    // (see compute_program_hash), so it is structural and never needs patching on a cache hit.
    std::vector<uint32_t> writer_compile_args = {
        h2d_socket.get_config_buffer_address(),  // recv_socket_config_addr
        output_page_size,                        // page_size
        static_cast<uint32_t>(pull_from_host),   // pull_from_host
    };
    writer_compile_args.insert(
        writer_compile_args.end(), output_accessor_compile_time_args.begin(), output_accessor_compile_time_args.end());

    KernelDescriptor writer;
    writer.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async_h2d/device/kernels/"
        "h2d_receiver_writer.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = receiver_core_range_set;
    writer.compile_time_args = std::move(writer_compile_args);
    writer.config = WriterConfigDescriptor{};

    // The output address is a Buffer* binding patched by the framework on every cache hit;
    // num_pages derives from the tensor spec, which is in the hash.
    writer.emplace_runtime_args(
        receiver_core_coord,
        {
            output_buffer,  // output_base_addr
            num_pages,      // num_pages
        });

    ProgramDescriptor desc;
    desc.kernels.push_back(std::move(writer));
    return desc;
}

}  // namespace ttnn::experimental::prim
