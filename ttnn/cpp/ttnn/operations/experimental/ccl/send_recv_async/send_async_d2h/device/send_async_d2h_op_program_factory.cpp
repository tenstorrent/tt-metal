// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_d2h_op_device_operation.hpp"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
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

tt::tt_metal::ProgramDescriptor SendAsyncD2HDeviceOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    using namespace tt::tt_metal;

    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "send_async_d2h: the program factory requires a per-device mesh dispatch coordinate");

    const auto& d2h_socket = *operation_attributes.d2h_socket;
    const auto& input_tensor = tensor_args;

    // The D2H socket lives on exactly one mesh coordinate; only that coordinate gets a program.
    // Validation requires that coordinate to be part of the input tensor's coordinate set.
    // The active core (device and core coordinate) is part of the program hash.
    const auto active_core = get_d2h_active_core(d2h_socket);
    if (*mesh_dispatch_coordinate != active_core.device_coord) {
        return ProgramDescriptor{};
    }
    const auto sender_core_coord = active_core.core_coord;

    auto* input_buffer = input_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "send_async_d2h: input tensor buffer is null");

    const uint32_t input_page_size = input_buffer->aligned_page_size();
    const uint32_t num_pages = input_buffer->num_pages();

    const auto sender_core_range_set = CoreRangeSet({CoreRange(sender_core_coord, sender_core_coord)});

    ProgramDescriptor desc;

    // Single-page L1 scratch CB used to stage each tensor page between the local NOC read
    // from the tensor and the PCIe NOC write to host pinned memory. We never consume from
    // the CB - we just grab its base address as a stable L1 staging region.
    constexpr uint8_t scratch_cb_index = tt::CBIndex::c_0;
    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_page_size,
        .core_ranges = sender_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = scratch_cb_index,
            .data_format = data_format,
            .page_size = input_page_size,
        }}},
    });

    const auto input_accessor_args = tt::tt_metal::TensorAccessorArgs(*input_buffer);
    auto input_accessor_compile_time_args = input_accessor_args.get_compile_time_args();

    // The socket config buffer address is a compile-time arg and part of the program hash
    // (see compute_program_hash), so it is structural and never needs patching on a cache hit.
    std::vector<uint32_t> reader_compile_args = {
        d2h_socket.get_config_buffer_address(),   // send_socket_config_addr
        input_page_size,                          // page_size
        static_cast<uint32_t>(scratch_cb_index),  // scratch_cb_id
    };
    reader_compile_args.insert(
        reader_compile_args.end(), input_accessor_compile_time_args.begin(), input_accessor_compile_time_args.end());

    KernelDescriptor reader;
    reader.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async_d2h/device/kernels/"
        "d2h_sender_reader.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = sender_core_range_set;
    reader.compile_time_args = std::move(reader_compile_args);
    reader.named_compile_time_args = {{"scratch_cb_id", scratch_cb_index}};
    reader.config = ReaderConfigDescriptor{};

    // The input address is a Buffer* binding patched by the framework on every cache hit;
    // num_pages derives from the tensor spec, which is in the hash.
    reader.emplace_runtime_args(
        sender_core_coord,
        {
            input_buffer,  // input_base_addr
            num_pages,     // num_pages
        });

    desc.kernels.push_back(std::move(reader));
    return desc;
}

}  // namespace ttnn::experimental::prim
