// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "outbound_socket_service_sync_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

ProgramDescriptor OutboundSocketServiceSyncProgramFactory::create_descriptor(
    const OutboundSocketServiceSyncParams& args,
    const OutboundSocketServiceSyncInputs& tensor_args,
    Tensor& backing_out,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "outbound_socket_service_sync: the program factory requires a per-device mesh dispatch coordinate");
    const auto& coord = *mesh_dispatch_coordinate;
    const uint32_t idx = coord[0] * args.mesh_num_cols + coord[1];
    TT_FATAL(idx < args.data_ready_addrs.size(), "outbound_socket_service_sync: mesh coordinate {} out of range", idx);

    const auto& input = tensor_args.input;
    const bool has_metadata = args.metadata_size_bytes > 0;

    // Per-coord service-core targeting (logical -> physical NoC coord).
    auto* mesh_device = backing_out.device();
    const CoreCoord service_logical(args.service_core_x[idx], args.service_core_y[idx]);
    const CoreCoord service_phys = mesh_device->worker_core_from_logical_core(service_logical);
    const uint32_t data_ready_addr = args.data_ready_addrs[idx];
    const uint32_t metadata_l1_addr = has_metadata ? args.metadata_addrs[idx] : 0;

    const uint32_t page_size = args.page_size;
    const uint32_t num_pages = args.num_pages;
    const CoreRangeSet worker_crs(args.worker_cores);

    // Enumerate worker cores row-major (y outer, x inner) -- must match the service's
    // worker enumeration so per-worker page slices stay stable.
    std::vector<CoreCoord> workers;
    for (uint32_t y = args.worker_cores.start_coord.y; y <= args.worker_cores.end_coord.y; ++y) {
        for (uint32_t x = args.worker_cores.start_coord.x; x <= args.worker_cores.end_coord.x; ++x) {
            workers.emplace_back(x, y);
        }
    }
    const uint32_t num_workers = static_cast<uint32_t>(workers.size());
    TT_FATAL(num_workers > 0, "outbound_socket_service_sync: worker_cores must contain at least one core");

    auto* input_buffer = input.buffer();
    auto* backing_buffer = backing_out.buffer();
    Buffer* metadata_buffer = has_metadata ? tensor_args.metadata->buffer() : nullptr;

    ProgramDescriptor desc;

    // Single-slot scratch CB, sized to one page, used as per-page staging L1.
    desc.cbs.push_back(CBDescriptor{
        .total_size = page_size,
        .core_ranges = worker_crs,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(args.scratch_cb_index),
            .data_format = datatype_to_dataformat_converter(input.dtype()),
            .page_size = page_size,
        }}},
    });

    // CT args (uniform across workers). Base addresses are NOT here -- they are runtime
    // BufferBindings. input and backing share the spec, so ONE accessor block describes
    // both; the metadata accessor (if any) is packed right after it.
    std::vector<uint32_t> ct_args = {
        page_size,
        args.scratch_cb_index,
        args.metadata_size_bytes,
        static_cast<uint32_t>(args.metadata_only ? 1u : 0u),
    };
    TensorAccessorArgs(input_buffer).append_to(ct_args);
    if (has_metadata) {
        TensorAccessorArgs(metadata_buffer).append_to(ct_args);
    }

    KernelDescriptor writer;
    writer.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/outbound_socket_service_sync/device/kernels/"
        "outbound_socket_service_sync_writer.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = worker_crs;
    writer.compile_time_args = std::move(ct_args);
    writer.config = WriterConfigDescriptor{};

    // Distribute pages across workers; the leading workers absorb the remainder so no
    // page is dropped.
    const uint32_t base = num_pages / num_workers;
    const uint32_t remainder = num_pages % num_workers;
    uint32_t cursor = 0;
    for (uint32_t i = 0; i < num_workers; ++i) {
        const uint32_t n = base + (i < remainder ? 1u : 0u);
        const uint32_t start_page = cursor;
        const uint32_t end_page = cursor + n;
        cursor += n;

        // Buffer* entries auto-register as BufferBindings (patched on cache hits); the
        // rest are plain uint32 scalars. Order MUST match the kernel's get_arg_val
        // indices.
        KernelDescriptor::RTArgList rt_args;
        rt_args.push_back(input_buffer);                           // 0: input base (patched per dispatch)
        rt_args.push_back(backing_buffer);                         // 1: sender backing base
        rt_args.push_back(data_ready_addr);                        // 2: service-core data_ready counter
        rt_args.push_back(static_cast<uint32_t>(service_phys.x));  // 3: service core NoC x
        rt_args.push_back(static_cast<uint32_t>(service_phys.y));  // 4: service core NoC y
        rt_args.push_back(start_page);                             // 5
        rt_args.push_back(end_page);                               // 6
        if (has_metadata) {
            rt_args.push_back(metadata_buffer);   // 7: metadata input base
            rt_args.push_back(metadata_l1_addr);  // 8: service-core metadata L1
        }
        writer.emplace_runtime_args(workers[i], rt_args);
    }

    desc.kernels.push_back(std::move(writer));
    return desc;
}

}  // namespace ttnn::experimental::prim
