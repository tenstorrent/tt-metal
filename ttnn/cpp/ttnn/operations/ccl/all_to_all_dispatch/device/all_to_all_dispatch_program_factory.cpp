// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <limits>

namespace ttnn::operations::ccl {

namespace detail {

uint32_t get_num_pages(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }

uint32_t get_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }

uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

uint32_t get_num_rows(const ttnn::Tensor& tensor) {
    auto logical_volume = tensor.logical_shape().volume();
    auto hidden_size = tensor.logical_shape()[-1];
    TT_FATAL(logical_volume % hidden_size == 0, "Logical volume must be divisible by hidden size");
    return logical_volume / hidden_size;
}

std::pair<std::array<uint32_t, 6>, std::array<uint32_t, 6>> get_cb_sizes(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& mapping_tensor,
    uint32_t num_links,
    std::optional<uint32_t> axis) {
    auto aligned_input_page_size = get_aligned_page_size(input_tensor);
    auto aligned_indices_page_size = get_aligned_page_size(indices_tensor);
    auto aligned_mapping_page_size = get_aligned_page_size(mapping_tensor);
    uint32_t tokens_per_device = get_num_rows(input_tensor);
    uint32_t tokens_per_core = tt::div_up(tokens_per_device, num_links);

    auto mapping_pages = get_num_pages(mapping_tensor);

    auto mesh_view = input_tensor.device()->get_view();
    uint32_t num_devices = mesh_view.num_devices();

    uint32_t dispatch_devices =
        axis.has_value() ? (axis.value() == 0 ? mesh_view.num_rows() : mesh_view.num_cols()) : num_devices;

    constexpr uint32_t buffering_factor = 2;
    constexpr uint32_t num_packet_headers = 2;

    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    std::array<uint32_t, 6> cb_sizes = {
        buffering_factor * aligned_input_page_size,
        tokens_per_core * aligned_indices_page_size,
        mapping_pages * aligned_mapping_page_size,
        num_devices * tokens_per_core * sizeof(uint8_t),
        tokens_per_device * dispatch_devices * aligned_indices_page_size,
        num_packet_headers * packet_header_size_bytes,
    };

    std::array<uint32_t, 6> cb_page_sizes = {
        aligned_input_page_size,
        aligned_indices_page_size,
        aligned_mapping_page_size,
        tokens_per_core * sizeof(uint8_t),
        aligned_indices_page_size,
        packet_header_size_bytes,
    };

    return {cb_sizes, cb_page_sizes};
}

}  // namespace detail

namespace {

// Build the ProgramDescriptor for one device's slice of the all-to-all-dispatch
// workload.  Per-coord variation is real here: src_chip_id, the linearized mesh
// index, the DIRECTIONS define (writer kernel) and the fabric-connection runtime
// args all depend on `mesh_coordinate`, so callers must build one descriptor
// per coord and cannot reuse a single descriptor across the mesh.
tt::tt_metal::ProgramDescriptor build_dispatch_program_descriptor(
    const AllToAllDispatchDeviceOperation::operation_attributes_t& operation_attributes,
    const AllToAllDispatchDeviceOperation::tensor_args_t& tensor_args,
    AllToAllDispatchDeviceOperation::tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tt::tt_metal::GlobalSemaphore& init_semaphore,
    const tt::tt_metal::GlobalSemaphore& cross_device_semaphore) {
    using namespace tt::tt_metal;
    using AllToAllTransferType = AllToAllDispatchDeviceOperation::AllToAllTransferType;

    ProgramDescriptor desc;

    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.expert_indices_tensor;
    auto mapping_tensor = tensor_args.expert_mapping_tensor;
    const auto& output_tensor = tensor_return_value.at(0);
    const auto& metadata_tensor = tensor_return_value.at(1);
    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    uint32_t src_mesh_id = *src_fabric_node_id.mesh_id;
    uint32_t src_chip_id = (uint32_t)src_fabric_node_id.chip_id;
    uint32_t linearized_mesh_coord = common::get_linearized_index(mesh_coordinate, mesh_view);

    log_debug(
        tt::LogOp,
        "\nCreating all to all dispatch program for mesh coordinate: ({}, {}) with mesh id: {} "
        "chip id: {} linearized mesh coord: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        src_mesh_id,
        src_chip_id,
        linearized_mesh_coord);

    const auto [neighbors, directions] =
        common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    auto input_shape = input_tensor.tensor_spec().logical_shape();
    auto indices_shape = indices_tensor.tensor_spec().logical_shape();
    auto mapping_shape = mapping_tensor.tensor_spec().logical_shape();

    uint32_t num_devices = mesh_view.num_devices();
    uint32_t dispatch_devices =
        operation_attributes.axis.has_value()
            ? operation_attributes.axis.value() == 0 ? mesh_view.num_rows() : mesh_view.num_cols()
            : mesh_view.num_devices();

    uint32_t hidden_size = input_shape[-1];
    uint32_t batch_size = input_shape[0] * dispatch_devices;

    uint32_t tokens_per_device = detail::get_num_rows(input_tensor);
    uint32_t selected_experts_k = indices_shape[-1];
    uint32_t experts = mapping_shape[-2];

    auto input_page_size = detail::get_page_size(input_tensor);
    auto indices_page_size = detail::get_page_size(indices_tensor);
    auto mapping_page_size = detail::get_page_size(mapping_tensor);
    auto output_page_size = detail::get_page_size(output_tensor);
    auto metadata_page_size = detail::get_page_size(metadata_tensor);

    auto input_pages = detail::get_num_pages(input_tensor);
    auto indices_pages = detail::get_num_pages(indices_tensor);
    auto mapping_pages = detail::get_num_pages(mapping_tensor);
    auto output_pages = detail::get_num_pages(output_tensor);
    auto metadata_pages = detail::get_num_pages(metadata_tensor);

    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(indices_tensor.dtype());
    auto mapping_data_format = tt::tt_metal::datatype_to_dataformat_converter(mapping_tensor.dtype());

    // input sharded buffer
    uint32_t input_tensor_cb_id = tt::CBIndex::c_0;
    // full indices buffer
    uint32_t indices_tensor_cb_id = tt::CBIndex::c_1;
    // full mapping buffer
    uint32_t mapping_tensor_cb_id = tt::CBIndex::c_2;
    // client interface
    uint32_t packet_header_cb_id = tt::CBIndex::c_3;
    // book-keeping buffer to avoid sending the same token multiple times
    uint32_t send_preparation_buffer_id = tt::CBIndex::c_4;
    // intermediate buffer for holding metadata before writing out to the device (for FullPacket impl)
    uint32_t metadata_buffer_id = tt::CBIndex::c_5;

    uint32_t aligned_input_page_size = detail::get_aligned_page_size(input_tensor);
    log_debug(
        tt::LogOp,
        "input shape: {}, input_pages: {}, input_page_size: {}, aligned_input_page_size: {}",
        input_tensor.logical_shape(),
        input_pages,
        input_page_size,
        aligned_input_page_size);

    uint32_t aligned_indices_page_size = detail::get_aligned_page_size(indices_tensor);
    log_debug(
        tt::LogOp,
        "indices shape: {}, indices_pages: {}, indices_page_size: {}, aligned_indices_page_size: {}",
        indices_tensor.logical_shape(),
        indices_pages,
        indices_page_size,
        aligned_indices_page_size);

    uint32_t aligned_mapping_page_size = detail::get_aligned_page_size(mapping_tensor);
    log_debug(
        tt::LogOp,
        "mapping shape: {}, mapping_pages: {}, mapping_page_size: {}, aligned_mapping_page_size: {}",
        mapping_tensor.logical_shape(),
        mapping_pages,
        mapping_page_size,
        aligned_mapping_page_size);

    uint32_t aligned_output_page_size = detail::get_aligned_page_size(output_tensor);
    log_debug(
        tt::LogOp,
        "output shape: {}, output_pages: {}, output_page_size: {}, aligned_output_page_size: {}",
        output_tensor.logical_shape(),
        output_pages,
        output_page_size,
        aligned_output_page_size);

    uint32_t aligned_metadata_page_size = detail::get_aligned_page_size(metadata_tensor);
    log_debug(
        tt::LogOp,
        "metadata shape: {}, metadata_pages: {}, metadata_page_size: {}, aligned_metadata_page_size: {}",
        metadata_tensor.logical_shape(),
        metadata_pages,
        metadata_page_size,
        aligned_metadata_page_size);

    auto [cb_sizes, cb_page_sizes] =
        detail::get_cb_sizes(input_tensor, indices_tensor, mapping_tensor, num_links, operation_attributes.axis);

    auto worker_core_range_set = operation_attributes.worker_core_range_set;

    auto subdevice_cores = corerange_to_cores(worker_core_range_set);
    TT_FATAL(
        subdevice_cores.size() >= num_links,
        "Not enough cores {} to send all links {}",
        subdevice_cores.size(),
        num_links);

    uint32_t tokens_per_core = tt::div_up(tokens_per_device, num_links);
    uint32_t num_cores = std::min(num_links, tt::div_up(tokens_per_device, tokens_per_core));
    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);

    // create circular buffers (descriptor style)
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_sizes[0],
        .core_ranges = sender_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_tensor_cb_id),
            .data_format = input_data_format,
            .page_size = cb_page_sizes[0],
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_sizes[1],
        .core_ranges = sender_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(indices_tensor_cb_id),
            .data_format = indices_data_format,
            .page_size = cb_page_sizes[1],
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_sizes[2],
        .core_ranges = sender_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(mapping_tensor_cb_id),
            .data_format = mapping_data_format,
            .page_size = cb_page_sizes[2],
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_sizes[5],
        .core_ranges = sender_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(packet_header_cb_id),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = cb_page_sizes[5],
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_sizes[3],
        .core_ranges = sender_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(send_preparation_buffer_id),
            .data_format = tt::DataFormat::UInt8,
            .page_size = cb_page_sizes[3],
        }}},
    });
    if (operation_attributes.impl == AllToAllTransferType::FullPacket) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb_sizes[4],
            .core_ranges = sender_core_grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(metadata_buffer_id),
                .data_format = mapping_data_format,
                .page_size = cb_page_sizes[4],
            }}},
        });
    }

    // Enumerate the mesh in row-major order to populate the DEST_CHIP_ID / DEST_MESH_ID
    // define arrays consumed by the writer kernel.  Mirrors the legacy use of
    // tensor_coords.coords(), which for these CCL ops covers the entire mesh.
    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord_fabric_node_id : mesh_view.get_fabric_node_ids()) {
        dest_mesh_id.push_back(*coord_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)coord_fabric_node_id.chip_id);
    }
    log_debug(tt::LogOp, "dest_chip_id: {}", common::stringify(dest_chip_id));
    log_debug(tt::LogOp, "dest_mesh_id: {}", common::stringify(dest_mesh_id));
    log_debug(tt::LogOp, "directions: {}", common::stringify(directions));

    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_id,
        indices_tensor_cb_id,
        mapping_tensor_cb_id,
        packet_header_cb_id,
        send_preparation_buffer_id,

        input_pages,
        indices_pages,
        mapping_pages,
        output_pages,
        metadata_pages,

        input_page_size,
        indices_page_size,
        mapping_page_size,
        output_page_size,
        metadata_page_size,

        num_devices,
        hidden_size,
        batch_size,
        selected_experts_k,
        experts,
        tokens_per_device,

        num_links,
        (uint32_t)topology,

        src_mesh_id,
        (uint32_t)src_chip_id,
        mesh_view.num_rows(),
        mesh_view.num_cols(),

        aligned_input_page_size,
        aligned_indices_page_size,
        aligned_mapping_page_size,
        aligned_output_page_size,
        aligned_metadata_page_size,

        (uint32_t)fabric_max_packet_size,

        l1_alignment,
        metadata_buffer_id,
        operation_attributes.impl == AllToAllTransferType::PageByPage ? 1u : 0u,
        linearized_mesh_coord,
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(mapping_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(metadata_tensor.buffer()).append_to(reader_compile_time_args);

    const auto& writer_compile_time_args = reader_compile_time_args;

    std::map<std::string, std::string> reader_defines = {
        {"AXIS", std::to_string(operation_attributes.axis.has_value() ? operation_attributes.axis.value() : -1)},
    };

    // Code-gen a mesh-position to fabric chip ID array for the writer kernel
    // Code-gen a mesh-position to mesh-id array for the writer kernel
    // Code-gen a direction array that is set to true when a direction has a valid connection (when a neighbor exists or
    // if it's along a valid cluster axis)
    std::map<std::string, std::string> writer_defines = {
        {"DEST_CHIP_ID", common::stringify(dest_chip_id)},
        {"DEST_MESH_ID", common::stringify(dest_mesh_id)},
        {"DIRECTIONS", common::stringify(directions)}};

    if (operation_attributes.axis.has_value()) {
        writer_defines["AXIS"] = std::to_string(operation_attributes.axis.value());
    }

    // Build kernel descriptors.  Push them onto desc.kernels NOW so we can refer to
    // them by stable index in the per-link runtime-args loop.
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/device/kernels/dataflow/reader_all_to_all_dispatch.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = sender_core_grid;
    reader_kernel_desc.compile_time_args = reader_compile_time_args;
    reader_kernel_desc.defines = {reader_defines.begin(), reader_defines.end()};
    reader_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_1,
    };

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/device/kernels/dataflow/writer_all_to_all_dispatch.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = sender_core_grid;
    writer_kernel_desc.compile_time_args = writer_compile_time_args;
    writer_kernel_desc.defines = {writer_defines.begin(), writer_defines.end()};
    writer_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_0,
    };

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    constexpr KernelHandle ternary_reader_kernel_id = 0;
    constexpr KernelHandle binary_writer_kernel_id = 1;

    uint32_t link_id = 0;
    uint32_t tokens_per_core_start = 0;
    for (const auto& sender_core : sender_cores) {
        const uint32_t token_range_start = tokens_per_core_start;
        const uint32_t token_range_end = std::min(tokens_per_core_start + tokens_per_core, tokens_per_device);
        tokens_per_core_start = token_range_end;

        // Reader runtime args: positions 0..4 are Buffer*'s for BufferBinding fast-path.
        KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.push_back(input_tensor.buffer());
        reader_rt_args.push_back(indices_tensor.buffer());
        reader_rt_args.push_back(mapping_tensor.buffer());
        reader_rt_args.push_back(output_tensor.buffer());
        reader_rt_args.push_back(metadata_tensor.buffer());
        reader_rt_args.push_back((uint32_t)cross_device_semaphore.address());
        reader_rt_args.push_back(token_range_start);
        reader_rt_args.push_back(token_range_end);
        desc.kernels[ternary_reader_kernel_id].emplace_runtime_args(sender_core, reader_rt_args);

        // The fabric helper appends to a plain std::vector<uint32_t>; build the writer
        // args there first, then promote them onto the kernel descriptor with positions
        // 0..4 swapped to Buffer*'s for BufferBinding.  Semaphore addresses remain plain
        // uint32_t (stable across cache hits).
        std::vector<uint32_t> writer_runtime_args = {
            input_tensor.buffer()->address(),     // placeholder
            indices_tensor.buffer()->address(),   // placeholder
            mapping_tensor.buffer()->address(),   // placeholder
            output_tensor.buffer()->address(),    // placeholder
            metadata_tensor.buffer()->address(),  // placeholder
            (uint32_t)cross_device_semaphore.address(),
            (uint32_t)init_semaphore.address(),
            token_range_start,
            token_range_end,
        };
        for (const auto& neighbor_coordinate : neighbors) {
            log_debug(
                tt::LogOp,
                "Connection between mesh coord ({}, {}) and ({}, {}) at core {} will choose link_id: {} and handles "
                "token indices from {} to {}",
                mesh_coordinate[0],
                mesh_coordinate[1],
                neighbor_coordinate[0],
                neighbor_coordinate[1],
                sender_core,
                link_id,
                token_range_start,
                token_range_end);
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                src_fabric_node_id,
                mesh_device->get_fabric_node_id(neighbor_coordinate),
                link_id,
                desc,
                sender_core,
                writer_runtime_args);
        }

        KernelDescriptor::RTArgList writer_rt_args_builder;
        writer_rt_args_builder.reserve(writer_runtime_args.size());
        writer_rt_args_builder.push_back(input_tensor.buffer());
        writer_rt_args_builder.push_back(indices_tensor.buffer());
        writer_rt_args_builder.push_back(mapping_tensor.buffer());
        writer_rt_args_builder.push_back(output_tensor.buffer());
        writer_rt_args_builder.push_back(metadata_tensor.buffer());
        for (size_t i = 5; i < writer_runtime_args.size(); ++i) {
            writer_rt_args_builder.push_back(writer_runtime_args[i]);
        }
        desc.kernels[binary_writer_kernel_id].emplace_runtime_args(sender_core, writer_rt_args_builder);
        link_id++;
    }

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor AllToAllDispatchDeviceOperation::AllToAllDispatchSparse::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    using namespace tt::tt_metal;

    // Workload-scoped resources: allocate the two GlobalSemaphores once per
    // cache miss and run the cross-device Synchronize barrier here so it's
    // amortised across every per-coord program build below.  Storing them on
    // WorkloadDescriptor::semaphores hands ownership to the program-cache so
    // they stay alive for the lifetime of the cached MeshWorkload.
    auto* mesh_device = tensor_args.input_tensor.device();
    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    auto final_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    tt::tt_metal::distributed::Synchronize(
        mesh_device, std::nullopt, {});  // interaction with subdevice needs to be investigated

    WorkloadDescriptor workload_descriptor;
    workload_descriptor.semaphores.push_back(init_barrier_semaphore);
    workload_descriptor.semaphores.push_back(final_barrier_semaphore);

    // Per-coord variation (src_chip_id, linearized mesh index, fabric-connection
    // runtime args, DIRECTIONS define) means we MUST build one ProgramDescriptor
    // per coord rather than reuse a single descriptor across the mesh.
    workload_descriptor.programs.reserve(tensor_coords.coords().size());
    for (const auto& coord : tensor_coords.coords()) {
        ProgramDescriptor desc = build_dispatch_program_descriptor(
            operation_attributes,
            tensor_args,
            tensor_return_value,
            coord,
            init_barrier_semaphore,
            final_barrier_semaphore);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return workload_descriptor;
}

}  // namespace ttnn::operations::ccl
