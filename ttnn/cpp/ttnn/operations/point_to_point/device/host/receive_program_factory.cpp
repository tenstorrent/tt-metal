// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <filesystem>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "point_to_point_device_op.hpp"

namespace ttnn::operations::point_to_point {

ttnn::device_operation::ProgramArtifacts receive_program_factory(
    const PointToPointOp::operation_attributes_t& operation_attributes,
    PointToPointOp::tensor_return_value_t& output_tensors,
    const tt::tt_metal::GlobalSemaphore& semaphore) {
    using namespace tt::tt_metal;

    const KernelSpecName reader_kernel{"reader"};
    const KernelSpecName writer_kernel{"writer"};
    const ScratchpadSpecName packet_header_scratchpad{"packet_header"};
    const ScratchpadSpecName packet_scratchpad{"packet"};
    const DFBSpecName receiver_dfb{"receiver"};
    const TensorParamName intermediate_param{"intermediate"};
    const TensorParamName output_param{"output"};
    const experimental::GlobalSemaphoreParamName sender_semaphore_param{"sender_semaphore"};
    const experimental::FabricConnectionSpecName connection_spec{"connection"};

    auto* mesh_device = dynamic_cast<MeshDevice*>(output_tensors.at(0).device());

    const auto& send_coord = operation_attributes.send_coord;
    const auto& receive_coord = operation_attributes.receive_coord;
    const auto& intermediate_tensor = output_tensors.at(0);
    const auto& output_tensor = output_tensors.at(1);

    // basic accounting
    const uint32_t output_num_pages = data_movement::get_num_pages(output_tensor);
    const uint32_t output_page_size_bytes = output_tensor.tensor_spec().compute_page_size_bytes();
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    // figure out packets
    const auto [packet_size_bytes, num_pages_per_packet, num_page_segments, total_packets] =
        detail::compute_aligned_packet_dims(
            output_tensor.dtype(), output_page_size_bytes, output_num_pages, l1_alignment);
    // distribute work
    const CoreCoord use_cores = {1, 1};

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_packets_per_core_group_1, num_packets_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(use_cores, total_packets);

    tt::DataFormat inter_dataformat = tt::tt_metal::datatype_to_dataformat_converter(intermediate_tensor.dtype());

    // Scratchpad for packet headers
    constexpr auto buffering_factor = 2;  // this is in other fabric kernels
    constexpr auto num_packet_headers_storable = 2;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    // DFB for sender reader->writer kernels
    const uint32_t dfb_num_pages = 3 * num_pages_per_packet;

    Group<ScratchpadSpec> scratchpads = {
        ScratchpadSpec{
            .unique_id = packet_header_scratchpad,
            .size_per_node = packet_header_size_bytes * (num_packet_headers_storable * buffering_factor),
        },
        // Scratchpad for loading up pages that are collected into packets
        ScratchpadSpec{
            .unique_id = packet_scratchpad,
            .size_per_node = packet_size_bytes * 1,
        },
    };

    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = receiver_dfb,
            .entry_size = output_page_size_bytes,
            .num_entries = dfb_num_pages,
            .data_format_metadata = inter_dataformat,
        },
    };

    const auto& topology = operation_attributes.topology;
    const auto this_fabric_id = mesh_device->get_fabric_node_id(receive_coord);
    const auto [num_hops, sender_is_forward, next_fabric_id] =
        detail::fabric_1d_routing(mesh_device, receive_coord, send_coord, topology);

    KernelSpec reader_kernel_spec{
        .unique_id = reader_kernel,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_receive_metal2.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = receiver_dfb,
                    .accessor_name = "receiver",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .scratchpad_bindings =
            {
                ScratchpadBinding{
                    .scratchpad_spec_name = packet_header_scratchpad,
                    .accessor_name = "packet_header",
                },
                ScratchpadBinding{
                    .scratchpad_spec_name = packet_scratchpad,
                    .accessor_name = "packet",
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = intermediate_param,
                    .accessor_name = "packet_buffer",
                },
            },
        .compile_time_args = {{"alignment", l1_alignment}},
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {
                        "page_idx_start",
                        "page_idx_end",
                        "max_pages_per_packet",
                        "packet_size_bytes",
                        "page_size_bytes",
                        "page_segments",
                        "sender_num_hops",
                    },
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    reader_kernel_spec.advanced_options.global_semaphore_bindings = {
        {
            .global_semaphore_parameter_name = sender_semaphore_param,
            .accessor_name = "sender_semaphore",
        },
    };
    reader_kernel_spec.advanced_options.fabric_connection_bindings = {
        {
            .fabric_connection_spec_name = connection_spec,
            .accessor_name = "connection",
        },
    };

    // And the writer
    KernelSpec writer_kernel_spec{
        .unique_id = writer_kernel,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_gen_metal2.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = receiver_dfb,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = output_param,
                    .accessor_name = "dst",
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id", "page_bytes"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    constexpr auto link_idx = 0;  // for single link implementation

    experimental::FabricConnectionSpec fabric_connection{
        .unique_id = connection_spec,
        .src_fabric_node_id = this_fabric_id,
    };

    KernelRunArgs reader_run_args{.kernel = reader_kernel};
    KernelRunArgs writer_run_args{.kernel = writer_kernel};

    uint32_t page_idx_start = 0, page_idx_end = 0;
    for (auto c : corerange_to_cores(all_cores, std::nullopt)) {
        uint32_t increment = 0;
        if (core_group_1.contains(c)) {
            increment = num_packets_per_core_group_1 * num_pages_per_packet;
        } else if (core_group_2.contains(c)) {
            increment = num_packets_per_core_group_2 * num_pages_per_packet;
        } else {
            continue;
        }
        increment = std::min(increment, output_num_pages - page_idx_start);
        page_idx_end += increment;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            c,
            {
                {"page_idx_start", page_idx_start},
                {"page_idx_end", page_idx_end},
                {"max_pages_per_packet", num_pages_per_packet},
                {"packet_size_bytes", packet_size_bytes},
                {"page_size_bytes", output_page_size_bytes},
                {"page_segments", num_page_segments},
                {"sender_num_hops", num_hops},
            });

        fabric_connection.targets[c] = {
            experimental::FabricConnectionTarget{
                .dst_fabric_node_id = next_fabric_id,
                .link_idx = link_idx,
            },
        };

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            c,
            {
                {"num_tiles", increment},
                {"start_id", page_idx_start},
                {"page_bytes", output_page_size_bytes},
            });

        page_idx_start += increment;
    }

    ProgramSpec spec{
        .name = "point_to_point_receive",
        .kernels = {std::move(reader_kernel_spec), std::move(writer_kernel_spec)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .scratchpads = std::move(scratchpads),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = intermediate_param, .spec = intermediate_tensor.tensor_spec()},
                TensorParameter{.unique_id = output_param, .spec = output_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "receive",
                    .kernels = {reader_kernel, writer_kernel},
                    .target_nodes = all_cores,
                },
            },
    };
    spec.advanced_options.global_semaphore_parameters = {
        {.unique_id = sender_semaphore_param, .memory_type = BufferType::L1},
    };
    spec.advanced_options.fabric_connections = {std::move(fabric_connection)};

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args.emplace(intermediate_param, intermediate_tensor.mesh_tensor());
    run_args.tensor_args.emplace(output_param, output_tensor.mesh_tensor());
    run_args.advanced_options.global_semaphore_args.emplace(sender_semaphore_param, std::cref(semaphore));

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}
}  // namespace ttnn::operations::point_to_point
