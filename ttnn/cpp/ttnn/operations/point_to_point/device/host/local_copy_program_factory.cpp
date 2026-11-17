// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <filesystem>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include "point_to_point_device_op.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::point_to_point {

// Build a single-device local copy program: input_tensor -> output_tensor on one
// device, with no fabric. Reuses the op's generic unary reader/writer kernels through
// one dataflow buffer. Used only for the same-device (send_coord == receive_coord) case.
ttnn::device_operation::ProgramArtifacts local_copy_program_factory(
    const PointToPointOp::tensor_args_t& tensor_args, PointToPointOp::tensor_return_value_t& output_tensors) {
    const KernelSpecName reader_kernel{"reader"};
    const KernelSpecName writer_kernel{"writer"};
    const DFBSpecName pages_dfb{"pages"};
    const TensorParamName input_param{"input"};
    const TensorParamName output_param{"output"};

    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& output_tensor = output_tensors.at(1);  // final output; intermediate is unused here

    const uint32_t num_pages = data_movement::get_num_pages(input_tensor);
    const uint32_t page_size_bytes = input_tensor.tensor_spec().compute_page_size_bytes();
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t aligned_page_size_bytes = tt::round_up(page_size_bytes, l1_alignment);
    const tt::DataFormat data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    // Spread the pages across the device's worker grid so the copy is bounded by DRAM
    // bandwidth, not one core's NoC latency. Same kernels as the fabric path — just many
    // cores, each copying a disjoint contiguous page range.
    auto* mesh_device = dynamic_cast<MeshDevice*>(input_tensor.device());
    const CoreCoord compute_grid = mesh_device->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_grid, num_pages);

    // One DFB shared reader -> writer.
    constexpr uint32_t dfb_num_pages = 2;
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = pages_dfb,
            .entry_size = aligned_page_size_bytes,
            .num_entries = dfb_num_pages,
            .data_format_metadata = data_format,
        },
    };

    // Reader: input_tensor -> DFB.
    KernelSpec reader_kernel_spec{
        .unique_id = reader_kernel,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/"
                                        "reader_unary_interleaved_start_id_gen_metal2.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = pages_dfb,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = input_param,
                    .accessor_name = "src",
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id", "page_bytes"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // Writer: DFB -> output_tensor.
    KernelSpec writer_kernel_spec{
        .unique_id = writer_kernel,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_gen_metal2.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = pages_dfb,
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

    KernelRunArgs reader_run_args{.kernel = reader_kernel};
    KernelRunArgs writer_run_args{.kernel = writer_kernel};

    // Per core: a disjoint contiguous slice of pages.
    uint32_t start_id = 0;
    for (const CoreCoord& c : corerange_to_cores(all_cores, std::nullopt)) {
        uint32_t core_pages = 0;
        if (core_group_1.contains(c)) {
            core_pages = num_pages_per_core_group_1;
        } else if (core_group_2.contains(c)) {
            core_pages = num_pages_per_core_group_2;
        } else {
            continue;
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            c,
            {
                {"num_tiles", core_pages},
                {"start_id", start_id},
                {"page_bytes", page_size_bytes},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            c,
            {
                {"num_tiles", core_pages},
                {"start_id", start_id},
                {"page_bytes", page_size_bytes},
            });

        start_id += core_pages;
    }

    ProgramSpec spec{
        .name = "point_to_point_local_copy",
        .kernels = {std::move(reader_kernel_spec), std::move(writer_kernel_spec)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = input_param, .spec = input_tensor.tensor_spec()},
                TensorParameter{.unique_id = output_param, .spec = output_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "local_copy",
                    .kernels = {reader_kernel, writer_kernel},
                    .target_nodes = all_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args.emplace(input_param, input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(output_param, output_tensor.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::point_to_point
