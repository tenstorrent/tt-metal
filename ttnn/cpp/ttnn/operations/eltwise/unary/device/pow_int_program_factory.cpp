// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pow_int_device_operation.hpp"

#include <utility>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;
using ttnn::device_operation::ProgramArtifacts;

ProgramArtifacts PowIntDeviceOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const Tensor& input = tensor_args.input;

    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    const Tile& tile = input.tensor_spec().tile();
    const uint32_t single_tile_size = tile.get_tile_size(data_format);
    // A sharded allocation can contain padding pages in a partly filled final shard. TensorAccessor page ids are
    // logical tensor page ids, so splitting the physical buffer page count would read and write those padding pages
    // when the output has a different memory layout.
    const uint32_t num_tiles = input.physical_volume() / tile.get_tile_hw();

    const CoreCoord grid_size = input.device()->compute_with_storage_grid_size();
    [[maybe_unused]] const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
            split_work_to_cores(grid_size, num_tiles);

    const m2::DFBSpecName IN_DFB{"in"};
    const m2::DFBSpecName OUT_DFB{"out"};
    const m2::TensorParamName INPUT{"input"};
    const m2::TensorParamName OUTPUT{"output"};
    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    // Double-buffered so the reader/writer overlap with compute.
    constexpr uint32_t num_dfb_entries = 2;
    const m2::DataflowBufferSpec in_dfb{
        .unique_id = IN_DFB,
        .entry_size = single_tile_size,
        .num_entries = num_dfb_entries,
        .data_format_metadata = data_format,
        .tile_format_metadata = tile,
    };
    const m2::DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = num_dfb_entries,
        .data_format_metadata = data_format,
        .tile_format_metadata = tile,
    };

    // Shared eltwise/unary Metal 2.0 dataflow kernels; binding names are their interface.
    const m2::KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "reader_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    const m2::KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    // 32-bit integers are exact only in a 32-bit Dest, unpacked straight to Dest (SrcA would truncate).
    // UInt16 fits in SrcA and keeps the default unpack path; it is widened in Dest by the kernel.
    m2::ComputeHardwareConfig compute_hw{
        .fpu_math_fidelity = MathFidelity::HiFi4,
        .sfpu_precision_mode = Precision::Precise,
        .enable_32_bit_dest = true,
    };
    if (data_format != tt::DataFormat::UInt16) {
        compute_hw.unpack_modes.emplace(IN_DFB, UnpackMode::UnpackToDest);
    }

    const m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_pow_int.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .compile_time_args =
            {{"exponent", operation_attributes.exponent}, {"data_format", static_cast<uint32_t>(data_format)}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = std::move(compute_hw),
    };

    m2::KernelRunArgs reader_run_args{.kernel = READER};
    m2::KernelRunArgs writer_run_args{.kernel = WRITER};
    m2::KernelRunArgs compute_run_args{.kernel = COMPUTE};
    uint32_t start_id = 0;
    for (const CoreCoord& core : corerange_to_cores(all_cores)) {
        const uint32_t tiles_on_core =
            core_group_1.contains(core) ? num_tiles_per_core_group_1 : num_tiles_per_core_group_2;
        m2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"num_pages", tiles_on_core}, {"start_id", start_id}});
        m2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"num_pages", tiles_on_core}, {"start_id", start_id}});
        m2::AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"num_tiles", tiles_on_core}});
        start_id += tiles_on_core;
    }

    m2::ProgramSpec spec{
        .name = "pow_int",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters =
            {m2::TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
             m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = {m2::WorkUnitSpec{
            .name = "pow_int", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
