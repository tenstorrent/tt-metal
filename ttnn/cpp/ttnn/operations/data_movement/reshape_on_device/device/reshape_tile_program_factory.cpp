// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_tile_program_factory.hpp"

#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/discovery/hal.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/utils/work_split.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/dataflow_buffer_spec.hpp>
#include <tt-metalium/workload/specification/kernel_spec.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/specification/tensor_parameter.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/math.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

ttnn::device_operation::ProgramArtifacts ReshapeTileProgramFactory::create_program_artifacts(
    const ttnn::prim::ReshapeOnDeviceParams& /*operation_attributes*/,
    const ttnn::prim::ReshapeOnDeviceInputs& tensor_args,
    ttnn::Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;

    const CoreRangeSet core_ranges{CoreRange{{0, 0}, {0, 0}}};
    const CoreCoord core{0, 0};

    tt::DataFormat dfb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(dfb_data_format);

    const MeshTensor& input_mesh_tensor = input_tensor.mesh_tensor();
    const MeshTensor& output_mesh_tensor = output_tensor.mesh_tensor();

    auto output_shape = output_tensor.padded_shape();

    // Tile count must come from the output tensor to match what the reader produces; deriving it
    // from the input deadlocks when the input and output have different tile padding.
    uint32_t num_tiles = output_tensor.physical_volume() / tt::constants::TILE_HW;

    constexpr uint32_t num_input_tiles = 2;

    const DFBSpecName IN0{"in0"};
    const DFBSpecName SCRATCH{"scratch"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    // Primary input DFB: holds tiles streamed from DRAM/L1 into the writer.
    Group<DataflowBufferSpec> dataflow_buffers = {DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = dfb_data_format,
    }};
    Group<DFBBinding> reader_dfb_bindings = {DFBBinding{
        .dfb_spec_name = IN0,
        .accessor_name = "in0",
        .endpoint_type = DFBEndpointType::PRODUCER,
    }};
    Table<std::string, std::string> reader_defines;

    bool src0_is_dram = input_tensor.memory_config().buffer_type() == BufferType::DRAM;
    uint32_t alignment = src0_is_dram ? hal::get_dram_alignment() : hal::get_l1_alignment();

    // Wide-alignment scratch DFB: only needed when DRAM/L1 alignment exceeds a face's
    // worth of bytes (the reader uses it as a temporary aligned landing pad).
    // Three things depend on this one condition. The DFB is declared, the reader binds it,
    // and the reader gets the WIDE_ALIGNMENT_SCRATCH define that the kernel tests.
    // The reader is the only kernel that touches this DFB. It binds both endpoints.
    if (alignment > (tt::constants::FACE_WIDTH * input_tensor.element_size())) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = SCRATCH,
            .entry_size = alignment,
            .num_entries = 1,
            .data_format_metadata = dfb_data_format,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = SCRATCH,
            .accessor_name = "intermed",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = SCRATCH,
            .accessor_name = "intermed",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        reader_defines["WIDE_ALIGNMENT_SCRATCH"] = "1";
    }

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
            "reader_unary_reshape_interleaved.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src0"},
            },
        .compile_time_args = {{"ALIGNMENT", alignment}, {"ELEMENT_SIZE", input_tensor.element_size()}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"input_Wt", "output_N", "output_C", "output_Ht", "output_Wt"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    KernelRunArgs reader_run_args{
        .kernel = READER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            core,
            {{"input_Wt", input_tensor.padded_shape()[3] / tt::constants::TILE_WIDTH},
             {"output_N", static_cast<uint32_t>(output_shape[0])},
             {"output_C", static_cast<uint32_t>(output_shape[1])},
             {"output_Ht", static_cast<uint32_t>(output_shape[2]) / tt::constants::TILE_HEIGHT},
             {"output_Wt", static_cast<uint32_t>(output_shape[3]) / tt::constants::TILE_WIDTH}}),
    };

    // Writer runtime args: total tile count and start_id (always 0 here, single-core).
    KernelRunArgs writer_run_args{
        .kernel = WRITER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(core, {{"num_pages", num_tiles}, {"start_id", 0u}}),
    };

    ProgramSpec spec{
        .name = "reshape_on_device_tile",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                // The input declares the same relaxations for the same reasons as in the row-major
                // factory (reshape_rm_program_factory.cpp). The output stays an exact match.
                TensorParameter{
                    .unique_id = INPUT,
                    .spec = input_tensor.tensor_spec(),
                    .relaxations = {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true},
                },
                TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER},
                    .target_nodes = core_ranges,
                },
            },
    };

    ProgramRunArgs run_params;
    run_params.kernel_run_args.push_back(std::move(reader_run_args));
    run_params.kernel_run_args.push_back(std::move(writer_run_args));
    run_params.tensor_args.emplace(INPUT, TensorArgument{input_mesh_tensor});
    run_params.tensor_args.emplace(OUTPUT, TensorArgument{output_mesh_tensor});

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
