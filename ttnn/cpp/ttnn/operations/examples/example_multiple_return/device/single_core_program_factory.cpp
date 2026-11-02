// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_multiple_return_device_operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::operations::examples {

ttnn::device_operation::ProgramArtifacts
ExampleMultipleReturnDeviceOperation::ExampleMultipleReturnProgramFactory::create_program_artifacts(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName DFB_SRC0{"src0"};
    const DFBSpecName DFB_OUTPUT{"output"};
    const TensorParamName SRC{"src"};
    const TensorParamName DST1{"dst1"};
    const TensorParamName DST2{"dst2"};

    const auto& input_tensor = tensor_args.input_tensor.mesh_tensor();

    const auto& output_tensor1 = tensor_return_value.at(0);
    const auto& output_tensor2 = tensor_return_value.at(1);
    const bool has_output1 = output_tensor1.has_value();
    const bool has_output2 = output_tensor2.has_value();

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(dfb_data_format);

    auto output_dtype = output_tensor1.has_value() ? output_tensor1.value().dtype() : output_tensor2.value().dtype();
    tt::DataFormat dfb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_dtype);
    uint32_t single_tile_size_output = tt::tile_size(dfb_data_format_output);

    uint32_t num_tiles = input_tensor.physical_volume() / tt::constants::TILE_HW;

    CoreCoord compute_with_storage_grid_size = {1, 1};
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    ProgramSpec spec;
    spec.name = "example_multiple_return";

    // Tensors: the input is always bound; each output is bound only when the operation returns it.
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = SRC,
        .spec = input_tensor.tensor_spec(),
    });
    if (has_output1) {
        spec.tensor_parameters.push_back(TensorParameter{
            .unique_id = DST1,
            .spec = output_tensor1->mesh_tensor().tensor_spec(),
        });
    }
    if (has_output2) {
        spec.tensor_parameters.push_back(TensorParameter{
            .unique_id = DST2,
            .spec = output_tensor2->mesh_tensor().tensor_spec(),
        });
    }

    // Dataflow buffers: declared on the spec; placement follows from the kernel bindings below.
    constexpr uint32_t num_input_tiles = 2;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = DFB_SRC0,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = dfb_data_format,
    });

    constexpr uint32_t num_output_tiles = 2;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = DFB_OUTPUT,
        .entry_size = single_tile_size_output,
        .num_entries = num_output_tiles,
        .data_format_metadata = dfb_data_format_output,
    });

    // Kernels: the spec points at the existing kernel sources, no duplication needed.
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "reader_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DFB_SRC0,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = SRC,
                    .accessor_name = "src",
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // The writer drains the output buffer to each returned output. An output that is not returned
    // has no tensor to bind, so its binding and the matching define are both omitted, and the
    // kernel compiles out the write to it.
    KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/examples/example_multiple_return/device/kernels/writer_multiple.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DFB_OUTPUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    if (has_output1) {
        writer.compiler_options.defines.emplace("RETURN_OUTPUT1", "1");
        writer.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = DST1,
            .accessor_name = "dst1",
        });
    }
    if (has_output2) {
        writer.compiler_options.defines.emplace("RETURN_OUTPUT2", "1");
        writer.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = DST2,
            .accessor_name = "dst2",
        });
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu_metal2.cpp",
        // Legacy compute kernels default to -O3; Metal 2.0 defaults every kernel to -O2.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DFB_SRC0,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_OUTPUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config =
            ComputeHardwareConfig{
                .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                .sfpu_precision_mode = Precision::Precise,  // math_approx_mode = false
            },
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        // Tensor addresses are not runtime args: they flow through the tensor bindings above, which
        // the framework refreshes on every cache hit.
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_pages", num_tiles_per_core}, {"start_id", num_tiles_written}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_tiles", num_tiles_per_core}, {"start_id", num_tiles_written}});

        // Tile counts derive from the input spec, which is part of the program hash, so a cache hit
        // guarantees they are unchanged and nothing has to re-apply them.
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"num_tiles", num_tiles_per_core}});

        num_tiles_written += num_tiles_per_core;
    }

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(std::move(compute));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "example_multiple_return",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_cores,
    });

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        std::move(reader_run_args),
        std::move(writer_run_args),
        std::move(compute_run_args),
    };
    run_args.tensor_args.emplace(SRC, input_tensor);
    if (has_output1) {
        run_args.tensor_args.emplace(DST1, output_tensor1->mesh_tensor());
    }
    if (has_output2) {
        run_args.tensor_args.emplace(DST2, output_tensor2->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::operations::examples
