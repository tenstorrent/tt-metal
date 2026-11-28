// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "isin_program_factory.hpp"

#include "../isin_common.hpp"

#include <tt-metalium/memory/buffer/buffer.hpp>
#include <tt-metalium/utils/work_split.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

static DataflowBufferSpec make_dfb(
    const DFBSpecName& dfb_name, const DataType& dtype, const uint32_t& page_size_bytes) {
    return DataflowBufferSpec{
        .unique_id = dfb_name,
        .entry_size = page_size_bytes,
        .num_entries = 1,
        .data_format_metadata = datatype_to_dataformat_converter(dtype),
    };
}

ttnn::device_operation::ProgramArtifacts IsInProgramFactory::create_program_artifacts(
    const IsinParams& args, const IsinInputs& tensor_args, Tensor& output_tensor) {
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName ELEMENTS_DFB{"elements"};
    const DFBSpecName TEST_ELEMENTS_DFB{"test_elements"};
    const DFBSpecName OUTPUT_DFB{"output"};
    const TensorParamName ELEMENTS{"elements"};
    const TensorParamName TEST_ELEMENTS{"test_elements"};
    const TensorParamName OUTPUT{"output"};

    const auto& elements_tensor = tensor_args.elements_tensor;
    const auto& test_elements_tensor = tensor_args.test_elements_tensor;

    const auto& elements_dtype = elements_tensor.dtype();
    const auto& test_elements_dtype = test_elements_tensor.dtype();

    const bool& invert = args.invert;
    const uint32_t& single_fetch_subchunk_size = args.single_fetch_subchunk_size;

    auto* elements_buffer = elements_tensor.buffer();
    auto* test_elements_buffer = test_elements_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(elements_buffer != nullptr, "Elements tensor's buffer is null");
    TT_FATAL(test_elements_buffer != nullptr, "Test elements tensor's buffer is null");
    TT_FATAL(output_buffer != nullptr, "Output tensor's buffer is null");

    const auto& elements_mesh_tensor = elements_tensor.mesh_tensor();
    const auto& test_elements_mesh_tensor = test_elements_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();

    // input dtype byte sizes
    const uint32_t& elements_datum_size = elements_tensor.element_size();
    const uint32_t& test_elements_datum_size = test_elements_tensor.element_size();
    const uint32_t& output_datum_size = output_tensor.element_size();

    // input row byte sizes
    const uint32_t& elements_subchunk_size_bytes = single_fetch_subchunk_size * elements_datum_size;
    const uint32_t& test_elements_subchunk_size_bytes = single_fetch_subchunk_size * test_elements_datum_size;
    const uint32_t& output_subchunk_size_bytes = single_fetch_subchunk_size * output_datum_size;

    auto* device = elements_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // Both kernels get the full set; the shared decoder in isin_common.hpp reads all of them.
    const KernelSpec::CompileTimeArgs compile_time_args{
        {"elements_size", elements_tensor.logical_volume()},
        {"test_elements_size", test_elements_tensor.logical_volume()},
        {"single_fetch_subchunk_size", single_fetch_subchunk_size},
        {"invert", static_cast<uint32_t>(invert)},
        {"elements_tensor_datum_size", elements_tensor.element_size()},
        {"output_tensor_datum_size", output_datum_size},
    };

    // The final tensor to be dealt with by the isin device operation is flattened to 1D and the number of subchunks
    // (each of elements, test_elements and output) that they have to be split into depends on the L1 available size per
    // core (to hold elements, test_elements and output subchunks at the same time). The number of cores utilized is at
    // least the number of subchunks work has been split into.
    const uint32_t subchunks_num =
        (elements_tensor.logical_volume() + single_fetch_subchunk_size - 1) / single_fetch_subchunk_size;
    const auto
        [num_cores,                       // number of cores utilized
         all_cores,                       // set of all cores used
         core_group_1,                    // Primary core group
         core_group_2,                    // Secondary core group
         num_subchunks_per_core_group_1,  // Number of subchunks each core in the primary group processes
         num_subchunks_per_core_group_2   // Number of subchunks each core in the secondary group processes
    ] = split_work_to_cores(compute_with_storage_grid_size, subchunks_num);

    // Each DFB holds one subchunk. The reader loads elements and test_elements into their DFBs and reads them back
    // itself, so it is both producer and consumer of those two.
    ProgramSpec spec{
        .name = "isin",
        .dataflow_buffers =
            {
                make_dfb(ELEMENTS_DFB, elements_dtype, elements_subchunk_size_bytes),
                make_dfb(TEST_ELEMENTS_DFB, test_elements_dtype, test_elements_subchunk_size_bytes),
                make_dfb(OUTPUT_DFB, OUTPUT_TENSOR_DATA_TYPE, output_subchunk_size_bytes),
            },
        .tensor_parameters =
            {
                TensorParameter{.unique_id = ELEMENTS, .spec = elements_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = TEST_ELEMENTS, .spec = test_elements_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "isin",
                    .kernels = {READER, WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    constexpr const char* READER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/experimental/isin/device/kernels/dataflow/isin_reader.cpp";
    constexpr const char* WRITER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/experimental/isin/device/kernels/dataflow/isin_writer.cpp";

    KernelSpec reader_kernel{
        .unique_id = READER,
        .source = READER_KERNEL_PATH,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = ELEMENTS_DFB,
                    .accessor_name = "elements",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = ELEMENTS_DFB,
                    .accessor_name = "elements",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = TEST_ELEMENTS_DFB,
                    .accessor_name = "test_elements",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = TEST_ELEMENTS_DFB,
                    .accessor_name = "test_elements",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_DFB,
                    .accessor_name = "output",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = ELEMENTS, .accessor_name = "elements"},
                TensorBinding{.tensor_parameter_name = TEST_ELEMENTS, .accessor_name = "test_elements"},
            },
        .compile_time_args = compile_time_args,
        .runtime_arg_schema = {.runtime_arg_names = {"subchunks_per_core", "subchunks_offset"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer_kernel{
        .unique_id = WRITER,
        .source = WRITER_KERNEL_PATH,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUTPUT_DFB,
                    .accessor_name = "output",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
            },
        .compile_time_args = compile_time_args,
        .runtime_arg_schema = {.runtime_arg_names = {"subchunks_per_core", "subchunks_offset"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const auto cores = grid_to_cores(num_cores, compute_with_storage_grid_size.x, num_cores_y);
    uint32_t subchunks_offset = 0;
    std::for_each(cores.cbegin(), cores.cend(), [&](const CoreCoord& core) {
        uint32_t subchunks_per_core = 0;
        if (core_group_1.contains(core)) {
            subchunks_per_core = num_subchunks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            subchunks_per_core = num_subchunks_per_core_group_2;
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"subchunks_per_core", subchunks_per_core}, {"subchunks_offset", subchunks_offset}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"subchunks_per_core", subchunks_per_core}, {"subchunks_offset", subchunks_offset}});

        subchunks_offset += subchunks_per_core;
    });

    spec.kernels = {std::move(reader_kernel), std::move(writer_kernel)};

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)},
        .tensor_args =
            {
                {ELEMENTS, elements_mesh_tensor},
                {TEST_ELEMENTS, test_elements_mesh_tensor},
                {OUTPUT, output_mesh_tensor},
            },
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}
}  // namespace ttnn::experimental::prim
