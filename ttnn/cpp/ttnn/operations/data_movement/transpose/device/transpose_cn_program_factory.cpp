// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_cn_program_factory.hpp"
#include "transpose_utils.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TransposeCNProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Spec-scope resource names. The DFB accessor names are the tokens the reader and writer
    // kernels use (dfb::in0), so they are part of this factory's contract with its kernels.
    // Declared function-locally: this op's factories share one translation unit in the unity
    // build, so file-scope names would collide across them.
    const DFBSpecName IN0{"in0"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    const auto& input_tensor = tensor_args.input;
    const auto& input = input_tensor.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();
    auto input_shape = input_tensor.padded_shape();
    bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_cn needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_cn needs to be allocated in a buffer on device!");

    tt::DataFormat dfb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t page_shape[2] = {TILE_WIDTH, TILE_HEIGHT};
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        page_shape[0] = 1;
        page_shape[1] = input_shape[-1];
    }
    uint32_t page_size = page_shape[0] * page_shape[1];
    uint32_t stick_size = (row_major) ? page_shape[1] * input_tensor.element_size() : tt::tile_size(dfb_data_format);

    Buffer* src0_buffer = input_tensor.buffer();
    IDevice* device = input_tensor.device();

    uint32_t num_tensor_pages = input_tensor.physical_volume() / page_size;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_pages);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_input_pages = 2;

    ProgramSpec spec{.name = "transpose_cn"};

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = stick_size,
        .num_entries = num_input_pages,
        .data_format_metadata = dfb_data_format,
    });

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()});

    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (row_major) {
        reader_defines.insert({"CN_RM", "1"});
        writer_defines.insert({"CN_RM", "1"});
    }

    // `read_size` / `write_size` drive the sharded multi-page split helper on the row-major path;
    // `page_size` is the single-page NOC transfer size on the tile path. Both are emitted on both
    // paths, matching the legacy kernels' unconditional compile-time reads.
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                  "reader_unary_transpose_cn_interleaved_start_id.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .compile_time_args = {{"page_size", src0_buffer->aligned_page_size()}, {"read_size", stick_size}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"N", "C", "HtWt", "batch_step", "channel_step", "num_pages", "start_id", "hw", "n"}},
        .hw_config = create_reader_datamovement_config(device->arch()),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                  "writer_unary_transpose_cn_interleaved_start_id.cpp",
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .compile_time_args = {{"page_size", dst_buffer->aligned_page_size()}, {"write_size", stick_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = create_writer_datamovement_config(device->arch()),
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = all_cores,
    });

    // Set runtime arguments for each core
    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t Wt = W / page_shape[1];
    uint32_t Ht = H / page_shape[0];
    uint32_t HtWt = Ht * Wt;
    uint32_t CHtWt = C * HtWt;
    uint32_t NCHtWt = num_tensor_pages;
    uint32_t batch_step = CHtWt - HtWt;
    uint32_t channel_step = NCHtWt - HtWt;

    ProgramRunArgs run_args;
    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER};

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t num_pages_read = 0;
    for (const auto& core : cores) {
        uint32_t num_pages_per_core;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t hw = num_pages_read % HtWt;
        uint32_t curr_c = num_pages_read / HtWt;
        uint32_t n = curr_c % N;
        uint32_t start_tile = num_pages_read + (curr_c * batch_step) - (curr_c / N * channel_step);

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"N", N},
             {"C", C},
             {"HtWt", HtWt},
             {"batch_step", batch_step},
             {"channel_step", channel_step},
             {"num_pages", num_pages_per_core},
             {"start_id", start_tile},
             {"hw", hw},
             {"n", n}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_pages", num_pages_per_core}, {"start_id", num_pages_read}});

        num_pages_read += num_pages_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args.emplace(INPUT, input);
    run_args.tensor_args.emplace(OUTPUT, output);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
