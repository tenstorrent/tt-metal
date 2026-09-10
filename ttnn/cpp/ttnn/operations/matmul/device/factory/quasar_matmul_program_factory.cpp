// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_program_factory.hpp"
#include <map>
#include <string>

inline constexpr const char* IN0_DFB = "in0";
inline constexpr const char* IN1_DFB = "in1";
inline constexpr const char* OUT_DFB = "out";

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental::metal2_host_api;
static cached_program_t create(
    const MatmulParams& operation_attributes, const MatmulInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensors[0];
    const auto& input_tensor_b = tensor_args.input_tensors[1];
    auto& output_tensor = tensor_return_value;

    auto* src_a_buffer = input_tensor_a.buffer();
    auto* src_b_buffer = input_tensor_b.buffer();
    auto* dst_buffer = output_tensor.buffer();
    std::vector<m2::DataflowBufferSpec> dataflow_buffers;

    const NodeCoord node{0, 0};

    KernelSpec reader{
        .unique_id = READER,
        .source = KernelSpec::SourceFilePath{"kernels/dataflow/quasar_reader.cpp"},
        .compile_time_arg_bindings = {{"page_size", page_size}},
        .runtime_arguments_schema = {.named_runtime_args = {"num_pages"}},
        .dfb_bindings =
            {{.dfb_spec_name = DFB,
              .local_accessor_name = "out_dfb",
              .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {{
            .tensor_parameter_name = INPUT,
            .accessor_name = "input",  // kernel accesses as `ta::input`
        }},
        .config_spec =
            DataMovementConfiguration{
                .gen1_data_movement_config = {.processor = DataMovementProcessor::RISCV_0},
            },
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = KernelSpec::SourceFilePath{"kernels/writer.cpp"},
        .compile_time_arg_bindings = {{"page_size", page_size}},
        .runtime_arguments_schema = {.named_runtime_args = {"num_pages"}},
        .dfb_bindings =
            {{.dfb_spec_name = DFB,
              .local_accessor_name = "in_dfb",
              .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {{
            .tensor_parameter_name = OUTPUT,
            .accessor_name = "output",  // kernel accesses as `ta::output`
        }},
        .config_spec =
            DataMovementConfiguration{
                .gen1_data_movement_config = {.processor = DataMovementProcessor::RISCV_1},
            },
    };

    const tt::DataFormat src0_dfb_data_format = datatype_to_dataformat_converter(a.dtype());
    const uint32_t src0_single_tile_size = tile_size(src0_cb_data_format);
    const tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    const uint32_t num_pages = 2;

    DataflowBufferSpec in0_dfb{
        .unique_id = IN0_DFB,
        .entry_size = src0_single_tile_size,
        .num_entries = num_pages,
        .data_format_metadata = src0_cb_data_format,
    };

    ProgramSpec spec{
        .program_id = "matmul",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {dfb},
        .tensor_parameters =
            {
                {.unique_id = INPUT, .spec = input_tensor.tensor_spec()},
                {.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
            },
        .work_units = {{
            .unique_id = "main",
            .kernels = {READER, WRITER},
            .target_nodes = node,
        }},
    };

    tt::tt_metal::Program program{};
    spec.work_units = {std::move(work_unit)};
    m2::ProgramRunParams::KernelRunParams reader_params;
    reader_params.kernel_spec_name = W_READER_KERNEL;

    m2::ProgramRunParams::KernelRunParams writer_params;
    writer_params.kernel_spec_name = W_WRITER_KERNEL;

    m2::ProgramRunParams::KernelRunParams compute_params;
    compute_params.kernel_spec_name = W_COMPUTE_KERNEL;

    Program program = m2::MakeProgramFromSpec(*a.device(), spec);
    shared_variables_t shared{
        .cores = wd.cores,
        .core_group_1 = wd.core_group_1,
        .core_group_2 = wd.core_group_2,
        .num_rows_per_core_group_1 = wd.num_rows_per_core_group_1,
        .num_rows_per_core_group_2 = wd.num_rows_per_core_group_2,
        .Wt = Wt,
    };

    auto run_params = BuildRunParams(shared, a.mesh_tensor(), output.mesh_tensor());
    m2::SetProgramRunParameters(program, run_params);

    return cached_program_t{std::move(program), std::move(shared)};
}

static void override_runtime_arguments(
    cached_program_t& cached_program,
    const MatmulParams& operation_attributes,
    const MatmulInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto run_params = BuildRunParams(cached_program.shared_variables, tensor_args, tensor_return_value.mesh_tensor());
    m2::SetProgramRunParameters(cached_program.program, run_params);
}
}  // namespace ttnn::prim
