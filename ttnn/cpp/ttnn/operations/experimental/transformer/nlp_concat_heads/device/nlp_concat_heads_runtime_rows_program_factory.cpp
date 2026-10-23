// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "nlp_concat_heads_runtime_rows_program_factory.hpp"
#include "nlp_concat_heads_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
// The one common runtime word of the program: input rows in tiles.
constexpr const char* kLengthArg = "h_tiles";
}  // namespace

ttnn::device_operation::ProgramArtifacts NLPConcatHeadsRuntimeRowsProgramFactory::create_program_artifacts(
    const NlpConcatHeadsParams& /*operation_attributes*/, const Tensor& input, Tensor& output) {
    const auto& a = input;
    const auto& ashape = a.padded_shape();
    TT_FATAL(!a.is_sharded() && !output.is_sharded(), "runtime_rows requires an interleaved input and output");

    // The Metal 2.0 binding layer works with the Metalium tensor type; extract once at entry.
    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(data_format);

    CoreCoord compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_tensor_tiles = ashape[1] * ashape[3] / TILE_WIDTH;
    uint32_t in0_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t in0_w_tiles = ashape[3] / TILE_WIDTH;    // head_dim
    uint32_t in0_c = per_tensor_tiles / in0_w_tiles;  // num_heads

    // The full grid: each core derives its (row, head) block range on device from the common row count
    // (split by head too when there are fewer tile rows than cores).
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cores = num_cores_x * num_cores_y;
    const CoreRangeSet all_cores(CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1}));
    const bool head_parallel_ok = in0_c > 1;

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName IN0_DFB{"in0"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    // Reader and writer share the split inputs: static per-core index, common row count.
    const KernelSpec::CompileTimeArgs split_compile_time_args = {
        {"in0_w_tiles", in0_w_tiles},
        {"in0_c", in0_c},
        {"batch", ashape[0]},
        {"num_cores", num_cores},
        {"head_parallel_ok", static_cast<uint32_t>(head_parallel_ok)},
    };
    const KernelSpec::RuntimeArgSchema split_runtime_arg_schema = {
        .runtime_arg_names = {"core_idx"}, .common_runtime_arg_names = {kLengthArg}};

    KernelSpec reader_spec{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_concat_heads_runtime_rows.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0_DFB,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = INPUT,
            .accessor_name = "src",
        }},
        .compile_time_args = split_compile_time_args,
        .runtime_arg_schema = split_runtime_arg_schema,
        .hw_config = create_reader_datamovement_config(),
    };
    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
            "writer_tm_tile_layout_nlp_concat_heads_runtime_rows.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = OUTPUT,
            .accessor_name = "dst",
        }},
        .compile_time_args = split_compile_time_args,
        .runtime_arg_schema = split_runtime_arg_schema,
        .hw_config = create_writer_datamovement_config(),
    };

    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0_DFB,
        .entry_size = single_tile_size,
        .num_entries = per_tensor_tiles * 2,  // double buffer
        .data_format_metadata = data_format,
    });

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    reader_run_args.common_runtime_arg_values = {{kLengthArg, in0_h_tiles}};
    writer_run_args.common_runtime_arg_values = {{kLengthArg, in0_h_tiles}};
    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, /*row_wise=*/true);
    for (uint32_t i = 0; i < cores.size(); ++i) {
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, cores[i], {{"core_idx", i}});
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, cores[i], {{"core_idx", i}});
    }

    // The hash drops the row count, so the parameters accept any height.
    const TensorSpecRelaxations rows_vary{.dynamic_tensor_shape = true};
    ProgramSpec spec{
        .name = "nlp_concat_heads",
        .kernels = {std::move(reader_spec), std::move(writer_spec)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec(), .relaxations = rows_vary},
                TensorParameter{
                    .unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec(), .relaxations = rows_vary},
            },
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER},
            .target_nodes = all_cores,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, TensorArgument{input_mesh_tensor}},
        {OUTPUT, TensorArgument{output_mesh_tensor}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

// A hit re-binds the two tensors and rewrites the common row count the kernels derive their block
// ranges from (the hash leaves the row count out).
tt::tt_metal::experimental::ProgramRunArgs NLPConcatHeadsRuntimeRowsProgramFactory::override_runtime_arguments(
    const NlpConcatHeadsParams& /*operation_attributes*/,
    const Tensor& input,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const uint32_t h_tiles = input.padded_shape()[2] / TILE_HEIGHT;
    ProgramRunArgs params;
    params.kernel_run_args = {
        KernelRunArgs{.kernel = KernelSpecName{"reader"}, .common_runtime_arg_values = {{kLengthArg, h_tiles}}},
        KernelRunArgs{.kernel = KernelSpecName{"writer"}, .common_runtime_arg_values = {{kLengthArg, h_tiles}}},
    };
    params.tensor_args = {
        {TensorParamName{"input"}, TensorArgument{input.mesh_tensor()}},
        {TensorParamName{"output"}, TensorArgument{output.mesh_tensor()}},
    };
    return params;
}

}  // namespace ttnn::experimental::prim
