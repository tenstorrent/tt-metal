// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_multi_core_w_program_factory.hpp"

#include <filesystem>
#include <map>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <cstdint>
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace tt::constants;

ttnn::device_operation::ProgramArtifacts BcastMultiCoreWProgramFactory::create_program_artifacts(
    const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value) {
    const Tensor& a = tensor_args.input_a;
    const Tensor& b = tensor_args.input_b;
    Tensor& output = tensor_return_value;

    const auto& a_mt = a.mesh_tensor();
    const auto& b_mt = b.mesh_tensor();
    const auto& out_mt = output.mesh_tensor();

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    const std::uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    const std::uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    const std::uint32_t H = ashape[-2];
    const std::uint32_t W = ashape[-1];
    const std::uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    const std::uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    const std::uint32_t bH = bshape[-2];
    const std::uint32_t bW = bshape[-1];
    const std::uint32_t NC = N * C;

    const std::uint32_t Wt = W / TILE_WIDTH;
    const std::uint32_t Ht = H / TILE_HEIGHT;

    const std::uint32_t num_btensor_tiles = NC * bH * bW / TILE_HW;
    (void)num_btensor_tiles;  // legacy dead reader arg (idx 7); not carried into the port
    const std::uint32_t bnc1 = (bN * bC == 1) ? 1 : 0;

    IDevice* device = a.device();

    const tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat src1_cb_data_format = datatype_to_dataformat_converter(b.dtype());
    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    const std::uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    const std::uint32_t src1_single_tile_size = tt::tile_size(src1_cb_data_format);
    const std::uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const std::uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const std::uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const std::uint32_t num_cores_total = num_cores_x * num_cores_y;
    const auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    constexpr bool row_major = false;
    const auto [num_cores, all_cores, core_group_1, core_group_2, Wt_per_core_group_1, Wt_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, Wt, row_major);
    (void)num_cores;
    (void)all_cores;

    const auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    const std::uint32_t num_input_tiles = 2;
    const std::uint32_t num_output_tiles = 2;

    // ---- Resource names (function-local: avoids unity-build anon-namespace collisions) ----
    const DFBSpecName IN0{"in0"};  // legacy CB c_0 (src0 / input_a)
    const DFBSpecName IN1{"in1"};  // legacy CB c_1 (src1 / input_b)
    const DFBSpecName OUT{"out"};  // legacy CB c_16 (output)
    const TensorParamName INPUT_A{"input_a"};
    const TensorParamName INPUT_B{"input_b"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // ---- DataflowBuffers (legacy CBs c_0 / c_1 / c_16) ----
    DataflowBufferSpec in0_dfb{
        .unique_id = IN0,
        .entry_size = src0_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = src0_cb_data_format,
    };
    DataflowBufferSpec in1_dfb{
        .unique_id = IN1,
        .entry_size = src1_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = src1_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = dst_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = dst_cb_data_format,
    };

    // ---- Tensor parameters ----
    TensorParameter input_a_param{.unique_id = INPUT_A, .spec = a.tensor_spec()};
    TensorParameter input_b_param{.unique_id = INPUT_B, .spec = b.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Compute defines (bcast math op + broadcast dim) ----
    Table<std::string, std::string> compute_defines(
        bcast_op_utils::get_defines(BcastOpDim::W, operation_attributes.math_op));

    // ---- Kernels ----
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/"
                                        "reader_bcast_w_interleaved_input_cols_partitioned.cpp"),
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = INPUT_A, .accessor_name = "src0"},
             TensorBinding{.tensor_parameter_name = INPUT_B, .accessor_name = "src1"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"src0_num_tiles", "NCHtWt", "NC", "Ht", "Wt", "nc1", "start_id", "HtWt", "Wt_skip"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/"
                                        "writer_unary_interleaved_input_cols_batched.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"Ht", "Wt", "Wt_read", "Wt_skip", "NC", "HtWt"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    ComputeHardwareConfig compute_hw = ComputeGen1Config{};  // legacy ComputeConfigDescriptor{} defaults
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            std::filesystem::path("ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_w.cpp"),
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.runtime_arg_names = {"B", "Ht", "Wt"}},
        .hw_config = compute_hw,
    };

    // ---- Per-core runtime args (all device cores; idle cores get zeros, as in legacy) ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_args{.kernel = READER};
    KernelRunArgs writer_args{.kernel = WRITER};
    KernelRunArgs compute_args{.kernel = COMPUTE};

    for (std::uint32_t i = 0, num_Wtiles_read = 0; i < num_cores_total; i++) {
        const CoreCoord& core = cores.at(i);
        std::uint32_t Wt_per_core;
        if (core_group_1.contains(core)) {
            Wt_per_core = Wt_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            Wt_per_core = Wt_per_core_group_2;
        } else {
            AddRuntimeArgsForNode(
                reader_args.runtime_arg_values,
                core,
                {{"src0_num_tiles", 0u},
                 {"NCHtWt", 0u},
                 {"NC", 0u},
                 {"Ht", 0u},
                 {"Wt", 0u},
                 {"nc1", 0u},
                 {"start_id", 0u},
                 {"HtWt", 0u},
                 {"Wt_skip", 0u}});
            AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"B", 0u}, {"Ht", 0u}, {"Wt", 0u}});
            AddRuntimeArgsForNode(
                writer_args.runtime_arg_values,
                core,
                {{"Ht", 0u}, {"Wt", 0u}, {"Wt_read", 0u}, {"Wt_skip", 0u}, {"NC", 0u}, {"HtWt", 0u}});
            continue;
        }
        const std::uint32_t num_tensor_tiles_per_core = NC * Ht * Wt_per_core;
        const std::uint32_t Wt_skip = Wt - Wt_per_core;

        AddRuntimeArgsForNode(
            reader_args.runtime_arg_values,
            core,
            {{"src0_num_tiles", num_tensor_tiles_per_core},
             {"NCHtWt", num_tensor_tiles_per_core},
             {"NC", NC},
             {"Ht", Ht},
             {"Wt", Wt_per_core},
             {"nc1", bnc1},
             {"start_id", num_Wtiles_read},
             {"HtWt", Ht * Wt},
             {"Wt_skip", Wt_skip}});

        AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"B", NC}, {"Ht", Ht}, {"Wt", Wt_per_core}});

        AddRuntimeArgsForNode(
            writer_args.runtime_arg_values,
            core,
            {{"Ht", Ht},
             {"Wt", Wt_per_core},
             {"Wt_read", num_Wtiles_read},
             {"Wt_skip", Wt_skip},
             {"NC", NC},
             {"HtWt", Ht * Wt}});

        num_Wtiles_read += Wt_per_core;
    }

    ProgramSpec spec{
        .name = "bcast_multi_core_w",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {in0_dfb, in1_dfb, out_dfb},
        .tensor_parameters = {input_a_param, input_b_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_device_cores}},
    };

    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args), std::move(compute_args)};
    run_args.tensor_args = {{INPUT_A, a_mt}, {INPUT_B, b_mt}, {OUTPUT, out_mt}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
