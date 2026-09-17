// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_rm_program_factory.hpp"

#include <bit>
#include <cstdint>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts FillRMProgramFactory::create_program_artifacts(
    const FillRmParams& operation_attributes, const FillRmInputs& tensor_args, Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;
    const auto& output_mesh_tensor = output.mesh_tensor();

    const std::uint32_t N = operation_attributes.N;
    const std::uint32_t C = operation_attributes.C;
    const std::uint32_t H = operation_attributes.H;
    const std::uint32_t W = operation_attributes.W;
    const std::uint32_t hFill = operation_attributes.hFill;
    const std::uint32_t wFill = operation_attributes.wFill;
    const float val_hi = operation_attributes.val_hi;
    const float val_lo = operation_attributes.val_lo;

    const NodeCoord node{0, 0};

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const std::uint32_t single_tile_size = tt::tile_size(cb_data_format);

    const std::uint32_t num_cb_tiles = 16;
    TT_FATAL(
        W < 1024 * num_cb_tiles,
        "Width (W) must be less than {} for kernel simplification. Got W={}, num_cb_tiles={}",
        1024 * num_cb_tiles,
        W,
        num_cb_tiles);

    // Resource names
    const KernelSpecName READER{"reader"};
    const ScratchpadSpecName IN0{"in0"};
    const ScratchpadSpecName IN1{"in1"};
    const TensorParamName OUT{"out"};

    // Two single-toucher scratchpads: the reader fills each and uses it as the NoC write source.
    // (Formerly self-loop DFBs; a single DM kernel filled and drained each, so the FIFO machinery
    // synchronized nothing. Converted to Scratchpad for Quasar, which rejects DM self-loop DFBs.)
    ScratchpadSpec sp_in0{
        .unique_id = IN0,
        .size_per_node = single_tile_size * num_cb_tiles,
    };
    ScratchpadSpec sp_in1{
        .unique_id = IN1,
        .size_per_node = single_tile_size * num_cb_tiles,
    };

    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/fill_rm/device/kernels/dataflow/fill_rm_interleaved.cpp",
        .scratchpad_bindings =
            {
                ScratchpadBinding{.scratchpad_spec_name = IN0, .accessor_name = "in0"},
                ScratchpadBinding{.scratchpad_spec_name = IN1, .accessor_name = "in1"},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUT, .accessor_name = "out"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"NC", "H", "W", "fillH", "fillW", "val_hi", "val_lo"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(output.device()->arch()),
    };

    ProgramSpec spec{
        .name = "fill_rm",
        .kernels = {reader},
        .scratchpads = {sp_in0, sp_in1},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = OUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{.name = "main", .kernels = {READER}, .target_nodes = node},
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node,
                {
                    {"NC", std::uint32_t(N * C)},
                    {"H", std::uint32_t(H)},
                    {"W", std::uint32_t(W)},
                    {"fillH", std::uint32_t(hFill)},
                    {"fillW", std::uint32_t(wFill)},
                    {"val_hi", std::uint32_t(std::bit_cast<std::uint16_t>(bfloat16(val_hi)))},
                    {"val_lo", std::uint32_t(std::bit_cast<std::uint16_t>(bfloat16(val_lo)))},
                }),
        },
    };
    run_args.tensor_args = {
        {OUT, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
