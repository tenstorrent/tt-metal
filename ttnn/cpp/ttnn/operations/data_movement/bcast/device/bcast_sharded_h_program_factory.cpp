// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_sharded_h_program_factory.hpp"

#include <cmath>
#include <filesystem>
#include <map>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <cstdint>
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace tt::constants;

ttnn::device_operation::ProgramArtifacts BcastShardedHProgramFactory::create_program_artifacts(
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
    const std::uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    const std::uint32_t NC = N * C;

    IDevice* device = a.device();

    const auto shard_spec = a.shard_spec().value();
    const auto all_cores = shard_spec.grid;
    const std::uint32_t ncores = shard_spec.num_cores();

    std::uint32_t ncores_x = device->compute_with_storage_grid_size().x;

    const auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(
        out_shard_spec.num_cores() == ncores,
        "Output tensor should have same number of cores {} as input tensor {}",
        out_shard_spec.num_cores(),
        ncores);

    const auto act_df = datatype_to_dataformat_converter(a.dtype());
    const auto b_df = datatype_to_dataformat_converter(b.dtype());
    const auto out_df = datatype_to_dataformat_converter(output.dtype());

    const std::uint32_t input_tile_size = tt::tile_size(act_df);
    const std::uint32_t input1_tile_size = tt::tile_size(b_df);
    const std::uint32_t output_tile_size = tt::tile_size(out_df);

    TT_FATAL(input_tile_size == output_tile_size, "Input and output tile size should be same");

    const std::uint32_t ntiles_along_width = std::ceil(shard_spec.shape[1] / (float)TILE_WIDTH);
    const std::uint32_t ntiles_along_height = std::ceil(shard_spec.shape[0] / (float)TILE_HEIGHT);
    const std::uint32_t num_tile_per_core = ntiles_along_width * ntiles_along_height;

    std::uint32_t Wt, Ht;
    if (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        ncores_x = all_cores.ranges().begin()->end_coord.y + 1;
        Wt = shard_spec.shape[1] / TILE_WIDTH;
        Ht = shard_spec.shape[0] / TILE_HEIGHT;
    } else if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        Wt = shard_spec.shape[1] / TILE_WIDTH;
        Ht = shard_spec.shape[0] / TILE_HEIGHT;
        TT_ASSERT(
            (shard_spec.shape[0] % (bN * TILE_HEIGHT) == 0),
            "Shard height per batch must be divisible by TILE_HEIGHT {} {} {} ",
            shard_spec.shape[0],
            bN,
            TILE_HEIGHT);
    } else {
        TT_THROW("Unsupported memory layout");
    }

    TT_ASSERT(
        (shard_spec.shape[0] % TILE_HEIGHT == 0) && (shard_spec.shape[0] % TILE_WIDTH == 0),
        "Shard shapes must be multiple of TILE_HEIGHT ");

    const std::uint32_t aligned_input_tile_nbytes =
        round_up_to_mul32(input_tile_size);  // will have issue if the page is not multiple of 32

    const std::uint32_t num_input_tiles = (b.padded_shape()[-1] * output.element_size() + TILE_HW - 1) / TILE_HW;

    // ---- Resource names (function-local: avoids unity-build anon-namespace collisions) ----
    const DFBSpecName IN0{"in0"};  // legacy CB c_0 (src0 / input_a) — borrowed
    const DFBSpecName IN1{"in1"};  // legacy CB c_1 (src1 / input_b)
    const DFBSpecName OUT{"out"};  // legacy CB c_16 (output) — borrowed, self-loop
    const TensorParamName INPUT_A{"input_a"};
    const TensorParamName INPUT_B{"input_b"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName COMPUTE{"compute"};

    // ---- DataflowBuffers ----
    // c_0 borrows the resident input_a shard; the reader push_backs it (no NoC read) to signal the
    // resident tiles, compute consumes → ordinary 1P+1C.
    DataflowBufferSpec in0_dfb{
        .unique_id = IN0,
        .entry_size = aligned_input_tile_nbytes,
        .num_entries = num_tile_per_core,
        .data_format_metadata = act_df,
        .borrowed_from = INPUT_A,
    };
    DataflowBufferSpec in1_dfb{
        .unique_id = IN1,
        .entry_size = input1_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = b_df,
    };
    // c_16 borrows the resident output shard; compute produces into it and nothing drains it
    // (no writer kernel) → self-loop (compute bound PRODUCER + CONSUMER).
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = aligned_input_tile_nbytes,
        .num_entries = num_tile_per_core,
        .data_format_metadata = out_df,
        .borrowed_from = OUTPUT,
    };

    // ---- Tensor parameters (input_a / output back borrowed DFBs; input_b read via accessor) ----
    TensorParameter input_a_param{.unique_id = INPUT_A, .spec = a.tensor_spec()};
    TensorParameter input_b_param{.unique_id = INPUT_B, .spec = b.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Compute defines ----
    Table<std::string, std::string> compute_defines(
        bcast_op_utils::get_defines(BcastOpDim::H, operation_attributes.math_op));

    // ---- Kernels (reader + compute; no writer — output is resident) ----
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/reader_bcast_h_sharded.cpp"),
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_B, .accessor_name = "src1"}},
        // Legacy arg names on the reader side: the kernel reads index 4 as "NC" (fed Ht_per_core) and
        // index 5 as "batch_offset" (fed tile_offset). Names preserve the kernel-side identifiers.
        .runtime_arg_schema = {.runtime_arg_names = {"Ht", "Wt", "offset", "NC", "batch_offset"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    ComputeHardwareConfig compute_hw = ComputeGen1Config{};  // legacy ComputeConfigDescriptor{} defaults
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            std::filesystem::path("ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_h.cpp"),
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .runtime_arg_schema = {.runtime_arg_names = {"B", "Ht", "Wt"}},
        .hw_config = compute_hw,
    };

    // ---- Per-core runtime args (over the shard grid) ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_args{.kernel = READER};
    KernelRunArgs compute_args{.kernel = COMPUTE};

    const std::uint32_t ncores_y = ncores / ncores_x;
    for (std::uint32_t i = 0; i < ncores; i++) {
        CoreCoord core;
        std::uint32_t offset = 0;
        std::uint32_t Ht_per_core = 0;
        if (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            core = {i / ncores_x, i % ncores_x};
            Ht_per_core = Wt * Ht;
            if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
                offset = Wt * (i / ncores_x) + Wt * ncores_y * ((i % ncores_x) / (ncores_x / bN));
            } else {
                offset = Wt * (i % ncores_x) + Wt * ncores_x * ((i / ncores_x) / (ncores_y / bN));
            }
        } else if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            core = {i % ncores_x, i / ncores_x};
            if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
                offset = Wt * (core.x + core.y * ncores_x);
            } else {
                offset = Wt * (ncores_y * core.x + core.y);
                if (core.y == ncores_y) {
                    offset = Wt * (ncores_y * ncores_x + core.x);
                }
            }
            Ht_per_core = Ht / bN;
        }
        const std::uint32_t tile_offset = Wt * ncores;  // used in multi batch weight for block sharded

        AddRuntimeArgsForNode(
            reader_args.runtime_arg_values,
            core,
            {{"Ht", Ht}, {"Wt", Wt}, {"offset", offset}, {"NC", Ht_per_core}, {"batch_offset", tile_offset}});

        AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"B", NC}, {"Ht", Ht}, {"Wt", Wt}});
    }

    ProgramSpec spec{
        .name = "bcast_sharded_h",
        .kernels = {reader, compute},
        .dataflow_buffers = {in0_dfb, in1_dfb, out_dfb},
        .tensor_parameters = {input_a_param, input_b_param, output_param},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, COMPUTE}, .target_nodes = all_cores}},
    };

    run_args.kernel_run_args = {std::move(reader_args), std::move(compute_args)};
    run_args.tensor_args = {{INPUT_A, a_mt}, {INPUT_B, b_mt}, {OUTPUT, out_mt}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
