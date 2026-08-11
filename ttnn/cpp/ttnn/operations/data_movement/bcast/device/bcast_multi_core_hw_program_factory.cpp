// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_multi_core_hw_program_factory.hpp"

#include <filesystem>
#include <map>
#include <optional>
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

ttnn::device_operation::ProgramArtifacts BcastMultiCoreHWProgramFactory::create_program_artifacts(
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
    const std::uint32_t NC = N * C;

    const std::uint32_t Wt = W / TILE_WIDTH;
    const std::uint32_t Ht = H / TILE_HEIGHT;
    const std::uint32_t HtWt = Ht * Wt;

    const std::uint32_t num_tensor_tiles = NC * Ht * Wt;

    const std::uint32_t bnc1 = (bN * bC == 1) ? 1 : 0;

    IDevice* device = a.device();

    std::optional<ShardSpec> shard_spec = std::nullopt;
    const bool src0_sharded = a.memory_config().is_sharded();
    const bool output_sharded = output.memory_config().is_sharded();
    if (src0_sharded) {
        shard_spec = a.shard_spec().value();
    } else if (output_sharded) {
        shard_spec = output.shard_spec().value();
    }

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

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);
    (void)num_cores;

    const std::uint32_t num_input_tiles = 2;
    std::uint32_t num_tiles_per_shard = 0;
    if (shard_spec.has_value()) {
        num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        num_tiles_per_core_group_1 = num_tiles_per_shard;
        num_tiles_per_core_group_2 = 0;
        all_cores = shard_spec.value().grid;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
    }

    // Legacy CB (c_0 / c_16) sizing: the HW factory used tt::tile_size directly for the page size in both
    // interleaved and sharded configs (unlike the ShardedH factory's round_up_to_mul32) — preserved verbatim.
    const std::uint32_t num_input_tiles_cb0 = src0_sharded ? num_tiles_per_shard : num_input_tiles;
    const std::uint32_t num_output_tiles = output_sharded ? num_tiles_per_shard : 2;

    // Sharding is all-or-nothing for HW (validate forces in0 and output to the same layout), so the
    // borrowed DFBs live on the shard grid; kernels run there. Interleaved runs on the full grid with idle
    // cores zero-filled, exactly as legacy.
    const bool is_sharded = shard_spec.has_value();
    const CoreRangeSet target_nodes = is_sharded ? all_cores : CoreRangeSet(all_device_cores);

    // ---- Resource names (function-local: avoids unity-build anon-namespace collisions) ----
    const DFBSpecName IN0{"in0"};  // legacy CB c_0 (src0 / input_a) — borrowed when src0 is sharded
    const DFBSpecName IN1{"in1"};  // legacy CB c_1 (src1 / input_b)
    const DFBSpecName OUT{"out"};  // legacy CB c_16 (output) — borrowed when output is sharded
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
        .num_entries = num_input_tiles_cb0,
        .data_format_metadata = src0_cb_data_format,
    };
    if (src0_sharded) {
        // c_0 borrows the resident input_a shard; the reader push_backs it (no NoC read) to signal the
        // resident tiles, compute consumes → ordinary 1P+1C.
        in0_dfb.borrowed_from = INPUT_A;
    }
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
    if (output_sharded) {
        // c_16 borrows the resident output shard; compute produces into it, the (donor) writer wait_fronts
        // it as a sync barrier → 1P+1C (HW always binds the writer, so c_16 is never a self-loop).
        out_dfb.borrowed_from = OUTPUT;
    }

    // ---- Tensor parameters ----
    TensorParameter input_a_param{.unique_id = INPUT_A, .spec = a.tensor_spec()};
    TensorParameter input_b_param{.unique_id = INPUT_B, .spec = b.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Defines ----
    Table<std::string, std::string> reader_defines;
    if (bnc1) {
        reader_defines["BCAST_SCALAR"] = "1";
    }
    if (src0_sharded) {
        reader_defines["IN0_SHARDED"] = "1";
    }

    Table<std::string, std::string> compute_defines(
        bcast_op_utils::get_defines(BcastOpDim::HW, operation_attributes.math_op));
    if (bnc1) {
        compute_defines["BCAST_SCALAR"] = "1";
    }

    Table<std::string, std::string> writer_defines;
    if (output_sharded) {
        writer_defines["OUT_SHARDED"] = "1";
    }

    // ---- Conditional tensor bindings ----
    // src0 is read via tensor::src0 only in the interleaved config; when sharded it is resident and backs
    // the borrowed c_0. src1 is always interleaved-read. Mirrors the reader's IN0_SHARDED #ifdef gate.
    Group<TensorBinding> reader_tensor_bindings;
    if (!src0_sharded) {
        reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = INPUT_A, .accessor_name = "src0"});
    }
    reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = INPUT_B, .accessor_name = "src1"});

    // The (reused) writer fork reads the output through tensor::dst only in the interleaved config; when
    // sharded it just wait_fronts the resident borrowed c_16 (its OUT_SHARDED #ifdef gate).
    Group<TensorBinding> writer_tensor_bindings;
    if (!output_sharded) {
        writer_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"});
    }

    // ---- Kernels ----
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/"
                                        "reader_bcast_hw_interleaved_partitioned.cpp"),
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = reader_tensor_bindings,
        .runtime_arg_schema =
            {.runtime_arg_names = {"num_tiles", "HtWt", "base_start_id_HtWt", "curr_id_from_base", "bcast_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // Rung-1 reuse of the shared eltwise/unary writer's Metal 2.0 fork (do not edit — it already has
    // consumers). Its interface (dfb::out CONSUMER, tensor::dst, args num_pages/start_id, OUT_SHARDED) is
    // the constraint the bindings below are built against.
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_metal2.cpp"),
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = writer_tensor_bindings,
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    ComputeHardwareConfig compute_hw = ComputeGen1Config{};  // legacy ComputeConfigDescriptor{} defaults
    KernelSpec compute{
        .unique_id = COMPUTE,
        // Rung-2 fork of the lent bcast_hw.cpp compute kernel (rotate_half still binds the legacy original).
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_hw_metal2.cpp"),
        // Compute default opt_level is O3 in legacy but O2 in Metal 2.0 — set it explicitly to preserve.
        .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.runtime_arg_names = {"B", "Ht", "Wt"}},
        .hw_config = compute_hw,
    };

    // ---- Per-core runtime args ----
    // Interleaved: iterate the full grid; active cores (work-split groups) get real args, idle cores get
    // no-op args (they are part of the all_device_cores WorkUnit, as in legacy). Sharded: only shard-grid
    // cores are in the WorkUnit, so non-shard cores are skipped entirely.
    ProgramRunArgs run_args;
    KernelRunArgs reader_args{.kernel = READER};
    KernelRunArgs writer_args{.kernel = WRITER};
    KernelRunArgs compute_args{.kernel = COMPUTE};

    for (std::uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        std::uint32_t num_tensor_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            if (!is_sharded) {
                AddRuntimeArgsForNode(
                    reader_args.runtime_arg_values,
                    core,
                    {{"num_tiles", 0u},
                     {"HtWt", 0u},
                     {"base_start_id_HtWt", 0u},
                     {"curr_id_from_base", 0u},
                     {"bcast_id", 0u}});
                AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"B", 1u}, {"Ht", 1u}, {"Wt", 0u}});
                AddRuntimeArgsForNode(writer_args.runtime_arg_values, core, {{"num_pages", 0u}, {"start_id", 0u}});
            }
            continue;
        }

        AddRuntimeArgsForNode(
            reader_args.runtime_arg_values,
            core,
            {{"num_tiles", num_tensor_tiles_per_core},
             {"HtWt", HtWt},
             {"base_start_id_HtWt", num_tiles_read / HtWt * HtWt},
             {"curr_id_from_base", num_tiles_read % HtWt},
             {"bcast_id", bnc1 ? 0u : num_tiles_read / HtWt}});

        AddRuntimeArgsForNode(
            compute_args.runtime_arg_values, core, {{"B", 1u}, {"Ht", 1u}, {"Wt", num_tensor_tiles_per_core}});

        AddRuntimeArgsForNode(
            writer_args.runtime_arg_values,
            core,
            {{"num_pages", num_tensor_tiles_per_core}, {"start_id", num_tiles_read}});

        num_tiles_read += num_tensor_tiles_per_core;
    }

    ProgramSpec spec{
        .name = "bcast_multi_core_hw",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {in0_dfb, in1_dfb, out_dfb},
        .tensor_parameters = {input_a_param, input_b_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = target_nodes}},
    };

    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args), std::move(compute_args)};
    run_args.tensor_args = {{INPUT_A, a_mt}, {INPUT_B, b_mt}, {OUTPUT, out_mt}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
