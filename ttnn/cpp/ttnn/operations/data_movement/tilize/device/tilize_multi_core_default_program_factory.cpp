// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_default_program_factory.hpp"

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts TilizeMultiCoreDefaultProgramFactory::create_program_artifacts(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    const auto& input_tensor = a.mesh_tensor();
    const auto& output_tensor = output.mesh_tensor();

    // Metal 2.0 named resource handles.
    const DFBSpecName SRC_DFB{"src"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);
    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto logical_shape = a.logical_shape();
    uint32_t logical_width = logical_shape[-1];
    uint32_t ntiles_per_block = tt::div_up(logical_width, TILE_WIDTH);
    uint32_t ntiles = dst_buffer->num_pages();
    uint32_t nblocks = tt::div_up(ntiles, ntiles_per_block);
    auto* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, nblocks);

    // ------------------------------------------------------------------------
    // Dataflow buffers (formerly CB c_0 / c_16). One DFB per legacy CB.
    // ------------------------------------------------------------------------
    DataflowBufferSpec src_dfb_spec{
        .unique_id = SRC_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = ntiles_per_block,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = ntiles_per_block,
        .data_format_metadata = output_cb_data_format,
    };

    // ------------------------------------------------------------------------
    // Tensor parameters (Case 1): legacy Buffer* RTA + TensorAccessorArgs collapse to a binding.
    // ------------------------------------------------------------------------
    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_tensor.tensor_spec()};

    // ------------------------------------------------------------------------
    // Reader compile-time args: legacy CT arg 0 (aligned_page_size) was unused by the kernel and is
    // dropped. num_pages_in_row / size_of_valid_data_in_last_page_in_row describe the (optionally
    // ND-sharded) input row layout.
    // ------------------------------------------------------------------------
    uint32_t page_size = src0_buffer->page_size();
    uint32_t num_pages_in_row = 1;
    uint32_t size_of_valid_data_in_last_page_in_row = page_size;
    if (a.is_sharded()) {
        uint32_t shard_width =
            a.shard_spec().has_value() ? a.shard_spec().value().shape[1] : a.nd_shard_spec().value().shard_shape[-1];
        num_pages_in_row = tt::div_up(logical_width, shard_width);
        uint32_t padding_size = (num_pages_in_row * page_size) - (a.logical_shape()[-1] * a.element_size());
        size_of_valid_data_in_last_page_in_row = page_size - padding_size;
    }

    // ------------------------------------------------------------------------
    // Kernels.
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "reader_unary_stick_layout_split_rows_multicore.cpp"},
        .dfb_bindings = {ProducerOf(SRC_DFB, "in")},
        .tensor_bindings = {{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .compile_time_args =
            {{"num_pages_in_row", num_pages_in_row},
             {"size_of_valid_data_in_last_page_in_row", size_of_valid_data_in_last_page_in_row}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_rows", "num_tiles_per_block", "block_width_size", "num_full_blocks_in_row", "start_page_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_metal2.cpp"},
        .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    ComputeHardwareConfig::UnpackToDestModes unpack_to_dest_modes;
    if (fp32_llk_acc && input_cb_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_modes.insert({SRC_DFB, UnpackToDestMode::UnpackToDestFp32});
    }

    // The legacy factory created a dedicated cliff compute kernel with a different compile-time
    // block count. That maps 1:1 to two compute KernelSpecs of the same source (full + cliff),
    // each placed in its own WorkUnitSpec, both binding the shared src/out DFBs. per_core_block_cnt
    // stays a compile-time arg (preserves loop unrolling).
    const KernelSpecName COMPUTE_FULL_KERNEL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF_KERNEL{"compute_cliff"};
    bool has_cliff = !core_range_cliff.ranges().empty();

    auto make_compute = [&](const KernelSpecName& id, uint32_t nblocks_this_group) {
        return KernelSpec{
            .unique_id = id,
            .source =
                std::filesystem::path{
                    "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_compute_metal2.cpp"},
            .dfb_bindings = {ConsumerOf(SRC_DFB, "in"), ProducerOf(OUT_DFB, "out")},
            .compile_time_args =
                {{"per_core_block_cnt", nblocks_this_group}, {"per_core_block_tile_cnt", ntiles_per_block}},
            .hw_config =
                ComputeHardwareConfig{
                    .fp32_dest_acc_en = fp32_llk_acc,
                    .unpack_to_dest_mode = unpack_to_dest_modes,
                },
        };
    };

    std::vector<KernelSpec> kernels = {reader_spec, writer_spec, make_compute(COMPUTE_FULL_KERNEL, nblocks_per_core)};
    if (has_cliff) {
        kernels.push_back(make_compute(COMPUTE_CLIFF_KERNEL, nblocks_per_core_cliff));
    }

    // ------------------------------------------------------------------------
    // 1D distribution of blocks across cores. Reader/writer per-core args (start ids, page counts)
    // were already runtime in the legacy factory and remain per-node runtime args here.
    // ------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    reader_run.runtime_arg_values.reserve(ncores);
    writer_run.runtime_arg_values.reserve(ncores);

    uint32_t ncores_full = ncores - has_cliff;
    uint32_t tile_start_id = 0;
    uint32_t page_start_id = 0;
    const auto& cores = corerange_to_cores(available_grid);

    auto emit_core = [&](const CoreCoord& core, uint32_t nblocks_this_core) {
        const NodeCoord node = core;
        reader_run.runtime_arg_values.push_back(
            {node,
             {{"num_rows", nblocks_this_core * TILE_HEIGHT},
              {"num_tiles_per_block", ntiles_per_block},
              {"block_width_size", page_size},
              {"num_full_blocks_in_row", 1u},
              {"start_page_id", page_start_id}}});
        writer_run.runtime_arg_values.push_back(
            {node, {{"num_pages", ntiles_per_block * nblocks_this_core}, {"start_id", tile_start_id}}});

        tile_start_id += ntiles_per_block * nblocks_this_core;
        page_start_id += TILE_HEIGHT * nblocks_this_core * num_pages_in_row;
    };

    for (uint32_t i = 0; i < ncores_full; ++i) {
        emit_core(cores[i], nblocks_per_core);
    }
    if (has_cliff) {
        emit_core(cores[ncores_full], nblocks_per_core_cliff);
    }

    std::vector<WorkUnitSpec> work_units;
    work_units.push_back(WorkUnitSpec{
        .name = "tilize_multi_core_default_full",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_FULL_KERNEL},
        .target_nodes = core_range,
    });
    if (has_cliff) {
        work_units.push_back(WorkUnitSpec{
            .name = "tilize_multi_core_default_cliff",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_CLIFF_KERNEL},
            .target_nodes = core_range_cliff,
        });
    }

    ProgramSpec spec{
        .name = "tilize_multi_core_default",
        .kernels = std::move(kernels),
        .dataflow_buffers = {src_dfb_spec, out_dfb_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor)}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output_tensor)}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
